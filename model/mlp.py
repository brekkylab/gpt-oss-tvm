from math import prod
import numpy as np

from mlc_llm.op.moe_misc import gating_softmax_topk
from tvm import relax
from tvm.relax.frontend import nn
from tvm.script import tir as T

from .config import GPTOssConfig

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def swiglu(x: nn.Tensor, alpha: float = 1.702, limit: float = 7.0, out_dtype: str = "bfloat16") -> nn.Tensor:
    shape = x.shape
    batch_shape = shape[:-1]
    last_dim = shape[-1]
    d = last_dim // 2

    # reshape: (..., 2 * d) -> (..., d, 2)
    x_reshaped = nn.op.reshape(x, (*batch_shape, d, 2))

    chunks = relax.op.split(x_reshaped._expr, indices_or_sections=2, axis=-1)
    chunk_glu = relax.TupleGetItem(chunks, 0)
    chunk_linear = relax.TupleGetItem(chunks, 1)

    # squeeze: (..., d, 1) -> (..., d)
    x_glu_expr = relax.op.squeeze(chunk_glu, axis=-1)
    x_glu_expr = relax.op.astype(x_glu_expr, dtype=out_dtype)
    x_linear_expr = relax.op.squeeze(chunk_linear, axis=-1)
    x_linear_expr = relax.op.astype(x_linear_expr, dtype=out_dtype)

    alpha = nn.op.wrap_nested(relax.const(alpha, dtype=out_dtype), name="alpha_const")
    limit_expr = relax.const(limit, dtype=out_dtype)

    x_glu = nn.op.wrap_nested(relax.op.minimum(x_glu_expr, limit_expr), name="x_glu_clip")
    x_linear = nn.op.wrap_nested(
        relax.op.clip(x_linear_expr, min=-limit, max=limit),  # type: ignore
        name="x_linear_clip",
    )

    gating = x_glu * nn.op.sigmoid(alpha * x_glu)
    output = gating * (x_linear + 1.0)

    return output


class MLPBlock(nn.Module):
    def __init__(
        self,
        config: GPTOssConfig,
        rms_norm_eps: float = 1e-5,
        dtype: str | None = None,
    ):
        """this block has the inner-dequantization of MXFP4 format weights"""

        assert config
        if dtype is None:
            dtype = config.dtype  # default is "bfloat16"
        self.dtype = dtype
        self.num_experts = config.num_experts
        self.experts_per_token = config.num_experts_per_tok
        self.swiglu_limit = config.swiglu_limit
        self.world_size = 1  # > 1 for the distributed training
        self.hidden_size = config.hidden_size
        self.norm = nn.RMSNorm(
            config.hidden_size,
            axes=-1,
            epsilon=rms_norm_eps,
            bias=False,
            dtype=dtype,
        )
        self.gate = nn.Linear(
            config.hidden_size,
            self.num_experts,
            dtype=dtype,
        )
        assert config.intermediate_size % self.world_size == 0

        # for MXFP4 dequantization
        self.mxfp4_dtype = "uint8"
        self.mxfp4_group_size = 90
        self.mxfp4_block_length = 16  # bytes_per_block

        self.mlp1_weight_blocks = nn.Parameter(
            (
                self.num_experts,
                config.intermediate_size * 2 // self.world_size,
                self.mxfp4_group_size,
                self.mxfp4_block_length,
            ),
            dtype=self.mxfp4_dtype,
        )
        self.mlp1_weight_scales = nn.Parameter(
            (
                self.num_experts,
                config.intermediate_size * 2 // self.world_size,
                self.mxfp4_group_size,
            ),
            dtype=self.mxfp4_dtype,
        )
        self.mlp2_weight_blocks = nn.Parameter(
            (
                self.num_experts,
                config.hidden_size,
                self.mxfp4_group_size,
                self.mxfp4_block_length,
            ),
            dtype=self.mxfp4_dtype,
        )
        self.mlp2_weight_scales = nn.Parameter(
            (
                self.num_experts,
                config.hidden_size,
                self.mxfp4_group_size,
            ),
            dtype=self.mxfp4_dtype,
        )

        # mlp_weights are processed in `.forward()`

        self.mlp1_bias = nn.Parameter(
            (self.num_experts, config.intermediate_size * 2 // self.world_size),
            dtype=dtype,
        )
        self.mlp2_bias = nn.Parameter(
            (self.num_experts, config.hidden_size),
            dtype=dtype,
        )

    def __dequantize_mxfp4_weight(
        self,
        blocks: nn.Tensor,
        scales: nn.Tensor,
        rows_per_chunk: int | None = None,  # unused, kept for compatibility
    ) -> nn.Tensor:
        """
        remain to compare dequantize logics;
        instead of using this function directly,
        we dequantize MXFP4 in `run_einsum()` function
        """

        if rows_per_chunk is None:
            rows_per_chunk = 16384 * 512
        scales = scales.astype("int32") - 127
        assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"
        out_dtype = self.dtype

        lut = nn.Tensor.from_const(np.array(FP4_VALUES, dtype=out_dtype))

        *prefix_shape, group_size, bytes_per_block = blocks.shape
        total_rows = prod(prefix_shape) * group_size

        blocks = blocks.reshape(total_rows, bytes_per_block)
        scales = scales.reshape(total_rows, 1)

        # Process multiple rows per block for better GPU utilization
        rows_per_block = 8
        num_blocks = (total_rows + rows_per_block - 1) // rows_per_block

        @T.prim_func(private=True)
        def dequant_kernel(
            blocks_handle: T.handle,
            scales_handle: T.handle,
            lut_handle: T.handle,
            out_handle: T.handle,
        ):
            T.func_attr({"tir.noalias": True, "tir.is_scheduled": 1})
            blocks_buf = T.match_buffer(blocks_handle, (total_rows, bytes_per_block), "uint8")
            scales_buf = T.match_buffer(scales_handle, (total_rows, 1), "int32")
            lut_buf = T.match_buffer(lut_handle, (16,), out_dtype)
            out_buf = T.match_buffer(out_handle, (total_rows, 2 * bytes_per_block), out_dtype)

            # Shared memory for LUT (16 values) and cached exp2 scales (per row)
            lut_shared = T.alloc_buffer((16,), out_dtype, scope="shared")
            scale_shared = T.alloc_buffer((rows_per_block,), "float32", scope="shared")

            with T.block("init_dequantize"):
                for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
                    for local_idx in T.thread_binding(rows_per_block * bytes_per_block, thread="threadIdx.x"):
                        # Load LUT into shared memory (first 16 threads)
                        with T.block("load_lut"):
                            vblock, vlocal = T.axis.remap("SS", [block_idx, local_idx])
                            if vlocal < 16:
                                lut_shared[vlocal] = lut_buf[vlocal]

                        # Load and compute exp2 scales into shared memory (one per row)
                        with T.block("load_scales"):
                            vblock, vlocal = T.axis.remap("SS", [block_idx, local_idx])
                            local_row = vlocal // bytes_per_block
                            is_first_in_row = (vlocal % bytes_per_block) == 0
                            vi = vblock * rows_per_block + local_row
                            if is_first_in_row and vi < total_rows:
                                exp_val = scales_buf[vi, 0]
                                scale_shared[local_row] = T.exp2(T.cast(exp_val, "float32"))

                        # Sync threads to ensure shared memory is populated
                        with T.block("sync"):
                            T.evaluate(0)

                        # Main dequantization using shared memory
                        with T.block("run_dequantize"):
                            vblock, vlocal = T.axis.remap("SS", [block_idx, local_idx])
                            local_row = vlocal // bytes_per_block
                            vj = vlocal % bytes_per_block
                            vi = vblock * rows_per_block + local_row

                            if vi < total_rows:
                                block_val = blocks_buf[vi, vj]
                                ldexp_scale = scale_shared[local_row]

                                idx_low = T.cast(block_val & T.uint8(0x0F), "int64")
                                idx_high = T.cast(block_val >> T.uint8(4), "int64")

                                out_buf[vi, 2 * vj] = lut_shared[idx_low] * ldexp_scale
                                out_buf[vi, 2 * vj + 1] = lut_shared[idx_high] * ldexp_scale

        out = nn.tensor_ir_op(
            dequant_kernel,
            "dequant_mxfp4",
            args=[blocks, scales, lut],
            out=nn.Tensor.placeholder((total_rows, 2 * bytes_per_block), dtype=out_dtype),
        )

        return out.reshape(*prefix_shape, 2 * group_size * bytes_per_block)

    def run_einsum(
        self,
        input_tensor: nn.Tensor,
        expert_indices: nn.Tensor,
        weight_blocks: nn.Tensor,
        weight_scales: nn.Tensor,
        bias_tensor: nn.Tensor,
        safetensor_dtype: str | None = None,
    ) -> nn.Tensor:
        """
        Fused einsum with on-the-fly MXFP4 dequantization.
        """

        einsum_dtype = "float32"
        safetensor_dtype = self.dtype

        seq_len = input_tensor.shape[0]
        input_shape = input_tensor.shape
        input_is_3d = len(input_shape) == 3

        _, experts_per_token = expert_indices.shape
        num_all_experts, out_features, group_size, bytes_per_block = weight_blocks.shape
        in_features = group_size * bytes_per_block * 2  # 90 * 16 * 2 = 2880

        lut = nn.Tensor.from_const(np.array(FP4_VALUES, dtype=safetensor_dtype))

        @T.prim_func(private=True)
        def _einsum_fused(
            x_handle: T.handle,
            e_indices: T.Buffer((seq_len, experts_per_token), "int32"),
            blocks: T.Buffer((num_all_experts, out_features, group_size, bytes_per_block), "uint8"),
            scales: T.Buffer((num_all_experts, out_features, group_size), "uint8"),
            bias: T.Buffer((num_all_experts, out_features), safetensor_dtype),
            lut_buf: T.Buffer((16,), safetensor_dtype),
            out_handle: T.handle,
        ):
            T.func_attr({"op_pattern": 4, "tir.noalias": True})
            x = T.match_buffer(x_handle, input_shape, input_tensor.dtype)
            out = T.match_buffer(out_handle, (seq_len, experts_per_token, out_features), einsum_dtype)

            for s, r, c in T.grid(seq_len, experts_per_token, out_features):
                with T.block("compute_fused_einsum"):
                    seq_idx, expert_rank, out_idx = T.axis.remap("SSS", [s, r, c])
                    e_idx = e_indices[seq_idx, expert_rank]

                    sum_buffer = T.alloc_buffer((1,), dtype=einsum_dtype, scope="local")
                    sum_buffer[0] = T.cast(bias[e_idx, out_idx], einsum_dtype)

                    for in_idx in T.serial(in_features):
                        # Compute group_idx, byte_idx, nibble position from in_idx
                        group_idx = in_idx // 32  # 32 = bytes_per_block * 2
                        local_idx = in_idx % 32
                        byte_idx = local_idx // 2
                        is_high = local_idx % 2

                        # Read scale
                        scale_val = T.cast(scales[e_idx, out_idx, group_idx], "int32") - 127
                        ldexp_scale = T.exp2(T.cast(scale_val, "float32"))

                        # Read block and extract nibble without if-else branching
                        block_val = blocks[e_idx, out_idx, group_idx, byte_idx]
                        shift_amount = T.cast(is_high * 4, "uint8")
                        nibble_idx = T.cast((block_val >> shift_amount) & T.uint8(0x0F), "int64")

                        # Accumulate
                        if input_is_3d:
                            sum_buffer[0] += (
                                T.cast(x[seq_idx, expert_rank, in_idx], einsum_dtype)
                                * T.cast(lut_buf[nibble_idx], einsum_dtype)
                                * ldexp_scale
                            )
                        else:
                            sum_buffer[0] += (
                                T.cast(x[seq_idx, in_idx], einsum_dtype)
                                * T.cast(lut_buf[nibble_idx], einsum_dtype)
                                * ldexp_scale
                            )

                    out[seq_idx, expert_rank, out_idx] = sum_buffer[0]

        return nn.tensor_ir_op(
            _einsum_fused,
            "moe_einsum_fused",
            args=[input_tensor, expert_indices, weight_blocks, weight_scales, bias_tensor, lut],
            out=nn.Tensor.placeholder((seq_len, experts_per_token, out_features), einsum_dtype),
        )

    def run_gating(
        self,
        input_tensor: nn.Tensor,
        expert_weights: nn.Tensor,
        out_dtype: str | None = None,
    ):
        # f32 -> out_dtype
        seq_len, experts_per_token, out_features = input_tensor.shape
        block_dtype = "float32"
        if out_dtype is None:
            out_dtype = self.dtype

        @T.prim_func(private=True)
        def _apply_gate(
            x_handle: T.handle,
            e_weights: T.Buffer((seq_len, experts_per_token), out_dtype),
            out_handle: T.handle,
        ):
            T.func_attr({"op_pattern": 4, "tir.noalias": True})
            x = T.match_buffer(x_handle, (seq_len, experts_per_token, out_features), input_tensor.dtype)
            out = T.match_buffer(out_handle, (seq_len, out_features), out_dtype)

            for s, c in T.grid(seq_len, out_features):
                with T.block("apply_gate"):
                    seq_idx, out_idx = T.axis.remap("SS", [s, c])

                    gate_buffer = T.alloc_buffer((1,), dtype=block_dtype, scope="local")
                    gate_buffer[0] = T.cast(0.0, block_dtype)

                    for expert_rank in T.serial(experts_per_token):
                        gate_buffer[0] += T.cast(x[seq_idx, expert_rank, out_idx], block_dtype) * T.cast(
                            e_weights[seq_idx, expert_rank], block_dtype
                        )

                    out[seq_idx, out_idx] = T.cast(gate_buffer[0], out_dtype)

        return nn.tensor_ir_op(
            _apply_gate,
            "moe_gemv_gating",
            args=[input_tensor, expert_weights],
            out=nn.Tensor.placeholder((seq_len, out_features), out_dtype),
        )

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        b, seq_len, dim = x.shape
        x = nn.op.reshape(x, (b * seq_len, dim))

        t = self.norm(x)  # t: (seq_len, 2880)
        g = self.gate(t)  # g: (seq_len, 32)

        # here, type: (Tensor, Tensor)
        # top_k is 4
        # expert_indices: (seq_len, top_k)
        # expert_weights, expert_indices = nn.op.topk(g, k=self.experts_per_token, axis=-1, largest=True)
        # expert_weights = nn.op.softmax(expert_weights, axis=1)
        expert_weights, expert_indices = gating_softmax_topk(g, k=self.experts_per_token)

        # MLP #1
        # self.mlp1_weight: (32, 5760, 2880)
        # self.mlp1_bias: (32, 5760)
        t = self.run_einsum(
            t, expert_indices, self.mlp1_weight_blocks, self.mlp1_weight_scales, self.mlp1_bias
        )  # (seq_len, top_k, 5760)
        t = swiglu(t, limit=self.swiglu_limit, out_dtype="float32")  # (seq_len, top_k, 2880)

        # MLP #2
        # self.mlp2_weight: (32, 2880, 2880)
        # self.mlp2_bias: (32, 2880)
        t = self.run_einsum(t, expert_indices, self.mlp2_weight_blocks, self.mlp2_weight_scales, self.mlp2_bias)

        # Weighted sum of experts
        t = self.run_gating(t, expert_weights)
        t = nn.op.reshape(t, x.shape)  # this resolves inconsistent buffer error
        t = t.astype(self.dtype)

        return nn.op.reshape(x + t, (b, seq_len, dim))
