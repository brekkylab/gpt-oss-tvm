from typing import Optional

import numpy as np
from mlc_llm.op.moe_matmul import gemv
from mlc_llm.op.moe_misc import (
    gating_softmax_topk,
    get_indices,
    get_indptr,
    moe_cumsum,
    moe_sum,
    scatter_output,
)
from tvm import tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
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


class MLPBlock(nn.Module):
    def __init__(
        self,
        config: GPTOssConfig,
        rms_norm_eps: float = 1e-5,
        dtype: Optional[str] = None,
    ):
        assert config
        if dtype is None:
            dtype = config.dtype
        self.dtype = dtype
        self.num_experts = config.num_experts
        self.experts_per_token = config.num_experts_per_tok
        self.swiglu_limit = config.swiglu_limit
        self.world_size = 1
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
        self.intermediate_size = config.intermediate_size // self.world_size
        self.mxfp4_dtype = "uint8"
        self.mxfp4_group_size = 90
        self.mxfp4_block_length = 16

        # Transposed Layout: (E, group, b, N)
        self.mlp1_weight_blocks = nn.Parameter(
            (self.num_experts, self.mxfp4_group_size, self.mxfp4_block_length, self.intermediate_size * 2),
            dtype=self.mxfp4_dtype,
        )
        self.mlp1_weight_scales = nn.Parameter(
            (self.num_experts, self.mxfp4_group_size, self.intermediate_size * 2),
            dtype=self.mxfp4_dtype,
        )
        self.mlp2_weight_blocks = nn.Parameter(
            (self.num_experts, self.mxfp4_group_size, self.mxfp4_block_length, self.hidden_size),
            dtype=self.mxfp4_dtype,
        )
        self.mlp2_weight_scales = nn.Parameter(
            (self.num_experts, self.mxfp4_group_size, self.hidden_size),
            dtype=self.mxfp4_dtype,
        )
        self.mlp1_bias = nn.Parameter((self.num_experts, self.intermediate_size * 2), dtype=dtype)
        self.mlp2_bias = nn.Parameter((self.num_experts, self.hidden_size), dtype=dtype)

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        b, s, h = x.shape
        num_tokens = b * s
        x_reshaped = x.reshape(num_tokens, h)
        t = self.norm(x_reshaped)

        is_decode = (isinstance(num_tokens, int) and num_tokens == 1) or (
            isinstance(num_tokens, tir.IntImm) and num_tokens.value == 1
        )

        if is_decode:
            return self.forward_decode(x, t, b, s, h)
        else:
            return self.forward_prefill(x, t, b, s, h, num_tokens)

    def get_lut(self, dtype: str) -> Tensor:
        return nn.Tensor.from_const(np.array(FP4_VALUES, dtype=dtype))

    def forward_prefill(self, x, t, b, s, h, num_tokens):
        g = self.gate(t)
        k = self.experts_per_token
        num_experts = self.num_experts
        lut = self.get_lut(self.dtype)

        expert_weights, expert_indices = gating_softmax_topk(g, k)
        cumsum = moe_cumsum(expert_indices, num_experts)

        rev_indices, token_indices = get_indices(cumsum, expert_indices)

        num_tokens_val = num_tokens if not isinstance(num_tokens, int) else tir.const(num_tokens, "int32")
        indptr = get_indptr(cumsum, tir.IntImm("int32", num_experts), num_tokens_val, False, "int32")

        @T.prim_func(private=True)
        def _get_expert_ids_func(var_indptr: T.handle, var_out: T.handle):
            M_val = T.int32(is_size_var=True)
            ind_buf = T.match_buffer(var_indptr, (num_experts + 1,), "int32")
            out_buf = T.match_buffer(var_out, (M_val,), "int32")
            for i_b in T.thread_binding((M_val + 255) // 256, thread="blockIdx.x"):
                for i_t in T.thread_binding(256, thread="threadIdx.x"):
                    with T.block("lookup"):
                        vi = i_b * 256 + i_t
                        if vi < M_val:
                            res = T.alloc_buffer((1,), "int32", scope="local")
                            res[0] = 0
                            for e in T.serial(num_experts):
                                if ind_buf[e] <= vi:
                                    res[0] = e
                            out_buf[vi] = res[0]

        token_expert_ids = nn.tensor_ir_op(
            _get_expert_ids_func,
            "moe_expand_indptr",
            args=[indptr],
            out=nn.Tensor.placeholder([num_tokens * k], "int32"),
        )

        x_sorted = op.take(t, token_indices, axis=0).astype(self.dtype)

        # Note regarding group_gemm:
        # The mlc_llm.op.moe_matmul.group_gemm / dequantize_group_gemm operators rely on
        # standard integer packing (e.g. int4/int8) or fp8. Our weights use a custom MXFP4
        # format with LUT-based dequantization, which is incompatible with the library kernels.
        # Thus, we use our specialized 'mxfp4_moe_mlp*_prefill' kernels instead.
        x_sorted = self.mxfp4_moe_mlp1_prefill(
            x_sorted, self.mlp1_weight_blocks, self.mlp1_weight_scales, self.mlp1_bias, token_expert_ids, lut
        )
        x_sorted = self.mxfp4_moe_mlp2_prefill(
            x_sorted,
            self.mlp2_weight_blocks,
            self.mlp2_weight_scales,
            self.mlp2_bias,
            token_expert_ids,
            lut,
        )
        x_scattered = scatter_output(x_sorted, rev_indices).reshape(num_tokens, k, h)
        x_scattered = x_scattered * expert_weights.astype(self.dtype).reshape(num_tokens, k, 1)
        res = moe_sum(x_scattered, dim=1).reshape(b, s, h).astype(self.dtype)

        return x + res

    def forward_decode(self, x, t, b, s, h):
        k = self.experts_per_token
        num_experts = self.num_experts
        lut = self.get_lut(self.dtype)

        # Step 0: Gating using optimized gemv
        # w_gate: (num_experts, 1, h), t: (1, h) -> out: (num_experts, 1)
        gate_indptr = nn.Tensor.from_const(np.arange(self.num_experts, dtype="int32").reshape(1, self.num_experts))
        g = gemv(t, self.gate.weight.reshape(self.num_experts, 1, h), gate_indptr)
        g = g.reshape(1, self.num_experts)
        if self.gate.bias is not None:
            g = op.add(g, self.gate.bias.reshape(1, num_experts))

        expert_weights, expert_indices = gating_softmax_topk(g, k)

        token_expert_ids = expert_indices.reshape(k)
        x_sorted = op.broadcast_to(t, [k, h]).astype(self.dtype)

        # Step 1: Expert-Parallel MLP1
        x_intermediate = self.mxfp4_moe_mlp1_decode(
            x_sorted, self.mlp1_weight_blocks, self.mlp1_weight_scales, self.mlp1_bias, token_expert_ids, lut
        )
        # Step 2: Expert-Parallel MLP2
        x_expert_out = self.mxfp4_moe_mlp2_decode(
            x_intermediate,
            self.mlp2_weight_blocks,
            self.mlp2_weight_scales,
            self.mlp2_bias,
            token_expert_ids,
            lut,
        )

        # Step 3: Weighted Average across experts using gemv
        # (1, k) @ (k, h) -> (1, h)
        w_sum = x_expert_out.permute_dims(1, 0).reshape(1, h, k)
        indptr_sum = op.full([1, 1], 0, dtype="int32")
        res = gemv(expert_weights.astype(self.dtype), w_sum, indptr_sum)

        return x + res.reshape(b, s, h).astype(self.dtype)

    def mxfp4_moe_mlp1_prefill(self, x, blocks, scales, bias, token_expert_ids, lut) -> Tensor:
        """
        Implementation of the first MLP layer (Gate+Up projection) for the prefill phase using MXFP4 quantization.
        This kernel performs: Output = SwiGLU(X @ W_gate + b_gate, X @ W_up + b_up).

        Key Implementation Details:
        - **BFloat16 Global Memory**: Inputs, outputs, and LUTs utilize bfloat16 to maximize global memory bandwidth,
          which is the primary bottleneck for this kernel.
        - **Float32 Shared Memory**: Intermediate values in shared memory (`x_sh`, `lut_sh`) are cast to `float32`.
          This ensures numerical stability during accumulation at the cost of slightly higher shared memory pressure.
        - **Throughput Optimized**: The kernel processes 4 rows (tokens) per thread block (`T.unroll(4)`), increasing
          instruction-level parallelism and throughput for large batch sizes typical in prefill.

        Args:
            x (Tensor): Input tensor of shape (Total_Tokens, Hidden_Size). Dtype: bfloat16.
            blocks (Tensor): Quantized weight blocks (uint8) for both Gate and Up projections.
            scales (Tensor): Scaling factors (uint8) for dequantization.
            bias (Tensor): Bias terms for Gate and Up projections. Dtype: bfloat16.
            token_expert_ids (Tensor): Mapping of each token-expert pair to the specific expert ID.
            lut (Tensor): Lookup table for MXFP4 dequantization. Dtype: bfloat16.

        Returns:
            Tensor: Output tensor of shape (Total_Tokens, Intermediate_Size). Dtype: bfloat16.
        """
        (E, group_size, B_L, out_features) = blocks.shape
        B_val_outer = x.shape[0]
        N = out_features // 2

        @T.prim_func(private=True)
        def _func(
            var_x: T.handle,
            var_b: T.handle,
            var_s: T.handle,
            var_bias: T.handle,
            var_e: T.handle,
            var_lut: T.handle,
            var_o: T.handle,
        ):
            B_val = T.int32(is_size_var=True)
            T.func_attr({"op_pattern": 4, "tir.noalias": True, "tir.is_scheduled": 1})
            x_buf = T.match_buffer(var_x, (B_val, self.hidden_size), x.dtype)
            blocks_buf = T.match_buffer(var_b, (E, group_size, B_L, out_features), "uint8")
            scales_buf = T.match_buffer(var_s, (E, group_size, out_features), "uint8")
            bias_buf = T.match_buffer(var_bias, (E, out_features), bias.dtype)
            e_ids_buf = T.match_buffer(var_e, (B_val,), "int32")
            lut_buf = T.match_buffer(var_lut, (16,), self.dtype)
            out_buf = T.match_buffer(var_o, (B_val, N), self.dtype)
            # Thread Binding Strategy:
            # blockIdx.x covers the Output Features (N) dimension, tiled by 128.
            # blockIdx.y covers the Batch (Tokens) dimension, tiled by 4 (Throughput Optimization).
            # threadIdx.x is a standard 128-thread warp/block.
            for vj_b in T.thread_binding((N + 127) // 128, thread="blockIdx.x"):
                for vi_b in T.thread_binding((B_val + 3) // 4, thread="blockIdx.y"):
                    for vj_t in T.thread_binding(128, thread="threadIdx.x"):
                        with T.block("compute"):
                            vj = vj_b * 128 + vj_t
                            # Shared memory buffers are essentially cache for the inner loop.
                            # We use "float32" here to ensure numerical stability during accumulation.
                            # x_sh: Double buffer (2) for pipe-lining, 4 tokens (throughput), 32 dim chunk.
                            x_sh = T.alloc_buffer((2, 4, 32), "float32", scope="shared")
                            lut_sh = T.alloc_buffer((16,), "float32", scope="shared")
                            acc_g = T.alloc_buffer((4,), "float32", scope="local")
                            acc_l = T.alloc_buffer((4,), "float32", scope="local")
                            ids = T.alloc_buffer((4,), "int32", scope="local")
                            # Load LUT into shared memory, casting to float32 immediately.
                            if vj_t < 16:
                                lut_sh[vj_t] = T.cast(lut_buf[vj_t], "float32")
                            for t in T.unroll(4):
                                vi = vi_b * 4 + t
                                if vi < B_val:
                                    ids[t] = e_ids_buf[vi]
                                    if vj < N:
                                        acc_g[t] = T.cast(bias_buf[ids[t], vj * 2], "float32")
                                        acc_l[t] = T.cast(bias_buf[ids[t], vj * 2 + 1], "float32")
                                    else:
                                        acc_g[t] = 0.0
                                        acc_l[t] = 0.0
                                else:
                                    ids[t] = -1
                                    acc_g[t] = 0.0
                                    acc_l[t] = 0.0
                            t_ld, d_ld = vj_t // 32, vj_t % 32
                            # Load initial input tile to shared memory (Double Buffer 0).
                            # Cast bfloat16 -> float32 on load to ensure inner loop is pure float32.
                            if (vi_b * 4 + t_ld) < B_val:
                                x_sh[0, t_ld, d_ld] = T.cast(x_buf[vi_b * 4 + t_ld, d_ld], "float32")
                            T.tvm_storage_sync("shared")
                            # Inner Loop: Iterate over group_size (K dimension chunks).
                            for g in T.serial(group_size):
                                # Double buffering: Load next tile (g+1) while computing current tile (g).
                                if g + 1 < group_size:
                                    if (vi_b * 4 + t_ld) < B_val:
                                        x_sh[(g + 1) % 2, t_ld, d_ld] = T.cast(
                                            x_buf[vi_b * 4 + t_ld, (g + 1) * 32 + d_ld], "float32"
                                        )
                                if vj < N:
                                    for t in T.unroll(4):
                                        if ids[t] != -1:
                                            sg = T.exp2(
                                                T.cast(T.cast(scales_buf[ids[t], g, vj * 2], "int32") - 127, "float32")
                                            )
                                            sl = T.exp2(
                                                T.cast(
                                                    T.cast(scales_buf[ids[t], g, vj * 2 + 1], "int32") - 127, "float32"
                                                )
                                            )
                                            for b in T.unroll(B_L):
                                                # Dequantize weights using MXFP4 logic.
                                                # bk_g/bk_l: uint8 blocks containing packed 4-bit indices.
                                                # w0g/w1g: Look up actual float32 values from shared LUT.
                                                bk_g = blocks_buf[ids[t], g, b, vj * 2]
                                                bk_l = blocks_buf[ids[t], g, b, vj * 2 + 1]
                                                w0g = lut_sh[T.cast(bk_g & T.uint8(0x0F), "int32")]
                                                w1g = lut_sh[T.cast(bk_g >> T.uint8(4), "int32")]
                                                w0l = lut_sh[T.cast(bk_l & T.uint8(0x0F), "int32")]
                                                w1l = lut_sh[T.cast(bk_l >> T.uint8(4), "int32")]
                                                xx0 = x_sh[g % 2, t, b * 2]
                                                xx1 = x_sh[g % 2, t, b * 2 + 1]
                                                # Accumulate: Matrix multiplication (dot product)
                                                acc_g[t] = acc_g[t] + (xx0 * w0g + xx1 * w1g) * sg
                                                acc_l[t] = acc_l[t] + (xx0 * w0l + xx1 * w1l) * sl
                            T.tvm_storage_sync("shared")
                            # Post-processing: SwiGLU activation and output casting.
                            for t in T.unroll(4):
                                if (vi_b * 4 + t) < B_val and vj < N:
                                    # SwiGLU: (x * sigmoid(x)) * y
                                    # xg: Gate activation, xl: Linear activation (clipped)
                                    xg = T.min(acc_g[t], T.float32(self.swiglu_limit))
                                    xl = T.max(T.min(acc_l[t], T.float32(7.0)), T.float32(-7.0))
                                    gv = xg / (T.float32(1.0) + T.exp(-T.float32(1.702) * xg))
                                    out_buf[vi_b * 4 + t, vj] = T.cast(gv * (xl + T.float32(1.0)), self.dtype)

        return op.tensor_ir_op(
            _func,
            "mxfp4_moe_mlp1_prefill",
            args=[x, blocks, scales, bias, token_expert_ids, lut],
            out=Tensor.placeholder([B_val_outer, N], self.dtype),
        )

    def mxfp4_moe_mlp2_prefill(self, x, blocks, scales, bias, token_expert_ids, lut) -> Tensor:
        """
        Implementation of the second MLP layer (Down projection) for the prefill phase using MXFP4 quantization.
        This kernel performs: Output = X @ W_down + b_down.

        Key Implementation Details:
        - **BFloat16 Global Memory**: Inputs (activations from MLP1), outputs, and LUTs are kept in bfloat16 for bandwidth efficiency.
        - **Float32 Shared Memory**: Explicit casting to `float32` upon loading into shared memory ensures numerical stability during accumulation.
        - **Throughput Optimized**: Similar to MLP1, this uses 4-way unrolling to handle multiple tokens per block efficiently.

        Args:
            x (Tensor): Input tensor (activations from MLP1). Dtype: bfloat16.
            blocks (Tensor): Quantized weight blocks (uint8) for Down projection.
            scales (Tensor): Scaling factors (uint8).
            bias (Tensor): Bias terms. Dtype: bfloat16.
            token_expert_ids (Tensor): Expert IDs for routing.
            lut (Tensor): MXFP4 lookup table. Dtype: bfloat16.

        Returns:
            Tensor: Output tensor of shape (Total_Tokens, Hidden_Size). Dtype: bfloat16.
        """
        (E, group_size, B_L, N) = blocks.shape
        B_val_outer = x.shape[0]

        @T.prim_func(private=True)
        def _func(
            var_x: T.handle,
            var_b: T.handle,
            var_s: T.handle,
            var_bias: T.handle,
            var_e: T.handle,
            var_lut: T.handle,
            var_o: T.handle,
        ):
            B_val = T.int32(is_size_var=True)
            T.func_attr({"op_pattern": 4, "tir.noalias": True, "tir.is_scheduled": 1})
            x_buf = T.match_buffer(var_x, (B_val, self.hidden_size), self.dtype)
            blocks_buf = T.match_buffer(var_b, (E, group_size, B_L, N), "uint8")
            scales_buf = T.match_buffer(var_s, (E, group_size, N), "uint8")
            bias_buf = T.match_buffer(var_bias, (E, N), bias.dtype)
            e_ids_buf = T.match_buffer(var_e, (B_val,), "int32")
            lut_buf = T.match_buffer(var_lut, (16,), self.dtype)
            out_buf = T.match_buffer(var_o, (B_val, N), self.dtype)
            # Thread Binding Strategy:
            # blockIdx.x covers Output Features (N) tiled by 128.
            # blockIdx.y covers Tokens (B_val) tiled by 4.
            for vj_b in T.thread_binding((N + 127) // 128, thread="blockIdx.x"):
                for vi_b in T.thread_binding((B_val + 3) // 4, thread="blockIdx.y"):
                    for vj_t in T.thread_binding(128, thread="threadIdx.x"):
                        with T.block("compute"):
                            vj = vj_b * 128 + vj_t
                            # Use float32 shared memory for stability.
                            x_sh = T.alloc_buffer((2, 4, 32), "float32", scope="shared")
                            lut_sh = T.alloc_buffer((16,), "float32", scope="shared")
                            acc = T.alloc_buffer((4,), "float32", scope="local")
                            ids = T.alloc_buffer((4,), "int32", scope="local")
                            # Load LUT to shared memory in float32
                            if vj_t < 16:
                                lut_sh[vj_t] = T.cast(lut_buf[vj_t], "float32")
                            for t in T.unroll(4):
                                vi = vi_b * 4 + t
                                if vi < B_val:
                                    ids[t] = e_ids_buf[vi]
                                    acc[t] = T.cast(bias_buf[ids[t], vj], "float32") if vj < N else 0.0
                                else:
                                    ids[t] = -1
                                    acc[t] = 0.0
                            t_ld, d_ld = vj_t // 32, vj_t % 32
                            # Load initial input tile to shared memory (Double Buffer 0).
                            # Cast bfloat16 -> float32 on load to ensure inner loop is pure float32.
                            if (vi_b * 4 + t_ld) < B_val:
                                x_sh[0, t_ld, d_ld] = T.cast(x_buf[vi_b * 4 + t_ld, d_ld], "float32")
                            T.tvm_storage_sync("shared")
                            # Inner Loop: Iterate over K dimension chunks.
                            for g in T.serial(group_size):
                                # Double buffering: Load next tile.
                                if g + 1 < group_size:
                                    if (vi_b * 4 + t_ld) < B_val:
                                        x_sh[(g + 1) % 2, t_ld, d_ld] = T.cast(
                                            x_buf[vi_b * 4 + t_ld, (g + 1) * 32 + d_ld], "float32"
                                        )
                                if vj < N:
                                    for t in T.unroll(4):
                                        if ids[t] != -1:
                                            sc = T.exp2(
                                                T.cast(T.cast(scales_buf[ids[t], g, vj], "int32") - 127, "float32")
                                            )
                                            for b in T.unroll(B_L):
                                                # Dequantize weights (uint8 blocks -> 4-bit indices -> float32 LUT).
                                                bk = blocks_buf[ids[t], g, b, vj]
                                                w0 = lut_sh[T.cast(bk & T.uint8(0x0F), "int32")]
                                                w1 = lut_sh[T.cast(bk >> T.uint8(4), "int32")]
                                                acc[t] = (
                                                    acc[t]
                                                    + (x_sh[g % 2, t, b * 2] * w0 + x_sh[g % 2, t, b * 2 + 1] * w1) * sc
                                                )
                                T.tvm_storage_sync("shared")
                            # Output stored back to global memory (bfloat16).
                            for t in T.unroll(4):
                                if (vi_b * 4 + t) < B_val and vj < N:
                                    out_buf[vi_b * 4 + t, vj] = T.cast(acc[t], self.dtype)

        return op.tensor_ir_op(
            _func,
            "mxfp4_moe_mlp2_prefill",
            args=[x, blocks, scales, bias, token_expert_ids, lut],
            out=Tensor.placeholder([B_val_outer, N], self.dtype),
        )

    def mxfp4_moe_mlp1_decode(self, x, blocks, scales, bias, token_expert_ids, lut) -> Tensor:
        """
        Implementation of the first MLP layer (Gate+Up projection) for the decode phase.
        Optimized for low-latency, single-token generation.

        Key Implementation Details:
        - **Latency Optimized**: Unlike the prefill kernel, this processes 1 row (token/expert pair) per thread block.
          For the decode batch size of 4 (1 token * 4 experts), this results in 4 parallel blocks, maximizing GPU occupancy
          and minimizing latency compared to the batched approach of the prefill kernel.
        - **BFloat16 Global / Float32 Shared**: Retains the bandwidth-saving global BFloat16 layout while using
          Float32 for all on-chip compute and storage to ensure numerical stability during accumulation.

        Args:
            x (Tensor): Input tensor of shape (4, Hidden_Size) representing the 4 selected experts for the single decode token.
            blocks (Tensor): Quantized weight blocks.
            scales (Tensor): Scaling factors.
            bias (Tensor): Bias terms.
            token_expert_ids (Tensor): Indices of the 4 selected experts.
            lut (Tensor): MXFP4 lookup table.

        Returns:
            Tensor: Output tensor of shape (4, Intermediate_Size).
        """
        (E, group_size, B_L, out_features) = blocks.shape
        N = out_features // 2

        @T.prim_func(private=True)
        def _func(
            var_x: T.handle,
            var_b: T.handle,
            var_s: T.handle,
            var_bias: T.handle,
            var_e: T.handle,
            var_lut: T.handle,
            var_o: T.handle,
        ):
            T.func_attr({"op_pattern": 4, "tir.noalias": True, "tir.is_scheduled": 1})
            x_buf = T.match_buffer(var_x, (4, self.hidden_size), self.dtype)
            blocks_buf = T.match_buffer(var_b, (E, group_size, B_L, out_features), "uint8")
            scales_buf = T.match_buffer(var_s, (E, group_size, out_features), "uint8")
            bias_buf = T.match_buffer(var_bias, (E, out_features), self.dtype)
            e_ids_buf = T.match_buffer(var_e, (4,), "int32")
            lut_buf = T.match_buffer(var_lut, (16,), self.dtype)
            out_buf = T.match_buffer(var_o, (4, N), self.dtype)
            # Thread Binding (Latency Optimized):
            # blockIdx.x covers Output Features (N) tiled by 128.
            # blockIdx.y covers existing Tokens (4), 1 block per token.
            # This maximizes occupancy for small N=4 batches.
            for vj_b in T.thread_binding((N + 127) // 128, thread="blockIdx.x"):
                for vi_b in T.thread_binding(4, thread="blockIdx.y"):
                    for vj_t in T.thread_binding(128, thread="threadIdx.x"):
                        with T.block("compute"):
                            vj = vj_b * 128 + vj_t
                            # Decode kernel uses float32 shared memory.
                            # Unlike prefill, we only process 1 token per block, so the shape is (2, 32).
                            x_sh = T.alloc_buffer((2, 32), "float32", scope="shared")
                            lut_sh = T.alloc_buffer((16,), "float32", scope="shared")
                            acc_g = T.alloc_buffer((1,), "float32", scope="local")
                            acc_l = T.alloc_buffer((1,), "float32", scope="local")
                            # Load LUT to shared memory in float32
                            if vj_t < 16:
                                lut_sh[vj_t] = T.cast(lut_buf[vj_t], "float32")
                            eid = e_ids_buf[vi_b]
                            if vj < N:
                                acc_g[0] = T.cast(bias_buf[eid, vj * 2], "float32")
                                acc_l[0] = T.cast(bias_buf[eid, vj * 2 + 1], "float32")
                            else:
                                acc_g[0] = 0.0
                                acc_l[0] = 0.0
                            # Load input tile to shared memory, casting to float32.
                            if vj_t < 32:
                                x_sh[0, vj_t] = T.cast(x_buf[vi_b, vj_t], "float32")
                            T.tvm_storage_sync("shared")
                            # Inner Loop: Iterate over group_size chunks.
                            for g in T.serial(group_size):
                                # Double Buffering: Load next chunk (g+1)
                                if g + 1 < group_size:
                                    if vj_t < 32:
                                        x_sh[(g + 1) % 2, vj_t] = T.cast(x_buf[vi_b, (g + 1) * 32 + vj_t], "float32")
                                if vj < N:
                                    sg = T.exp2(T.cast(T.cast(scales_buf[eid, g, vj * 2], "int32") - 127, "float32"))
                                    sl = T.exp2(
                                        T.cast(T.cast(scales_buf[eid, g, vj * 2 + 1], "int32") - 127, "float32")
                                    )
                                    for b in T.unroll(B_L):
                                        # Dequantize: unpack 4-bit weights from uint8 blocks using LUT.
                                        bk_g = blocks_buf[eid, g, b, vj * 2]
                                        bk_l = blocks_buf[eid, g, b, vj * 2 + 1]
                                        w0g = lut_sh[T.cast(bk_g & T.uint8(0x0F), "int32")]
                                        w1g = lut_sh[T.cast(bk_g >> T.uint8(4), "int32")]
                                        w0l = lut_sh[T.cast(bk_l & T.uint8(0x0F), "int32")]
                                        w1l = lut_sh[T.cast(bk_l >> T.uint8(4), "int32")]
                                        xx0, xx1 = x_sh[g % 2, b * 2], x_sh[g % 2, b * 2 + 1]
                                        # Accumulate in float32
                                        acc_g[0] = acc_g[0] + (xx0 * w0g + xx1 * w1g) * sg
                                        acc_l[0] = acc_l[0] + (xx0 * w0l + xx1 * w1l) * sl
                                T.tvm_storage_sync("shared")
                            # Post-processing: SwiGLU + Output Cast.
                            if vj < N:
                                xg = T.min(acc_g[0], T.float32(self.swiglu_limit))
                                xl = T.max(T.min(acc_l[0], T.float32(7.0)), T.float32(-7.0))
                                gv = xg / (T.float32(1.0) + T.exp(-T.float32(1.702) * xg))
                                out_buf[vi_b, vj] = T.cast(gv * (xl + T.float32(1.0)), self.dtype)

        return nn.tensor_ir_op(
            _func,
            "mxfp4_moe_mlp1_decode",
            args=[x, blocks, scales, bias, token_expert_ids, lut],
            out=Tensor.placeholder([4, N], self.dtype),
        )

    def mxfp4_moe_mlp2_decode(self, x, blocks, scales, bias, token_expert_ids, lut) -> Tensor:
        """
        Implementation of the second MLP layer (Down projection) for the decode phase.
        Optimized for low-latency, single-token generation.

        Key Implementation Details:
        - **Latency Optimized**: Launches a separate thread block for each of the 4 rows (expert computations) to minimize
          tail latency for the single generated token.
        - **Mixed Precision**: Uses BFloat16 for global data movement and Float32 for shared memory/ALU operations
          to balance bandwidth usage with compiler stability.

        Args:
            x (Tensor): Input tensor from MLP1, shape (4, Intermediate_Size).
            blocks (Tensor): Quantized weight blocks.
            scales (Tensor): Scaling factors.
            bias (Tensor): Bias terms.
            token_expert_ids (Tensor): Indices of the 4 selected experts.
            lut (Tensor): MXFP4 lookup table.

        Returns:
            Tensor: Output tensor of shape (4, Hidden_Size).
        """
        (E, group_size, B_L, N) = blocks.shape

        @T.prim_func(private=True)
        def _func(
            var_x: T.handle,
            var_b: T.handle,
            var_s: T.handle,
            var_bias: T.handle,
            var_e: T.handle,
            var_lut: T.handle,
            var_o: T.handle,
        ):
            T.func_attr({"op_pattern": 4, "tir.noalias": True, "tir.is_scheduled": 1})
            x_buf = T.match_buffer(var_x, (4, self.hidden_size), self.dtype)
            blocks_buf = T.match_buffer(var_b, (E, group_size, B_L, N), "uint8")
            scales_buf = T.match_buffer(var_s, (E, group_size, N), "uint8")
            bias_buf = T.match_buffer(var_bias, (E, N), self.dtype)
            e_ids_buf = T.match_buffer(var_e, (4,), "int32")
            lut_buf = T.match_buffer(var_lut, (16,), self.dtype)
            out_buf = T.match_buffer(var_o, (4, N), self.dtype)
            # Thread Binding (Latency Optimized):
            # blockIdx.x covers Output Features (N) tiled by 128.
            # blockIdx.y covers Tokens (4), 1 block per token.
            for vj_b in T.thread_binding((N + 127) // 128, thread="blockIdx.x"):
                for vi_b in T.thread_binding(4, thread="blockIdx.y"):
                    for vj_t in T.thread_binding(128, thread="threadIdx.x"):
                        with T.block("compute"):
                            vj = vj_b * 128 + vj_t
                            # Decode MLP2: float32 shared memory for robustness.
                            x_sh = T.alloc_buffer((2, 32), "float32", scope="shared")
                            lut_sh = T.alloc_buffer((16,), "float32", scope="shared")
                            acc = T.alloc_buffer((1,), "float32", scope="local")
                            # Load LUT to shared memory in float32
                            if vj_t < 16:
                                lut_sh[vj_t] = T.cast(lut_buf[vj_t], "float32")
                            eid = e_ids_buf[vi_b]
                            acc[0] = T.cast(bias_buf[eid, vj], "float32") if vj < N else 0.0
                            if vj_t < 32:
                                x_sh[0, vj_t] = T.cast(x_buf[vi_b, vj_t], "float32")
                            T.tvm_storage_sync("shared")
                            # Inner Loop: Iterate over K dimension chunks.
                            for g in T.serial(group_size):
                                # Double buffering for input loading.
                                if g + 1 < group_size:
                                    if vj_t < 32:
                                        x_sh[(g + 1) % 2, vj_t] = T.cast(x_buf[vi_b, (g + 1) * 32 + vj_t], "float32")
                                if vj < N:
                                    sc = T.exp2(T.cast(T.cast(scales_buf[eid, g, vj], "int32") - 127, "float32"))
                                    for b in T.unroll(B_L):
                                        # Dequantize: unpack 4-bit weights from uint8 blocks using LUT.
                                        bk = blocks_buf[eid, g, b, vj]
                                        w0 = lut_sh[T.cast(bk & T.uint8(0x0F), "int32")]
                                        w1 = lut_sh[T.cast(bk >> T.uint8(4), "int32")]
                                        # Accumulate in float32.
                                        acc[0] = acc[0] + (x_sh[g % 2, b * 2] * w0 + x_sh[g % 2, b * 2 + 1] * w1) * sc
                                T.tvm_storage_sync("shared")
                            # Cast output to bfloat16 and store to global memory.
                            if vj < N:
                                out_buf[vi_b, vj] = T.cast(acc[0], self.dtype)

        return nn.tensor_ir_op(
            _func,
            "mxfp4_moe_mlp2_decode",
            args=[x, blocks, scales, bias, token_expert_ids, lut],
            out=Tensor.placeholder([4, N], self.dtype),
        )
