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

        # ROBUST HIGH-PERFORMANCE DECODE PATH (B=1 optimization)
        is_decode = False
        if isinstance(num_tokens, int):
            if num_tokens == 1:
                is_decode = True
        elif isinstance(num_tokens, tir.IntImm):
            if num_tokens.value == 1:
                is_decode = True

        if is_decode:
            return self.forward_decode(x, t, b, s, h)
        return self.forward_prefill(x, t, b, s, h, num_tokens)

    def forward_prefill(self, x, t, b, s, h, num_tokens):
        # Note: We use standard nn.Linear for gating in prefill as gemv is optimized for single-token.
        g = self.gate(t)
        k = self.experts_per_token
        num_experts = self.num_experts
        lut = nn.Tensor.from_const(np.array(FP4_VALUES, dtype="float32"))

        # Optimization: Use fused topk+softmax kernel
        expert_weights, expert_indices = gating_softmax_topk(g, k)
        cumsum = moe_cumsum(expert_indices, num_experts)

        # Note regarding group_gemm:
        # The mlc_llm.op.moe_matmul.group_gemm / dequantize_group_gemm operators rely on
        # standard integer packing (e.g. int4/int8) or fp8. Our weights use a custom MXFP4
        # format with LUT-based dequantization, which is incompatible with the library kernels.
        # Thus, we use our specialized 'mxfp4_moe_mlp*_prefill' kernels instead.
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

        x_sorted = op.take(t, token_indices, axis=0)
        x_sorted = self.mxfp4_moe_mlp1_prefill(
            x_sorted, self.mlp1_weight_blocks, self.mlp1_weight_scales, self.mlp1_bias, token_expert_ids, lut
        )
        x_sorted = self.mxfp4_moe_mlp2_prefill(
            x_sorted, self.mlp2_weight_blocks, self.mlp2_weight_scales, self.mlp2_bias, token_expert_ids, lut
        )
        x_scattered = scatter_output(x_sorted, rev_indices).reshape(num_tokens, k, h)
        x_scattered = x_scattered * expert_weights.astype("float32").reshape(num_tokens, k, 1)
        res = moe_sum(x_scattered, dim=1).reshape(b, s, h).astype(self.dtype)

        return x + res

    def forward_decode(self, x, t, b, s, h):
        k = self.experts_per_token
        num_experts = self.num_experts
        lut = nn.Tensor.from_const(np.array(FP4_VALUES, dtype="float32"))

        # Step 0: Gating using optimized gemv
        # w_gate: (num_experts, 1, h), t: (1, h) -> out: (num_experts, 1)
        gate_indptr = nn.Tensor.from_const(np.arange(self.num_experts, dtype="int32").reshape(1, self.num_experts))
        g = gemv(t, self.gate.weight.reshape(self.num_experts, 1, h), gate_indptr)
        g = g.reshape(1, self.num_experts)
        if self.gate.bias is not None:
            g = op.add(g, self.gate.bias.reshape(1, num_experts))

        expert_weights, expert_indices = gating_softmax_topk(g, k)

        token_expert_ids = expert_indices.reshape(k)
        x_sorted = op.broadcast_to(t, [k, h])

        # Step 1: Expert-Parallel MLP1
        x_intermediate = self.mxfp4_moe_mlp1_decode(
            x_sorted, self.mlp1_weight_blocks, self.mlp1_weight_scales, self.mlp1_bias, token_expert_ids, lut
        )
        # Step 2: Expert-Parallel MLP2
        x_expert_out = self.mxfp4_moe_mlp2_decode(
            x_intermediate, self.mlp2_weight_blocks, self.mlp2_weight_scales, self.mlp2_bias, token_expert_ids, lut
        )

        # Step 3: Weighted Average across experts using gemv
        # (1, k) @ (k, h) -> (1, h)
        w_sum = x_expert_out.permute_dims(1, 0).reshape(1, h, k)
        indptr_sum = op.full([1, 1], 0, dtype="int32")
        res = gemv(expert_weights.astype("float32"), w_sum, indptr_sum)

        return x + res.reshape(b, s, h).astype(self.dtype)

    def mxfp4_moe_mlp1_prefill(self, x, blocks, scales, bias, token_expert_ids, lut) -> Tensor:
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
            lut_buf = T.match_buffer(var_lut, (16,), "float32")
            out_buf = T.match_buffer(var_o, (B_val, N), "float32")
            for vj_b in T.thread_binding((N + 127) // 128, thread="blockIdx.x"):
                for vi_b in T.thread_binding((B_val + 3) // 4, thread="blockIdx.y"):
                    for vj_t in T.thread_binding(128, thread="threadIdx.x"):
                        with T.block("compute"):
                            vj = vj_b * 128 + vj_t
                            x_sh = T.alloc_buffer((2, 4, 32), "float32", scope="shared")
                            lut_sh = T.alloc_buffer((16,), "float32", scope="shared")
                            acc_g = T.alloc_buffer((4,), "float32", scope="local")
                            acc_l = T.alloc_buffer((4,), "float32", scope="local")
                            ids = T.alloc_buffer((4,), "int32", scope="local")
                            if vj_t < 16:
                                lut_sh[vj_t] = lut_buf[vj_t]
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
                            if (vi_b * 4 + t_ld) < B_val:
                                x_sh[0, t_ld, d_ld] = T.cast(x_buf[vi_b * 4 + t_ld, d_ld], "float32")
                            T.tvm_storage_sync("shared")
                            for g in T.serial(group_size):
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
                                                bk_g = blocks_buf[ids[t], g, b, vj * 2]
                                                bk_l = blocks_buf[ids[t], g, b, vj * 2 + 1]
                                                w0g = lut_sh[T.cast(bk_g & T.uint8(0x0F), "int32")]
                                                w1g = lut_sh[T.cast(bk_g >> T.uint8(4), "int32")]
                                                w0l = lut_sh[T.cast(bk_l & T.uint8(0x0F), "int32")]
                                                w1l = lut_sh[T.cast(bk_l >> T.uint8(4), "int32")]
                                                xx0 = x_sh[g % 2, t, b * 2]
                                                xx1 = x_sh[g % 2, t, b * 2 + 1]
                                                acc_g[t] = acc_g[t] + (xx0 * w0g + xx1 * w1g) * sg
                                                acc_l[t] = acc_l[t] + (xx0 * w0l + xx1 * w1l) * sl
                            T.tvm_storage_sync("shared")
                            for t in T.unroll(4):
                                if (vi_b * 4 + t) < B_val and vj < N:
                                    xg = T.min(acc_g[t], T.float32(self.swiglu_limit))
                                    xl = T.max(T.min(acc_l[t], T.float32(7.0)), T.float32(-7.0))
                                    gv = xg / (T.float32(1.0) + T.exp(-T.float32(1.702) * xg))
                                    out_buf[vi_b * 4 + t, vj] = gv * (xl + T.float32(1.0))

        return op.tensor_ir_op(
            _func,
            "mxfp4_moe_mlp1_prefill",
            args=[x, blocks, scales, bias, token_expert_ids, lut],
            out=Tensor.placeholder([B_val_outer, N], "float32"),
        )

    def mxfp4_moe_mlp2_prefill(self, x, blocks, scales, bias, token_expert_ids, lut) -> Tensor:
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
            x_buf = T.match_buffer(var_x, (B_val, self.hidden_size), "float32")
            blocks_buf = T.match_buffer(var_b, (E, group_size, B_L, N), "uint8")
            scales_buf = T.match_buffer(var_s, (E, group_size, N), "uint8")
            bias_buf = T.match_buffer(var_bias, (E, N), bias.dtype)
            e_ids_buf = T.match_buffer(var_e, (B_val,), "int32")
            lut_buf = T.match_buffer(var_lut, (16,), "float32")
            out_buf = T.match_buffer(var_o, (B_val, N), "float32")
            for vj_b in T.thread_binding((N + 127) // 128, thread="blockIdx.x"):
                for vi_b in T.thread_binding((B_val + 3) // 4, thread="blockIdx.y"):
                    for vj_t in T.thread_binding(128, thread="threadIdx.x"):
                        with T.block("compute"):
                            vj = vj_b * 128 + vj_t
                            x_sh = T.alloc_buffer((2, 4, 32), "float32", scope="shared")
                            lut_sh = T.alloc_buffer((16,), "float32", scope="shared")
                            acc = T.alloc_buffer((4,), "float32", scope="local")
                            ids = T.alloc_buffer((4,), "int32", scope="local")
                            if vj_t < 16:
                                lut_sh[vj_t] = lut_buf[vj_t]
                            for t in T.unroll(4):
                                vi = vi_b * 4 + t
                                if vi < B_val:
                                    ids[t] = e_ids_buf[vi]
                                    acc[t] = T.cast(bias_buf[ids[t], vj], "float32") if vj < N else 0.0
                                else:
                                    ids[t] = -1
                                    acc[t] = 0.0
                            t_ld, d_ld = vj_t // 32, vj_t % 32
                            if (vi_b * 4 + t_ld) < B_val:
                                x_sh[0, t_ld, d_ld] = x_buf[vi_b * 4 + t_ld, d_ld]
                            T.tvm_storage_sync("shared")
                            for g in T.serial(group_size):
                                if g + 1 < group_size:
                                    if (vi_b * 4 + t_ld) < B_val:
                                        x_sh[(g + 1) % 2, t_ld, d_ld] = x_buf[vi_b * 4 + t_ld, (g + 1) * 32 + d_ld]
                                if vj < N:
                                    for t in T.unroll(4):
                                        if ids[t] != -1:
                                            sc = T.exp2(
                                                T.cast(T.cast(scales_buf[ids[t], g, vj], "int32") - 127, "float32")
                                            )
                                            for b in T.unroll(B_L):
                                                bk = blocks_buf[ids[t], g, b, vj]
                                                w0 = lut_sh[T.cast(bk & T.uint8(0x0F), "int32")]
                                                w1 = lut_sh[T.cast(bk >> T.uint8(4), "int32")]
                                                acc[t] = (
                                                    acc[t]
                                                    + (x_sh[g % 2, t, b * 2] * w0 + x_sh[g % 2, t, b * 2 + 1] * w1) * sc
                                                )
                                T.tvm_storage_sync("shared")
                            for t in T.unroll(4):
                                if (vi_b * 4 + t) < B_val and vj < N:
                                    out_buf[vi_b * 4 + t, vj] = acc[t]

        return op.tensor_ir_op(
            _func,
            "mxfp4_moe_mlp2_prefill",
            args=[x, blocks, scales, bias, token_expert_ids, lut],
            out=Tensor.placeholder([B_val_outer, N], "float32"),
        )

    def mxfp4_moe_mlp1_decode(self, x, blocks, scales, bias, token_expert_ids, lut) -> Tensor:
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
            lut_buf = T.match_buffer(var_lut, (16,), "float32")
            out_buf = T.match_buffer(var_o, (4, N), "float32")
            for vj_b in T.thread_binding((N + 127) // 128, thread="blockIdx.x"):
                for vi_b in T.thread_binding(4, thread="blockIdx.y"):
                    for vj_t in T.thread_binding(128, thread="threadIdx.x"):
                        with T.block("compute"):
                            vj = vj_b * 128 + vj_t
                            x_sh = T.alloc_buffer((2, 32), "float32", scope="shared")
                            lut_sh = T.alloc_buffer((16,), "float32", scope="shared")
                            acc_g = T.alloc_buffer((1,), "float32", scope="local")
                            acc_l = T.alloc_buffer((1,), "float32", scope="local")
                            if vj_t < 16:
                                lut_sh[vj_t] = lut_buf[vj_t]
                            eid = e_ids_buf[vi_b]
                            if vj < N:
                                acc_g[0] = T.cast(bias_buf[eid, vj * 2], "float32")
                                acc_l[0] = T.cast(bias_buf[eid, vj * 2 + 1], "float32")
                            else:
                                acc_g[0] = 0.0
                                acc_l[0] = 0.0
                            if vj_t < 32:
                                x_sh[0, vj_t] = T.cast(x_buf[vi_b, vj_t], "float32")
                            T.tvm_storage_sync("shared")
                            for g in T.serial(group_size):
                                if g + 1 < group_size:
                                    if vj_t < 32:
                                        x_sh[(g + 1) % 2, vj_t] = T.cast(x_buf[vi_b, (g + 1) * 32 + vj_t], "float32")
                                if vj < N:
                                    sg = T.exp2(T.cast(T.cast(scales_buf[eid, g, vj * 2], "int32") - 127, "float32"))
                                    sl = T.exp2(
                                        T.cast(T.cast(scales_buf[eid, g, vj * 2 + 1], "int32") - 127, "float32")
                                    )
                                    for b in T.unroll(B_L):
                                        bk_g = blocks_buf[eid, g, b, vj * 2]
                                        bk_l = blocks_buf[eid, g, b, vj * 2 + 1]
                                        w0g = lut_sh[T.cast(bk_g & T.uint8(0x0F), "int32")]
                                        w1g = lut_sh[T.cast(bk_g >> T.uint8(4), "int32")]
                                        w0l = lut_sh[T.cast(bk_l & T.uint8(0x0F), "int32")]
                                        w1l = lut_sh[T.cast(bk_l >> T.uint8(4), "int32")]
                                        xx0, xx1 = x_sh[g % 2, b * 2], x_sh[g % 2, b * 2 + 1]
                                        acc_g[0] = acc_g[0] + (xx0 * w0g + xx1 * w1g) * sg
                                        acc_l[0] = acc_l[0] + (xx0 * w0l + xx1 * w1l) * sl
                                T.tvm_storage_sync("shared")
                            if vj < N:
                                xg = T.min(acc_g[0], T.float32(self.swiglu_limit))
                                xl = T.max(T.min(acc_l[0], T.float32(7.0)), T.float32(-7.0))
                                gv = xg / (T.float32(1.0) + T.exp(-T.float32(1.702) * xg))
                                out_buf[vi_b, vj] = gv * (xl + T.float32(1.0))

        return nn.tensor_ir_op(
            _func,
            "mxfp4_moe_mlp1_decode",
            args=[x, blocks, scales, bias, token_expert_ids, lut],
            out=Tensor.placeholder([4, N], "float32"),
        )

    def mxfp4_moe_mlp2_decode(self, x, blocks, scales, bias, token_expert_ids, lut) -> Tensor:
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
            x_buf = T.match_buffer(var_x, (4, self.hidden_size), "float32")
            blocks_buf = T.match_buffer(var_b, (E, group_size, B_L, N), "uint8")
            scales_buf = T.match_buffer(var_s, (E, group_size, N), "uint8")
            bias_buf = T.match_buffer(var_bias, (E, N), self.dtype)
            e_ids_buf = T.match_buffer(var_e, (4,), "int32")
            lut_buf = T.match_buffer(var_lut, (16,), "float32")
            out_buf = T.match_buffer(var_o, (4, N), "float32")
            for vj_b in T.thread_binding((N + 127) // 128, thread="blockIdx.x"):
                for vi_b in T.thread_binding(4, thread="blockIdx.y"):
                    for vj_t in T.thread_binding(128, thread="threadIdx.x"):
                        with T.block("compute"):
                            vj = vj_b * 128 + vj_t
                            x_sh = T.alloc_buffer((2, 32), "float32", scope="shared")
                            lut_sh = T.alloc_buffer((16,), "float32", scope="shared")
                            acc = T.alloc_buffer((1,), "float32", scope="local")
                            if vj_t < 16:
                                lut_sh[vj_t] = lut_buf[vj_t]
                            eid = e_ids_buf[vi_b]
                            acc[0] = T.cast(bias_buf[eid, vj], "float32") if vj < N else 0.0
                            if vj_t < 32:
                                x_sh[0, vj_t] = x_buf[vi_b, vj_t]
                            T.tvm_storage_sync("shared")
                            for g in T.serial(group_size):
                                if g + 1 < group_size:
                                    if vj_t < 32:
                                        x_sh[(g + 1) % 2, vj_t] = x_buf[vi_b, (g + 1) * 32 + vj_t]
                                if vj < N:
                                    sc = T.exp2(T.cast(T.cast(scales_buf[eid, g, vj], "int32") - 127, "float32"))
                                    for b in T.unroll(B_L):
                                        bk = blocks_buf[eid, g, b, vj]
                                        w0 = lut_sh[T.cast(bk & T.uint8(0x0F), "int32")]
                                        w1 = lut_sh[T.cast(bk >> T.uint8(4), "int32")]
                                        acc[0] = acc[0] + (x_sh[g % 2, b * 2] * w0 + x_sh[g % 2, b * 2 + 1] * w1) * sc
                                T.tvm_storage_sync("shared")
                            if vj < N:
                                out_buf[vi_b, vj] = acc[0]

        return nn.tensor_ir_op(
            _func,
            "mxfp4_moe_mlp2_decode",
            args=[x, blocks, scales, bias, token_expert_ids, lut],
            out=Tensor.placeholder([4, N], "float32"),
        )
