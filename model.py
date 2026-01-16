import dataclasses
from functools import partial
from math import log, pi, prod, sqrt
from typing import Any, Literal

import mlc_llm.compiler_pass  # noqa: F401
from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold
from tvm import relax, te, tir
from tvm.relax.frontend import nn
from tvm.script import tir as T

from weights import FP4_VALUES

logger = logging.getLogger(__name__)


def yarn_find_correction_dim(
    num_rotations: int,
    d: tir.Var,
    theta: float,
    max_position_embeddings: int,
):
    """Inverse dim formula to find dim based on number of rotations"""
    return (d * log(max_position_embeddings / (num_rotations * 2 * pi))) / (2 * log(theta))


def yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    d: tir.Var,
    theta: float,
    max_position_embeddings: int,
):
    """Find the correction range based on the number of rotations"""
    low = yarn_find_correction_dim(low_rot, d, theta, max_position_embeddings)
    high = yarn_find_correction_dim(high_rot, d, theta, max_position_embeddings)

    return tir.max(low, 0), tir.min(high, d - 1)


def rope_freq_yarn(
    s: tir.Var,
    d: tir.Var,
    d_range: int,
    theta: float,
    dtype: str,
    original_max_position_embeddings: int,
    scaling_factor: float,
    beta_fast: int,
    beta_slow: int,
):  # pylint: disable=too-many-arguments, too-many-locals
    """Compute the inverse frequency of RoPE for yarn RoPE scaling."""

    exponent = d * 2 % d_range / tir.const(d_range, "float32")
    freq_power = tir.power(theta, exponent)
    freq_extra = tir.const(1, "float32") / freq_power
    freq_inter = tir.const(1, "float32") / (scaling_factor * freq_power)

    low, high = yarn_find_correction_range(beta_fast, beta_slow, d_range, theta, original_max_position_embeddings)
    high = tir.if_then_else(low == high, high + 0.001, high)
    inv_freq_mask = tir.const(1, "float32") - tir.max(tir.min((d - low) / (high - low), 1.0), 0.0).astype("float32")

    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
    freq = s * inv_freq
    freq_var = tir.Var("freq", "float32")
    cos_freq = tir.cos(freq_var).astype(dtype)
    sin_freq = tir.sin(freq_var).astype(dtype)

    return cos_freq, sin_freq, {freq_var: freq}


# fixed structure for mlc-llm style
@dataclasses.dataclass
class GPTOssConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the gpt-oss model."""

    context_window_size: int = 40960
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    pipeline_parallel_stages: int = 1
    dtype: str = "bfloat16"
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    num_hidden_layers: int = 36
    num_experts: int = 128
    num_experts_per_tok: int = 4  # in gpt-oss, `experts_per_token`
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window_size: int = 128
    rope_theta: int | float = 150000
    rope_scaling: dict | None = None
    swiglu_limit: float = 7.0

    # to use alternating dense-sliding Attention
    sliding_window_pattern: int = 2

    def __post_init__(self):
        if self.rope_scaling is None:
            self.rope_scaling = {
                # "rope_type": "yarn",
                "rope_theta": float(self.rope_theta),
                "factor": 32.0,  # in gpt-oss, `rope_scaling_factor`
                "beta_fast": 32.0,  # in gpt-oss, `rope_ntk_beta`
                "beta_slow": 1.0,  # in gpt-oss, `rope_ntk_alpha`
                "truncate": False,
                "original_max_position_embeddings": 4096,  # in gpt-oss, `initial_context_length`
            }

        if "quantization_config" in self.kwargs:
            quantization_config = self.kwargs.get("quantization_config")
            # FIXME(if needed)
            pass

        # context_window_size = 40960, fixed
        if self.context_window_size == 0:
            for name in ["max_position_embeddings", "max_sequence_length"]:
                if name in self.kwargs:
                    self.context_window_size = self.kwargs.pop(name)
                    logger.info(
                        "%s not found in config.json. Falling back to %s (%d)",
                        bold("context_window_size"),
                        bold(name),
                        self.context_window_size,
                    )
                    break
            else:
                raise ValueError(
                    "Unable to determine the maximum sequence length, because none of "
                    "`context_window_size`, `max_position_embeddings` or `max_sequence_length` is "
                    "provided in `config.json`."
                )
            if self.num_key_value_heads == 0:
                self.num_key_value_heads = self.num_attention_heads
            if self.head_dim == 0:
                self.head_dim = self.hidden_size // self.num_attention_heads
            assert self.num_attention_heads % self.num_key_value_heads == 0
            # TODO: need to check `8192`; to another value
            if self.prefill_chunk_size == 0:
                self.prefill_chunk_size = min(self.context_window_size, 8192)
                logger.info(
                    "%s defaults to %d",
                    bold("prefill_chunk_size"),
                    self.prefill_chunk_size,
                )
            elif self.prefill_chunk_size > self.context_window_size:
                logger.info(
                    "Overriding %s from %d to %d",
                    bold("prefill_chunk_size"),
                    self.prefill_chunk_size,
                    min(self.context_window_size, 8192),
                )
                self.prefill_chunk_size = min(self.context_window_size, 8192)


class AttentionBlock(nn.Module):
    def __init__(self, config: GPTOssConfig, layer_idx: int = 0, dtype: str | None = None):
        assert config
        self.config = config
        if dtype is None:
            dtype = config.dtype  # default is "bfloat16"
        self.dtype = dtype
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window_size if layer_idx % 2 == 0 else 0
        self.rope_theta = config.rope_theta

        self.sinks = nn.Parameter((config.num_attention_heads,), dtype=dtype)
        self.norm = nn.RMSNorm(config.hidden_size, axes=-1, bias=False, dtype=dtype)
        qkv_dim = config.head_dim * (config.num_attention_heads + 2 * config.num_key_value_heads)
        self.qkv = nn.Linear(config.hidden_size, qkv_dim, bias=True, dtype=dtype)
        self.out = nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            bias=True,
            dtype=dtype,
        )
        self.sm_scale = 1 / sqrt(config.head_dim)
        self.layer_idx = layer_idx  # for MLC-LLM PagedKVCache

    def _rope(
        self,
        x: te.Tensor,
        positions: te.Tensor,
        rotary_dim: int,
        theta: float | tir.Var,
        rope_scaling: dict[str, Any] | None = None,
    ):
        x_dtype = x.dtype
        if rope_scaling is None:
            rope_scaling = {
                "rope_type": "yarn",
                "rope_theta": float(self.rope_theta),
                "factor": 32.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "truncate": False,
                "original_max_position_embeddings": 4096,
            }

        if isinstance(theta, tir.Var):
            theta = rope_scaling.get("rope_theta", self.rope_theta)

        yarn_ftn = partial(
            rope_freq_yarn,
            original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
            scaling_factor=rope_scaling["factor"],
            beta_fast=rope_scaling["beta_fast"],
            beta_slow=rope_scaling["beta_slow"],
        )

        def _rope_compute(batch_idx: tir.Var, seq_idx: tir.Var, head_idx: tir.Var, dim_idx: tir.Var):
            # suppose that rotary_dim % 2 == 0
            rotary_half = rotary_dim // 2
            freq_idx = dim_idx % rotary_half

            cos_freq, sin_freq, var_map = yarn_ftn(positions[seq_idx], freq_idx, rotary_dim, theta, "float32")

            # dim_idx: the index in the interval [0, head_dim)
            # z = r^(i*theta) = cos(theta) + i * sin(theta)
            # if dim_idx is in the half left of interval,
            # then dim_idx is assigned to real(z)
            # else, dim_idx is assigned to imag(z)
            is_imaginary = dim_idx >= rotary_half

            # is_imaginary => we already obtained y, so need to find x (- rotary_half)
            # not is_imaginary => we already obtained x, so need to find y (+ rotary_half)
            partner_d = tir.if_then_else(
                is_imaginary,
                dim_idx - rotary_half,
                dim_idx + rotary_half,
            )

            scale_val = rope_scaling.get("factor", 1.0)
            conc_val = 0.1 * log(scale_val) + 1.0 if scale_val > 1.0 else 1.0

            concentration = tir.const(conc_val, "float32")
            cos_val = cos_freq * concentration
            sin_val = sin_freq * concentration

            val = x[batch_idx, seq_idx, head_idx, dim_idx]
            partner_val = x[batch_idx, seq_idx, head_idx, partner_d]

            out_rope = tir.if_then_else(
                is_imaginary,
                partner_val * sin_val + val * cos_val,  # for y, y_rotated = x * sin + y * cos
                val * cos_val - partner_val * sin_val,  # for x, x_rotated = x * cos - y * sin
            )

            for var, value in var_map.items():
                out_rope = tir.Let(var, value, out_rope)

            return out_rope.astype(x_dtype)

        return te.compute(x.shape, _rope_compute, name="yarn_rope")

    def apply_rotation_to_qk(self, q: nn.Tensor, k: nn.Tensor, positions: nn.Tensor) -> tuple[nn.Tensor, nn.Tensor]:
        rope_ftn = partial(
            self._rope,
            rotary_dim=self.config.head_dim,
            theta=self.rope_theta,
        )
        q_embed = nn.op.tensor_expr_op(rope_ftn, "rope_q", [q, positions])
        k_embed = nn.op.tensor_expr_op(rope_ftn, "rope_k", [k, positions])

        return q_embed, k_embed

    def forward(
        self,
        x: nn.Tensor,
        paged_kv_cache: PagedKVCache,  # for MLC-LLM style
        query_positions: nn.Tensor,
        forward_to: Literal["prefill", "decode"],
    ) -> tuple[nn.Tensor, PagedKVCache]:
        d = self.head_dim
        n_q_heads = self.num_attention_heads
        n_kv_heads = self.num_key_value_heads

        t = self.norm(x)  # PreNorm

        # from `CausalLM.prefill()`,
        # get t of shape (batch_size, num_tokens, d)
        batch_size = 1

        # [1, seq_len, 2880]
        if len(t.shape) == 3:
            _, num_tokens, _ = t.shape  # batch_size, num_tokens, _; x: from embedding layer; batch size is 1 for test
        else:
            # [seq_len, 2880]
            num_tokens, _ = t.shape

        qkv = self.qkv(t)  # fused qkv
        qkv = nn.op.reshape(
            qkv,
            (batch_size, num_tokens, n_q_heads + n_kv_heads + n_kv_heads, d),
        )
        q, k, v = nn.op.split(qkv, [n_q_heads, n_q_heads + n_kv_heads], axis=2)

        q, k = self.apply_rotation_to_qk(q, k, query_positions)
        paged_kv_cache = paged_kv_cache.append_mha_kv(layer_id=self.layer_idx, k=k, v=v)

        # `lse_qk` is always f32
        attention, lse_qk = None, None
        match forward_to:
            case "prefill":
                attention, lse_qk = paged_kv_cache.self_attention(self.layer_idx, q=q, k=k, v=v, sm_scale=self.sm_scale)
            case "decode":
                attention, lse_qk = paged_kv_cache.cross_attention(
                    self.layer_idx, q=q, v_head_dim=d, sm_scale=self.sm_scale
                )
            case _:
                raise ValueError(f"forward_to {forward_to} not supported")

        # since TVM use log_2 for compute LSE internally,
        # we need to multiply log(2), natural log 2
        # cf. L2414 in `kv_cache.py`
        lse_qk = log(2) * lse_qk
        sinks_f32 = nn.op.astype(self.sinks, "float32")
        sigmoid_scale = nn.op.sigmoid(lse_qk - sinks_f32)

        sigmoid_scale = nn.op.unsqueeze(sigmoid_scale, dim=-1)
        sigmoid_scale = nn.op.astype(sigmoid_scale, self.dtype)

        attention = nn.op.multiply(attention, sigmoid_scale)

        t = nn.op.reshape(
            attention,
            (batch_size, num_tokens, n_q_heads * d),
        )

        t = self.out(t)
        t = t.astype(self.dtype)

        return x + t, paged_kv_cache


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
        mxfp4_dtype = "uint8"
        mxfp4_group_size = 90
        mxfp4_block_length = 16  # bytes_per_block

        self.mlp1_weight_blocks = nn.Parameter(
            (
                self.num_experts,
                config.intermediate_size * 2 // self.world_size,
                mxfp4_group_size,
                mxfp4_block_length,
            ),
            dtype=mxfp4_dtype,
        )
        self.mlp1_weight_scales = nn.Parameter(
            (
                self.num_experts,
                config.intermediate_size * 2 // self.world_size,
                mxfp4_group_size,
            ),
            dtype=mxfp4_dtype,
        )
        self.mlp2_weight_blocks = nn.Parameter(
            (
                self.num_experts,
                config.hidden_size,
                mxfp4_group_size,
                mxfp4_block_length,
            ),
            dtype=mxfp4_dtype,
        )
        self.mlp2_weight_scales = nn.Parameter(
            (
                self.num_experts,
                config.hidden_size,
                mxfp4_group_size,
            ),
            dtype=mxfp4_dtype,
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
        import numpy as np

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
        weight_tensor: nn.Tensor,
        bias_tensor: nn.Tensor,
        safetensor_dtype: str | None = None,
    ) -> nn.Tensor:
        # input_dtype -> f32
        einsum_dtype = "float32"
        if safetensor_dtype is None:
            safetensor_dtype = self.dtype

        # here, len(input_tensor.shape) is 2 or 3
        # (seq_len, 2880); normed tensor in 1st operation
        # (seq_len, top_k, 2880)

        seq_len = input_tensor.shape[0]
        in_features = input_tensor.shape[-1]
        input_shape = input_tensor.shape
        input_is_3d = len(input_shape) == 3

        _, experts_per_token = expert_indices.shape  # (seq_len, 4)
        num_all_experts, out_features, _ = weight_tensor.shape  # (32, 5760, 2880) | (32, 2880, 2880)

        @T.prim_func(private=True)
        def _einsum_matrix(
            x_handle: T.handle,
            e_indices: T.Buffer((seq_len, experts_per_token), "int32"),
            weight: T.Buffer((num_all_experts, out_features, in_features), safetensor_dtype),
            bias: T.Buffer((num_all_experts, out_features), safetensor_dtype),
            out_handle: T.handle,
        ):
            T.func_attr({"op_pattern": 4, "tir.noalias": True})  # pattern #4 is kOutEWiseFusable (matmul)
            x = T.match_buffer(x_handle, input_shape, input_tensor.dtype)
            out = T.match_buffer(out_handle, (seq_len, experts_per_token, out_features), einsum_dtype)

            for s, r, c in T.grid(seq_len, experts_per_token, out_features):
                # s, r, c for seq_idx, expert_rank, out_channel, resp.
                with T.block("compute_matmul"):
                    # S for spatial axis in the argument `kinds`
                    seq_idx, expert_rank, out_idx = T.axis.remap("SSS", [s, r, c])
                    e_idx = e_indices[seq_idx, expert_rank]

                    sum_buffer = T.alloc_buffer((1,), dtype=einsum_dtype, scope="local")
                    sum_buffer[0] = T.cast(bias[e_idx, out_idx], einsum_dtype)

                    for in_idx in T.serial(in_features):
                        if input_is_3d:
                            sum_buffer[0] += T.cast(x[seq_idx, expert_rank, in_idx], einsum_dtype) * T.cast(
                                weight[e_idx, out_idx, in_idx],
                                einsum_dtype,
                            )
                        else:
                            sum_buffer[0] += T.cast(x[seq_idx, in_idx], einsum_dtype) * T.cast(
                                weight[e_idx, out_idx, in_idx],
                                einsum_dtype,
                            )
                    out[seq_idx, expert_rank, out_idx] = sum_buffer[0]

        return nn.tensor_ir_op(
            _einsum_matrix,
            "moe_gemv_einsum",
            args=[input_tensor, expert_indices, weight_tensor, bias_tensor],
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
        expert_weights, expert_indices = nn.op.topk(g, k=self.experts_per_token, axis=-1, largest=True)
        expert_weights = nn.op.softmax(expert_weights, axis=1)

        # MLP #1
        # self.mlp1_weight: (32, 5760, 2880)
        # self.mlp1_bias: (32, 5760)
        mlp1_weight = self.__dequantize_mxfp4_weight(self.mlp1_weight_blocks, self.mlp1_weight_scales)
        t = self.run_einsum(t, expert_indices, mlp1_weight, self.mlp1_bias)  # (seq_len, top_k, 5760)
        del mlp1_weight
        t = swiglu(t, limit=self.swiglu_limit, out_dtype="float32")  # (seq_len, top_k, 2880)

        # MLP #2
        # self.mlp2_weight: (32, 2880, 2880)
        # self.mlp2_bias: (32, 2880)
        mlp2_weight = self.__dequantize_mxfp4_weight(self.mlp2_weight_blocks, self.mlp2_weight_scales)
        t = self.run_einsum(t, expert_indices, mlp2_weight, self.mlp2_bias)
        del mlp2_weight

        # Weighted sum of experts
        t = self.run_gating(t, expert_weights)
        t = nn.op.reshape(t, x.shape)  # this resolves inconsistent buffer error
        t = t.astype(self.dtype)

        return nn.op.reshape(x + t, (b, seq_len, dim))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTOssConfig, layer_idx: int, dtype: str | None = None):
        assert config
        if dtype is None:
            dtype = config.dtype
        self.attn = AttentionBlock(config, layer_idx, dtype=dtype)
        self.mlp = MLPBlock(config, dtype=dtype)

    def forward(
        self,
        x: nn.Tensor,
        paged_kv_cache: PagedKVCache,
        query_positions: nn.Tensor,
        forward_to: Literal["prefill", "decode"],
    ) -> tuple[nn.Tensor, PagedKVCache]:
        x, paged_kv_cache = self.attn(x, paged_kv_cache, query_positions, forward_to)
        x = self.mlp(x)

        return x, paged_kv_cache


class Transformer(nn.Module):
    def __init__(self, config: GPTOssConfig, dtype: str | None = None):
        assert config
        if dtype is None:
            dtype = config.dtype
        self.dtype = dtype
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.block = nn.ModuleList(
            [TransformerBlock(config, layer_idx, dtype=dtype) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, bias=False, dtype=dtype)
        # to use MLC-LLM style, this part is in `GPTOssForCausalLM()`
        # self.unembedding = nn.Linear(
        #     config.hidden_size,
        #     config.vocab_size,
        #     bias=False,
        #     dtype="bfloat16",
        # )

    def forward(
        self, x: nn.Tensor, paged_kv_cache: PagedKVCache, forward_to: Literal["prefill", "decode"]
    ) -> tuple[nn.Tensor, PagedKVCache]:
        query_positions = paged_kv_cache.get_query_positions(x.shape[0] * x.shape[1])
        # x = self.embedding(x)  # in embed() function

        for block in self.block:
            x, paged_kv_cache = block(x, paged_kv_cache, query_positions, forward_to)
        x = self.norm(x)

        # to use MLC-LLM style, this part is in `GPTOssForCausalLM()`
        # x = self.unembedding(x)

        return x, paged_kv_cache


class GPTOssForCausalLM(nn.Module):
    def __init__(self, config: GPTOssConfig, dtype: str | None = None):
        self.dtype = config.dtype if dtype is None else dtype
        self.model = Transformer(config, dtype=self.dtype)
        self.unembedding = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=self.dtype,
        )

        # from config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        # since we use a single GPU, define this arg to 1
        self.tensor_parallel_shards = config.tensor_parallel_shards

        # other setting
        self.sw_pattern = config.sliding_window_pattern

    def to(self, dtype: str | None = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def _get_logits(self, hidden_states: nn.Tensor) -> nn.Tensor:
        logits = self.unembedding(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")

        return logits

    def embed(self, input_ids: nn.Tensor) -> nn.Tensor:
        if self.tensor_parallel_shards > 1:
            input_ids = nn.op.ccl_broadcast_from_worker0(input_ids)
        embed_ = self.model.embedding(input_ids)

        return nn.op.reshape(embed_, (1, -1, self.hidden_size))

    def prefill(self, input_embed: nn.Tensor, paged_kv_cache: PagedKVCache) -> tuple[nn.Tensor, PagedKVCache]:
        op_ext.configure()

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape  # type: ignore
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        seq_len = input_embed.shape[1]
        hidden_states, paged_kv_cache = self.model(input_embed, paged_kv_cache, "prefill")
        hidden_states = nn.op.reshape(hidden_states, (1, seq_len, -1))
        hidden_states = nn.op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self._get_logits(hidden_states)

        return logits, paged_kv_cache

    def decode(self, input_embed: nn.Tensor, paged_kv_cache: PagedKVCache) -> tuple[nn.Tensor, PagedKVCache]:
        op_ext.configure()

        hidden_states, paged_kv_cache = self.model(input_embed, paged_kv_cache, "decode")
        logits = self._get_logits(hidden_states)

        return logits, paged_kv_cache

    def create_paged_kv_cache(  # pylint: disable=too-many-arguments
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
    ) -> PagedKVCache:
        attention_layer_setting: list[Literal["mha", "mha_sliding"]] = [
            "mha_sliding" if i % self.sw_pattern == 0 else "mha" for i in range(self.num_hidden_layers)
        ]

        return PagedKVCache.create_generic(
            attn_kind=attention_layer_setting,
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads // self.tensor_parallel_shards,
            num_key_value_heads=self.num_key_value_heads // self.tensor_parallel_shards,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            rope_mode=RopeMode.NONE,
            rope_scale=1,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
            dtype=self.dtype,
        )

    def get_default_spec(self):
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
