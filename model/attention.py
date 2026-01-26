from functools import partial
from math import log, pi, sqrt
from typing import Any, Literal

from mlc_llm.nn import PagedKVCache
from mlc_llm.support import logging
from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from .config import GPTOssConfig

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

            cos_freq, sin_freq, var_map = yarn_ftn(positions[seq_idx], freq_idx, rotary_dim, theta, x_dtype)

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

    def apply_rotation_to_qk(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        rope_ftn = partial(
            self._rope,
            rotary_dim=self.config.head_dim,
            theta=self.rope_theta,
        )
        q_embed = op.tensor_expr_op(rope_ftn, "rope_q", [q, positions])
        k_embed = op.tensor_expr_op(rope_ftn, "rope_k", [k, positions])

        return q_embed, k_embed

    def forward(
        self,
        x: nn.Tensor,
        paged_kv_cache: PagedKVCache,  # for MLC-LLM style
        query_positions: nn.Tensor,
        forward_to: Literal["prefill", "decode", "extend"],
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
            case "extend":
                # For continuing prefill: attend to both current tokens (self) and cached tokens (cross)
                o_self, lse_self = paged_kv_cache.self_attention(self.layer_idx, q=q, k=k, v=v, sm_scale=self.sm_scale)
                o_cross, lse_cross = paged_kv_cache.cross_attention(
                    self.layer_idx, q=q, v_head_dim=d, sm_scale=self.sm_scale
                )
                attention, lse_qk = paged_kv_cache.merge_attn_output_inplace(
                    o_self, lse_self, o_cross, lse_cross
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

        return x + t, paged_kv_cache
