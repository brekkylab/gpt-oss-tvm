from typing import Literal

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from tvm import te, tir
from tvm.relax.frontend import nn

from .attention import AttentionBlock
from .config import GPTOssConfig
from .mlp import MLPBlock


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
        forward_to: Literal["prefill", "decode", "extend"],
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
        self, x: nn.Tensor, paged_kv_cache: PagedKVCache, forward_to: Literal["prefill", "decode", "extend"]
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

    def extend(self, input_embed: nn.Tensor, paged_kv_cache: PagedKVCache) -> tuple[nn.Tensor, PagedKVCache]:
        """Extend mode: prefill new tokens while attending to both new tokens and cached context."""
        op_ext.configure()

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape  # type: ignore
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        seq_len = input_embed.shape[1]
        hidden_states, paged_kv_cache = self.model(input_embed, paged_kv_cache, "extend")
        hidden_states = nn.op.reshape(hidden_states, (1, seq_len, -1))
        hidden_states = nn.op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
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
            "extend": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
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
