import dataclasses
import logging
from typing import Any

from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


# fixed structure for mlc-llm style
@dataclasses.dataclass
class GPTOssConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the gpt-oss model."""

    context_window_size: int = 0
    prefill_chunk_size: int = 8192
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
                "rope_theta": float(self.rope_theta),
                "factor": 32.0,  # in gpt-oss, `rope_scaling_factor`
                "beta_fast": 32.0,  # in gpt-oss, `rope_ntk_beta`
                "beta_slow": 1.0,  # in gpt-oss, `rope_ntk_alpha`
                "truncate": False,
                "original_max_position_embeddings": 4096,  # in gpt-oss, `initial_context_length`
            }

        assert self.num_attention_heads > 0
        assert self.num_key_value_heads > 0
        assert self.head_dim > 0
        assert self.num_attention_heads % self.num_key_value_heads == 0

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
            # model originally supports 131_072(128k), but default value is capped to 40k.
            self.context_window_size = 40960
            logger.info(
                "Unable to determine the maximum sequence length, because none of "
                "`context_window_size`, `max_position_embeddings` or `max_sequence_length` is "
                "provided in `config.json`. So it is set to the default value %d.",
                self.context_window_size,
            )

        self.prefill_chunk_size = min(self.context_window_size, self.prefill_chunk_size)
