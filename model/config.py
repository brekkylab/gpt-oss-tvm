import dataclasses
import logging
from typing import Any, Dict, Optional, Union

from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


# fixed structure for mlc-llm style
@dataclasses.dataclass
class GPTOssConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the gpt-oss model."""

    context_window_size: int = 40960
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    pipeline_parallel_stages: int = 1
    dtype: str = "bfloat16"
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

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
    rope_theta: Union[int, float] = 150000
    rope_scaling: Optional[Dict] = None
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
