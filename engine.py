from pathlib import Path

import mlc_llm.compiler_pass  # noqa: F401
import numpy as np
import tvm
import tvm_ffi
from tvm import relax
from tvm.target import Target

from model import GPTOssConfig, GPTOssForCausalLM
from weights import TVMCheckpoint, TVMDeviceStr


class Engine:
    next_seq_id = 0

    def __init__(self, model_path: str | Path, target: TVMDeviceStr):
        self.model_path = Path(model_path)
        self.config = GPTOssConfig.from_file(self.model_path / "config.json")
        self.model = GPTOssForCausalLM(self.config)
        self.device = tvm.device(target, 0)

        # compile
        ex, params = self._compile_module(target)

        # load model
        self._vm = relax.VirtualMachine(ex, self.device, profile=False)
        self.params = TVMCheckpoint(path=model_path, target_device=target).load_packed_params(params)

        # get functions
        self._f_create_kv_cache = self._vm["create_tir_paged_kv_cache"]

        self._f_embed = self._vm["embed"]
        self._f_prefill = self._vm["prefill"]
        self._f_decode = self._vm["decode"]
        self._f_extend = self._vm["extend"]

        self._f_sample = tvm.get_global_func("vm.builtin.sample_top_p_from_logits")
        self.sample_parameters = {"temperature": 0.7, "top_p": 0.9, "random_seed": 0.5}

        self._f_kv_cache_clear = tvm.get_global_func("vm.builtin.kv_state_clear")
        self._f_add_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
        self._f_enable_sliding_window_for_seq = tvm.get_global_func(
            "vm.builtin.attention_kv_cache_enable_sliding_window_for_seq"
        )
        self._f_begin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
        self._f_end_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
        self._f_kv_popn = tvm.get_global_func("vm.builtin.kv_state_popn")
        self._f_kv_get_seq_len = tvm.get_global_func("vm.builtin.attention_kv_cache_get_total_sequence_length")

        self.paged_kv_cache = self._create_paged_kv_cache()
        self._warmup()

    def _compile_module(self, target: TVMDeviceStr):
        irmodule, params, _ = self.model.export_tvm(  # type: ignore[misc]
            spec=self.model.get_default_spec(),
            allow_extern=True,
        )

        metadata = {
            "model_type": "gpt-oss",
            "context_window_size": self.config.context_window_size,
            "sliding_window_size": self.config.sliding_window_size,
            "attention_sink_size": 0,
            "prefill_chunk_size": self.config.prefill_chunk_size,  # type: ignore
            "tensor_parallel_shards": self.config.tensor_parallel_shards,  # type: ignore
            "pipeline_parallel_stages": self.config.pipeline_parallel_stages,
            "kv_state_kind": "kv_cache",
            "max_batch_size": 1,
            "rope_theta": self.config.rope_theta,
        }

        with Target.from_device(target):
            target = Target.current()
            pipeline = relax.pipeline.get_pipeline(
                "mlc_llm",
                target=target,
                metadata=metadata,
                variable_bounds={
                    "total_seq_len": self.config.context_window_size,
                    "seq_len": self.config.prefill_chunk_size,
                    "batch_size": 1,
                },
            )
            print("Applying pipeline..")
            mod = pipeline(irmodule)

            print("Compiling..")
            ex = tvm.compile(mod, target)

        return ex, params

    def _create_paged_kv_cache(self):
        return self._f_create_kv_cache(  # pylint: disable=too-many-arguments
            tvm.runtime.ShapeTuple([1]),  # max_batch_size
            tvm.runtime.ShapeTuple(
                [self.config.context_window_size]
            ),  # max_total_seq_len: self.config.context_window_size
            tvm.runtime.ShapeTuple(
                [self.config.prefill_chunk_size]
            ),  # prefill_chunk_size: self.config.prefill_chunk_size
            tvm.runtime.ShapeTuple([16]),  # page_size
            tvm.runtime.ShapeTuple([1]),  # support_sliding_window
        )

    def clear_kv_cache(self):
        self._f_kv_cache_clear(self.paged_kv_cache)

    def begin_sequence(self, sliding_window_size: int | None = None, sink_size: int | None = None):
        seq_id = self.__class__.next_seq_id
        self.__class__.next_seq_id += 1
        self._f_add_sequence(self.paged_kv_cache, seq_id)
        sliding_window_size = sliding_window_size or self.config.sliding_window_size
        sink_size = sink_size or 0
        if sliding_window_size > 0:
            self._f_enable_sliding_window_for_seq(
                self.paged_kv_cache,
                seq_id,
                sliding_window_size,
                sink_size,
            )
        return seq_id

    def popn(self, seq_id: int, n: int):
        self._f_kv_popn(self.paged_kv_cache, seq_id, n)

    def get_kv_total_seq_len(self) -> int:
        return self._f_kv_get_seq_len(self.paged_kv_cache)

    def prefill(self, input_tokens: np.ndarray, sequence_id: int = 0) -> tvm_ffi.Tensor:
        input_tokens = tvm.runtime.tensor(input_tokens, device=self.device)
        input_embed: tvm_ffi.Tensor = self._f_embed(input_tokens, self.params)
        input_length = input_embed.shape[1]

        self._f_begin_forward(
            self.paged_kv_cache,
            tvm.runtime.ShapeTuple([sequence_id]),  # sequence id
            tvm.runtime.ShapeTuple([input_length]),  # added all tokens to kv cache
        )
        logits, self.paged_kv_cache = self._f_prefill(input_embed, self.paged_kv_cache, self.params)
        self._f_end_forward(self.paged_kv_cache)
        return logits

    def decode(self, input_tokens: np.ndarray, sequence_id: int = 0) -> tvm_ffi.Tensor:
        input_tokens = tvm.runtime.tensor(input_tokens, device=self.device)
        input_embed: tvm_ffi.Tensor = self._f_embed(input_tokens, self.params)

        self._f_begin_forward(
            self.paged_kv_cache,
            tvm.runtime.ShapeTuple([sequence_id]),  # sequence id
            tvm.runtime.ShapeTuple([1]),  # added all tokens to kv cache
        )
        logits, self.paged_kv_cache = self._f_decode(input_embed, self.paged_kv_cache, self.params)
        self._f_end_forward(self.paged_kv_cache)
        return logits

    def extend(self, input_tokens: np.ndarray, sequence_id: int = 0) -> tvm_ffi.Tensor:
        """Extend mode: prefill new tokens while attending to both new tokens and previously cached context."""
        input_tokens = tvm.runtime.tensor(input_tokens, device=self.device)
        input_embed: tvm_ffi.Tensor = self._f_embed(input_tokens, self.params)
        input_length = input_embed.shape[1]

        self._f_begin_forward(
            self.paged_kv_cache,
            tvm.runtime.ShapeTuple([sequence_id]),  # sequence id
            tvm.runtime.ShapeTuple([input_length]),  # added all tokens to kv cache
        )
        logits, self.paged_kv_cache = self._f_extend(input_embed, self.paged_kv_cache, self.params)
        self._f_end_forward(self.paged_kv_cache)
        return logits

    def sample(self, logits: tvm_ffi.Tensor) -> int:
        new_token_id: int = self._f_sample(logits, *self.sample_parameters.values())
        return new_token_id

    def _warmup(self):
        print("Warming up..")
        dummy_input = np.array([0], dtype="int32")
        seq_id = self.begin_sequence()
        self.prefill(dummy_input, seq_id)
        self.decode(dummy_input, seq_id)
        self.clear_kv_cache()
        self.__class__.next_seq_id = 0
