import time
from pathlib import Path
from typing import Callable

import numpy as np
import tvm
import tvm_ffi
from tvm import relax
from tvm.target import Target

from model import GPTOssConfig, GPTOssForCausalLM
from weights import TVMCheckpoint


class Engine:
    next_seq_id = 0

    def __init__(
        self,
        model_path: str | Path,
        target: str = "metal",
    ):
        self.model_path = Path(model_path)
        self.config = GPTOssConfig.from_file(self.model_path / "config.json")
        self.model = GPTOssForCausalLM(self.config)
        self.device = tvm.device(target, 0)

        # compile
        ex, params = self._compile_module(target)

        # load model
        self._vm = relax.VirtualMachine(ex, self.device)
        self.params = TVMCheckpoint(path=model_path, target_device=target).load_packed_params(params)

        # get functions
        self._f_create_kv_cache = self._vm["create_tir_paged_kv_cache"]

        self._f_embed = self._vm["embed"]
        self._f_prefill = self._vm["prefill"]
        self._f_decode = self._vm["decode"]

        self._f_sample = tvm.get_global_func("vm.builtin.sample_top_p_from_logits")
        self.sample_parameters = {"temperature": 0.7, "top_p": 0.9, "random_seed": 0.5}

        self._f_kv_cache_clear = tvm.get_global_func("vm.builtin.kv_state_clear")
        self._f_add_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
        self._f_enable_sliding_window_for_seq = tvm.get_global_func(
            "vm.builtin.attention_kv_cache_enable_sliding_window_for_seq"
        )
        self._f_begin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
        self._f_end_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")

        self.paged_kv_cache = self._create_paged_kv_cache()
        self.clear_kv_cache()

    def _compile_module(
        self,
        target: str,
    ):
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
            mod = pipeline(irmodule)
            print("Pipeline applied.")

            ex = tvm.compile(mod, target)
        print("Compile Done.")
        return ex, params

    def _create_paged_kv_cache(self):
        # TODO: replace hard-coded values
        # TODO: for now, config value causes `buf != nil` error
        return self._f_create_kv_cache(  # pylint: disable=too-many-arguments
            tvm.runtime.ShapeTuple([1]),  # max_batch_size
            tvm.runtime.ShapeTuple([8192]),  # max_total_seq_len: self.config.context_window_size
            tvm.runtime.ShapeTuple([8192]),  # prefill_chunk_size: self.config.prefill_chunk_size
            tvm.runtime.ShapeTuple([16]),  # page_size
            tvm.runtime.ShapeTuple([1]),  # support_sliding_window
        )

    def clear_kv_cache(self):
        self._f_kv_cache_clear(self.paged_kv_cache)

    def begin_sequence(
        self,
        sliding_window_size: int = -1,
        sink_size: int = 0,
    ):
        seq_id = self.__class__.next_seq_id
        self.__class__.next_seq_id += 1
        self._f_add_sequence(self.paged_kv_cache, seq_id)
        if sliding_window_size > 0:
            self._f_enable_sliding_window_for_seq(
                self.paged_kv_cache,
                seq_id,
                sliding_window_size,
                sink_size,
            )
        return seq_id

    def forward(
        self,
        input_tokens: np.ndarray,
        f_forward: Callable,
        sequence_id: int = 0,
    ) -> tvm_ffi.Tensor:
        input_embed: tvm_ffi.Tensor = self._f_embed(tvm.runtime.tensor(input_tokens, device=self.device), self.params)
        input_length = input_embed.shape[1]

        self._f_begin_forward(
            self.paged_kv_cache,
            tvm.runtime.ShapeTuple([sequence_id]),  # sequence id
            tvm.runtime.ShapeTuple([input_length]),  # added all tokens to kv cache
        )
        logits, self.paged_kv_cache = f_forward(input_embed, self.paged_kv_cache, self.params)
        self._f_end_forward(self.paged_kv_cache)

        return logits

    def prefill(
        self,
        input_tokens: np.ndarray,
        sequence_id: int = 0,
    ) -> tvm_ffi.Tensor:
        start_time = time.perf_counter()
        logits = self.forward(input_tokens, self._f_prefill, sequence_id)
        end_time = time.perf_counter()
        print(f"prefill: {end_time - start_time}s")
        return logits

    def decode(
        self,
        input_tokens: np.ndarray,
        sequence_id: int = 0,
    ) -> tvm_ffi.Tensor:
        start_time = time.perf_counter()
        logits = self.forward(input_tokens, self._f_decode, sequence_id)
        end_time = time.perf_counter()
        print(f"decode: {end_time - start_time}s")
        return logits

    def sample(self, logits: tvm_ffi.Tensor) -> int:
        start_time = time.perf_counter()
        new_token_id: int = self._f_sample(logits, *self.sample_parameters.values())
        end_time = time.perf_counter()
        print(f"sample: {end_time - start_time}s")
        return new_token_id
