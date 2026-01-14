from pathlib import Path
import random
from typing import Callable, Literal

from gpt_oss.torch.weights import Checkpoint
from ml_dtypes import bfloat16  # noqa: F401
import numpy as np
import torch

import tvm
from tvm import relax
from tvm.target import Target

from model import GPTOssConfig, GPTOssForCausalLM


class Engine():
    next_seq_id = 0

    def __init__(
        self,
        model_path: str | Path,
        target: str = "metal",
        param_dtype: Literal["bfloat16", "float32"] ="bfloat16",
    ):
        self.model_path = Path(model_path)
        self.config = GPTOssConfig.from_file(self.model_path / "config.json")
        self.model = GPTOssForCausalLM(self.config)
        self.device = tvm.device(target, 0)

        # compile
        ex, params = self._compile_module(target)
        loaded_params = self._load_params_from_path([k for k, _ in params], dtype=param_dtype)
        self.params = [tvm.runtime.tensor(param, device=self.device) for param in loaded_params]
        self._vm = self._create_vm(ex)

        # get functions
        self._f_create_kv_cache = self._vm["create_tir_paged_kv_cache"]

        self._f_embed = self._vm["embed"]
        self._f_prefill = self._vm["prefill"]
        # self._f_decode = self._vm["decode"]
        self._f_sample = tvm.get_global_func("vm.builtin.sample_top_p_from_logits")

        self._f_kv_cache_clear = tvm.get_global_func("vm.builtin.kv_state_clear")
        self._f_add_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
        self._f_enable_sliding_window_for_seq = tvm.get_global_func("vm.builtin.attention_kv_cache_enable_sliding_window_for_seq")
        self._f_begin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
        self._f_end_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")

        self.paged_kv_cache = self._create_paged_kv_cache()
        self.clear_kv_cache()

    def _load_params_from_path(
        self,
        param_keys: list[str],
        dtype: Literal["bfloat16", "float32"] = "bfloat16",
    ) -> list[tvm.runtime.Tensor]:

        def convert_key(dict_key: str) -> str:
            dict_key = dict_key.replace("model.", "")
            dict_key = dict_key.replace("norm.weight", "norm.scale")

            return dict_key

        def get_ndarray_from_checkpoint(
            checkpoint: Checkpoint,
            key: str,
            dtype: Literal["bfloat16", "float32"] = "bfloat16",
        ):
            original_torch_key= convert_key(key)
            return checkpoint.get(original_torch_key).float().numpy().astype(dtype)

        checkpoint = Checkpoint(str(self.model_path), torch.device("cpu"))

        params = [get_ndarray_from_checkpoint(checkpoint, k, dtype) for k in param_keys]
        return params

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
            # "quantization": args.quantization.name,
            "context_window_size": self.config.context_window_size,
            "sliding_window_size": getattr(self.config, "sliding_window_size", 128),
            "attention_sink_size": 0,
            "prefill_chunk_size": self.config.prefill_chunk_size,  # type: ignore
            "tensor_parallel_shards": self.config.tensor_parallel_shards,  # type: ignore
            "pipeline_parallel_stages": self.config.pipeline_parallel_stages,
            "kv_state_kind": "kv_cache",
            "max_batch_size": 1,
            "rope_theta": 150_000
        }

        # with tvm.transform.PassContext(opt_level=3):
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

    def _create_vm(self, ex: tvm.runtime.Executable):
        return relax.VirtualMachine(ex, self.device)

    def _run_tvm_function(
        self,
        func,
        input: np.ndarray,
        use_kv_cache: bool = False,
    ) -> np.ndarray:
        # convert data into tvm nd array
        input_tensor = tvm.runtime.tensor(input, device=self.device)

        # run function
        if use_kv_cache:
            tvm_output = func(input_tensor, self.paged_kv_cache, self.params)
        else:
            tvm_output = func(input_tensor, self.params)

        # TODO: Check this if-statement is really needed
        if isinstance(tvm_output, (list, tuple)):
            results = []
            for out in tvm_output:
                if hasattr(out, "numpy"):
                    results.append(out.numpy())
                else:
                    results.append(out)
            out = results
        elif hasattr(tvm_output, "numpy"):
            out = tvm_output.numpy()
        else:
            out = tvm_output
        # return result as numpy array
        return out

    def _create_paged_kv_cache(self):
        # TODO: replace hard-coded values
        return self._f_create_kv_cache(  # pylint: disable=too-many-arguments
            tvm.runtime.ShapeTuple([1]),  # max_batch_size
            tvm.runtime.ShapeTuple([8192]),  # max_total_seq_len
            tvm.runtime.ShapeTuple([8192]),  # prefill_chunk_size
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
    ) -> tvm.runtime.Tensor:
        input_embed = self._run_tvm_function(
            self._f_embed,
            input_tokens,
        )
        input_embed = np.expand_dims(input_embed, axis=0)   # [seq_len] -> [1, seq_len]
        input_length = input_embed.shape[1]

        self._f_begin_forward(
            self.paged_kv_cache,
            tvm.runtime.ShapeTuple([sequence_id]), # sequence id
            tvm.runtime.ShapeTuple([input_length])  # added all tokens to kv cache
        )
        logits, self.paged_kv_cache = self._run_tvm_function(
            f_forward,
            input_embed,
            use_kv_cache=True,
        )
        self._f_end_forward(self.paged_kv_cache)

        return logits

    def prefill(
        self,
        input_tokens: np.ndarray,
        sequence_id: int = 0,
        sample: bool = True,
    ) -> int:
        logits = self.forward(input_tokens, self._f_prefill, sequence_id)
        if not sample:
            return logits
        return self.sample(logits)

    # def decode(
    #     self,
    #     input_tokens: np.ndarray,
    #     sequence_id: int = 0,
    #     sample: bool = True,
    # ) -> int:
    #     logits = self.forward(input_tokens, self._f_decode, sequence_id)
    #     if not sample:
    #         return logits
    #     return self.sample(logits)

    def sample(self, logits: np.ndarray):
        new_token_id = self._f_sample(logits, self.temperature, self.top_p, random.random())
        return new_token_id
