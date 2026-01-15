from pathlib import Path
import random
import re
from typing import Callable, Optional, Literal, Sequence

from gpt_oss.torch.weights import Checkpoint
from ml_dtypes import bfloat16  # noqa: F401
import numpy as np
from numpy.typing import NDArray
import torch
from safetensors import safe_open

import tvm
from tvm import relax
from tvm import runtime
from tvm.relax.frontend import nn
from tvm.contrib.tvmjs import load_tensor_cache
from tvm.target import Target

from model import GPTOssConfig, GPTOssForCausalLM


# from configs import MODEL_PATH, SHARD_PATH

# Bytes per MXFP4 block: 32 FP4 numbers packed in 16 bytes
BYTES_PER_BLOCK = 16

# Map the names assumed in this implementation to the checkpoint names.
PARAM_NAME_MAP = {
    f"block.{n}.mlp.mlp1_bias": f"block.{n}.mlp.mlp1_bias" for n in range(36)
} | {
    f"block.{n}.mlp.mlp1_weight": (
        f"block.{n}.mlp.mlp1_weight.blocks", f"block.{n}.mlp.mlp1_weight.scales"
    ) for n in range(36)
} | {
    f"block.{n}.mlp.mlp2_bias": f"block.{n}.mlp.mlp2_bias" for n in range(36)
} | {
    f"block.{n}.mlp.mlp2_weight": (
        f"block.{n}.mlp.mlp2_weight.blocks", f"block.{n}.mlp.mlp2_weight.scales"
    ) for n in range(36)
}


def process_pre_tensor(name: str, raw_np: NDArray) -> NDArray:
    if "blocks" in name or "scales" in name:
        # MXFP4: uint8
        target_dtype = "uint8" if raw_np.dtype == np.uint8 else str(raw_np.dtype)
        processed_data = raw_np.astype(target_dtype)
    else:
        processed_data = raw_np.astype(bfloat16)

    print(f"Processed: {name} | Dtype: {processed_data.dtype} | Shape: {processed_data.shape}")

    return processed_data


def build_shards_from_safetensors(
        st_path: Path,
        block_indices: Optional[Sequence[int]] = None,
        include_unnumbered_blocks: bool = True,
        key_pattern: str = r"block\.(\d+)\.",
        device: Optional[runtime.Device] = None,
):

    st_path = Path(st_path) / "model.safetensors" if st_path.suffix != ".safetensors" else st_path
    if not st_path.exists():
        raise FileNotFoundError(
            f"safetensor not found: {st_path}\n"
            f"please run: `hf download openai/gpt-oss-20b --local-dir {st_path.parent}`"
        )

    block_indices = set(block_indices) if block_indices else None

    def include_key_with_pattern(key_name: str) -> bool:
        if block_indices is None:
            result = True

        else:
            matched = re.search(key_pattern, key_name)
            if matched:
                result = int(matched.group(1)) in block_indices
            else:
                result = include_unnumbered_blocks

        return result

    tvm_params = {}
    with safe_open(st_path, framework="numpy") as f:
        filtered_keys = [k for k in list(f.keys()) if include_key_with_pattern(k)]
        for name in filtered_keys:
            raw_np = f.get_tensor(name)
            tvm_params[name] = runtime.tensor(process_pre_tensor(name, raw_np), device=device)

    return tvm_params


class TVMCheckpoint:
    # MODEL_PATH = Path.cwd() / "ckpt-20b" / "original"
    # SHARD_PATH = Path.cwd() / "gpt_oss_tvm" / "params"

    def __init__(
            self,
            checkpoint_path: str | Path,
            target_device: Literal["cpu", "llvm", "metal", "vulkan", "cuda"] = "cpu",
            from_shards: bool = False,
    ):
        self.device_str = "llvm" if target_device == "cpu" else target_device
        self.device = tvm.device(self.device_str)

        if from_shards:
            checkpoint_path = str(checkpoint_path)  # == `str(SHARD_PATH)`
            self.tvm_params, _ = load_tensor_cache(checkpoint_path, device=self.device)
            print(f"Loaded {len(self.tvm_params)} parameters from shards: {checkpoint_path}")
        else:
            # checkpoint_path == `MODEL_PATH`
            self.tvm_params = build_shards_from_safetensors(
                checkpoint_path,
                block_indices=None,
                device=self.device,
            )
            print(f"Loaded {len(self.tvm_params)} parameters from safetensors: {checkpoint_path}")

    def get_tvm_tensor(self, name: str) -> runtime.Tensor:
        assert name in self.tvm_params, f"Tensor {name} not found in TVM shard."
        return self.tvm_params[name]

    def load_packed_params(
            self,
            params: list[tuple[str, nn.Parameter]],
            set_to_dtype: Optional[str] = None
    ) -> list[runtime.Tensor]:

        def convert_key(dict_key: str) -> str:
            replace_tuples = [
                ("model.", ""),
                ("norm.weight", "norm.scale"),
                # for MXFP4 dequantization
                ("weight_blocks", "weight.blocks"),
                ("weight_scales", "weight.scales")
            ]
            for old, new in replace_tuples:
                dict_key = dict_key.replace(old, new)
            return dict_key

        for i, (k, v) in enumerate(params):
            print(f"{i}: {k} | shape: {v.shape}")

        new_keys = [convert_key(k) for k, _ in params]
        print(f"Converted keys: {new_keys}")

        loaded_params = []
        for k in new_keys:
            tvm_tensor = self.get_tvm_tensor(k)
            cast_skip = any(weight_name in k for weight_name in {"weight.blocks", "weight.scales"})
            if set_to_dtype is not None and not cast_skip:
                expr_ = tvm_tensor.numpy().astype(set_to_dtype)
                tvm_tensor = runtime.tensor(expr_, device=self.device)
            loaded_params.append(tvm_tensor)

        return loaded_params



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
        self.params = TVMCheckpoint(
                checkpoint_path=model_path, target_device=target, from_shards=False
            ).load_packed_params(params, set_to_dtype=param_dtype)
        self._vm = self._create_vm(ex)

        # get functions
        self._f_create_kv_cache = self._vm["create_tir_paged_kv_cache"]

        self._f_embed = self._vm["embed"]
        self._f_prefill = self._vm["prefill"]
        self._f_decode = self._vm["decode"]

        self._f_sample = tvm.get_global_func("vm.builtin.sample_top_p_from_logits")
        self.sample_parameters = {"temperature": 0.7, "top_p": 0.9, "random_seed": 0.5}

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
        func: Callable,
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

        out = tvm_output.numpy() if hasattr(tvm_output, "numpy") else tvm_output

        return out


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

    def decode(
        self,
        input_tokens: np.ndarray,
        sequence_id: int = 0,
        sample: bool = True,
    ) -> int:
        logits = self.forward(input_tokens, self._f_decode, sequence_id)
        if not sample:
            return logits
        return self.sample(logits)

    def sample(self, logits: np.ndarray):
        new_token_id = self._f_sample(logits, *self.sample_parameters.values())
        return new_token_id
