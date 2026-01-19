import re
import time
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence

import mlc_llm.compiler_pass  # noqa: F401
import numpy as np
import torch
import tvm
import tvm_ffi
from safetensors import safe_open
from torch.utils.dlpack import to_dlpack

# from tqdm import tqdm
from tvm import relax, runtime
from tvm.relax.frontend import nn
from tvm.target import Target
from tvm_ffi import Device as TVMDevice
from tvm_ffi import DLDeviceType

from model import GPTOssConfig, GPTOssForCausalLM

# Map the names assumed in this implementation to the checkpoint names.
PARAM_NAME_MAP = (
    {f"block.{n}.mlp.mlp1_bias": f"block.{n}.mlp.mlp1_bias" for n in range(36)}
    | {
        f"block.{n}.mlp.mlp1_weight": (f"block.{n}.mlp.mlp1_weight.blocks", f"block.{n}.mlp.mlp1_weight.scales")
        for n in range(36)
    }
    | {f"block.{n}.mlp.mlp2_bias": f"block.{n}.mlp.mlp2_bias" for n in range(36)}
    | {
        f"block.{n}.mlp.mlp2_weight": (f"block.{n}.mlp.mlp2_weight.blocks", f"block.{n}.mlp.mlp2_weight.scales")
        for n in range(36)
    }
)


def to_pytorch_device(device: TVMDevice):
    mapping = {
        DLDeviceType.kDLCPU.value: "cpu",
        DLDeviceType.kDLCUDA.value: "cuda",
        DLDeviceType.kDLCUDAHost.value: "cpu",
        DLDeviceType.kDLVulkan.value: "vulkan",
        DLDeviceType.kDLMetal.value: "mps",
        DLDeviceType.kDLROCM.value: "cuda",  # PyTorch aliases this to cuda
        DLDeviceType.kDLCUDAManaged.value: "cuda",
        DLDeviceType.kDLOneAPI.value: "xpu",
    }
    return mapping.get(device.dlpack_device_type(), "unknown")


def load_original_safetensors(
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

    tvm_params: dict[str, tvm_ffi.Tensor] = {}
    torch_device = to_pytorch_device(device) if device is not None else "cpu"
    with safe_open(st_path, framework="pt", device=torch_device) as f:
        filtered_keys = [k for k in list(f.keys()) if include_key_with_pattern(k)]
        for name in filtered_keys:
            # for name in tqdm(filtered_keys, desc="Loading params"):
            torch_tensor: torch.Tensor = f.get_tensor(name)
            if "blocks" in name or "scales" in name:
                # MXFP4: uint8
                torch_tensor = torch_tensor.to(torch.uint8)
                # Transpose for memory coalescing (move N dimension to the end)
                if torch_tensor.ndim == 4:  # blocks: (E, N, group, b) -> (E, group, b, N)
                    torch_tensor = torch_tensor.permute(0, 2, 3, 1).contiguous()
                elif torch_tensor.ndim == 3:  # scales: (E, N, group) -> (E, group, N)
                    torch_tensor = torch_tensor.permute(0, 2, 1).contiguous()

            tvm_tensor = tvm.runtime.from_dlpack(to_dlpack(torch_tensor))
            tvm_params[name] = tvm_tensor

    return tvm_params


class TVMCheckpoint:
    def __init__(
        self,
        checkpoint_path: str | Path,
        target_device: Literal["cpu", "llvm", "metal", "vulkan", "cuda"] = "cpu",
    ):
        self.device_str = "llvm" if target_device == "cpu" else target_device
        self.device = tvm.device(self.device_str)

        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        self.tvm_params = load_original_safetensors(
            checkpoint_path,
            block_indices=None,
            device=self.device,
        )

    def get_tvm_tensor(self, name: str) -> runtime.Tensor:
        assert name in self.tvm_params, f"Tensor {name} not found in TVM shard."
        return self.tvm_params[name]

    def load_packed_params(self, params: list[tuple[str, nn.Parameter]]) -> list[runtime.Tensor]:
        def convert_key(dict_key: str) -> str:
            replace_tuples = [
                ("model.", ""),
                ("norm.weight", "norm.scale"),
                # for MXFP4 dequantization
                ("weight_blocks", "weight.blocks"),
                ("weight_scales", "weight.scales"),
            ]
            for old, new in replace_tuples:
                dict_key = dict_key.replace(old, new)
            return dict_key

        new_keys = [convert_key(k) for k, _ in params]

        loaded_params = []
        for k in new_keys:
            tvm_tensor = self.get_tvm_tensor(k)
            loaded_params.append(tvm_tensor)

        return loaded_params


class Engine:
    next_seq_id = 0

    def __init__(
        self,
        model_path: str | Path,
        target: str = "metal",
        dump_metal_path: Optional[str | Path] = None,
    ):
        self.model_path = Path(model_path)
        self.config = GPTOssConfig.from_file(self.model_path / "config.json")
        self.model = GPTOssForCausalLM(self.config)
        self.device = tvm.device(target, 0)

        # compile
        ex, params = self._compile_module(target)
        if dump_metal_path:
            self._dump_metal_source(ex, Path(dump_metal_path))

        # load model
        self._vm = relax.VirtualMachine(ex, self.device, profile=False)
        self.params = TVMCheckpoint(checkpoint_path=model_path, target_device=target).load_packed_params(params)

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
        self._warmup()

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
            "rope_theta": 150_000,
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

    def _dump_metal_source(self, ex: relax.VMExecutable, path: Path):
        def find_metal(m, seen=None):
            if seen is None:
                seen = set()
            if m in seen:
                return None
            seen.add(m)

            try:
                if hasattr(m, "inspect_source"):
                    src = m.inspect_source()
                    if src and "#include <metal_stdlib>" in src:
                        return m
            except:
                pass

            try:
                sub_mods = []
                if hasattr(m, "imports"):
                    sub_mods = m.imports
                elif hasattr(m, "imports_"):
                    sub_mods = m.imports_()

                for sub in sub_mods:
                    res = find_metal(sub, seen)
                    if res:
                        return res
            except:
                pass
            return None

        metal_mod = find_metal(ex.mod)
        if metal_mod:
            source = metal_mod.inspect_source()
            with open(path, "w") as f:
                f.write(source)
            print(f"Metal source saved to {path}")
        else:
            print("No Metal module found to dump.")

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
        sliding_window_size: int = None,
        sink_size: int = None,
    ):
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
        input_tokens = tvm.runtime.tensor(input_tokens, device=self.device)
        input_embed: tvm_ffi.Tensor = self._f_embed(input_tokens, self.params)
        input_length = input_embed.shape[1]

        self._f_begin_forward(
            self.paged_kv_cache,
            tvm.runtime.ShapeTuple([sequence_id]),  # sequence id
            tvm.runtime.ShapeTuple([input_length]),  # added all tokens to kv cache
        )
        start_time = time.perf_counter()
        # result = self._vm.profile("prefill", input_embed, self.paged_kv_cache, self.params)
        # print(result)
        logits, self.paged_kv_cache = self._f_prefill(input_embed, self.paged_kv_cache, self.params)
        end_time = time.perf_counter()
        print(f"prefill: {end_time - start_time}s")
        # logits = self.forward(input_tokens, self._f_prefill, sequence_id)
        self._f_end_forward(self.paged_kv_cache)
        return logits

    def decode(
        self,
        input_tokens: np.ndarray,
        sequence_id: int = 0,
    ) -> tvm_ffi.Tensor:
        input_tokens = tvm.runtime.tensor(input_tokens, device=self.device)
        input_embed: tvm_ffi.Tensor = self._f_embed(input_tokens, self.params)

        self._f_begin_forward(
            self.paged_kv_cache,
            tvm.runtime.ShapeTuple([sequence_id]),  # sequence id
            tvm.runtime.ShapeTuple([1]),  # added all tokens to kv cache
        )
        start_time = time.perf_counter()
        # result = self._vm.profile("decode", input_embed, self.paged_kv_cache, self.params)
        # print(result)
        logits, self.paged_kv_cache = self._f_decode(input_embed, self.paged_kv_cache, self.params)
        end_time = time.perf_counter()
        print(f"decode: {end_time - start_time}s")
        self._f_end_forward(self.paged_kv_cache)
        # logits = self.forward(input_tokens, self._f_decode, sequence_id)
        return logits

    def sample(self, logits: tvm_ffi.Tensor) -> int:
        start_time = time.perf_counter()
        new_token_id: int = self._f_sample(logits, *self.sample_parameters.values())
        # print(f"sample: {time.perf_counter() - start_time}s")
        return new_token_id

    def _warmup(self):
        dummy_input = np.array([0], dtype="int32")
        seq_id = self.begin_sequence()
        self.prefill(dummy_input, seq_id)
        self.clear_kv_cache()
        self.__class__.next_seq_id = 0
