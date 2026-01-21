import gc
from pathlib import Path
from typing import Literal

import torch
import tvm
import tvm_ffi
from safetensors import safe_open
from torch.utils.dlpack import to_dlpack
from tqdm import tqdm
from tvm.relax.frontend import nn
from tvm_ffi import Device as TVMDevice
from tvm_ffi import DLDeviceType

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]

# Bytes per MXFP4 block: 32 FP4 numbers packed in 16 bytes
BYTES_PER_BLOCK = 16

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


def tvm_device_to_pytorch_device(device: TVMDevice) -> Literal["cpu", "cuda", "vulkan", "mps", "xpu", "unknown"]:
    mapping = {
        DLDeviceType.kDLCPU.value: "cpu",
        DLDeviceType.kDLCUDA.value: "cuda",
        DLDeviceType.kDLCUDAHost.value: "cpu",
        DLDeviceType.kDLVulkan.value: "vulkan",
        DLDeviceType.kDLMetal.value: "mps",
        DLDeviceType.kDLROCM.value: "cuda",  # PyTorch aliases this to cuda
        DLDeviceType.kDLROCMHost.value: "cpu",
        DLDeviceType.kDLCUDAManaged.value: "cuda",
        DLDeviceType.kDLOneAPI.value: "xpu",
    }
    return mapping.get(device.dlpack_device_type(), "unknown")


class TVMCheckpoint:
    def __init__(
        self,
        path: str | Path,
        target_device: Literal["cpu", "llvm", "metal", "vulkan", "cuda"] = "cpu",
    ):
        self.path = Path(path)
        self.device_str = "llvm" if target_device == "cpu" else target_device
        self.device = tvm.device(self.device_str)

        self.tvm_params = self._load_safetensors()

    def _load_safetensors(self) -> dict[str, tvm_ffi.Tensor]:
        st_path = self.path / "model.safetensors" if self.path.suffix != ".safetensors" else self.path
        if not st_path.exists():
            raise FileNotFoundError(
                f"safetensor not found: {st_path}\nplease run: `hf download openai/gpt-oss-20b --local-dir {st_path.parent}`"
            )

        tvm_params: dict[str, tvm_ffi.Tensor] = {}
        pytorch_device = tvm_device_to_pytorch_device(self.device)
        with safe_open(st_path, framework="pt", device=pytorch_device) as f:
            for name in tqdm(f.keys(), desc="Loading params"):
                torch_tensor: torch.Tensor = f.get_tensor(name)
                if "blocks" in name or "scales" in name:
                    torch_tensor = torch_tensor.to(torch.uint8)
                tvm_tensor = tvm.runtime.from_dlpack(to_dlpack(torch_tensor))
                tvm_params[name] = tvm_tensor

                # prevent memory pressure
                gc.collect()

        return tvm_params

    def get_tvm_tensor(self, name: str) -> tvm_ffi.Tensor:
        assert name in self.tvm_params, f"Tensor {name} not found in TVM shard."
        return self.tvm_params[name]

    def load_packed_params(self, params: list[tuple[str, nn.Parameter]]) -> list[tvm_ffi.Tensor]:
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
