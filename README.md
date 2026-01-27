# gpt-oss-tvm

<a href="https://github.com/brekkylab/gpt-oss-tvm/blob/main/LICENSE"><img src="https://img.shields.io/github/license/brekkylab/gpt-oss-tvm" alt="License"></a>

<a href="https://tvm.apache.org"><img src="https://img.shields.io/badge/TVM-0b4973.svg?style=for-the-badge&logo=Apache&logoColor=white" alt="TVM"></a>
<a href="https://github.com/openai/gpt-oss"><img src="https://img.shields.io/badge/gpt--oss-0fa47f?style=for-the-badge&logo=openai&logoColor=white" alt="gpt-oss"></a>
<a href="https://github.com/brekkylab/gpt-oss-tvm/wiki"><img src="https://img.shields.io/badge/Wiki-181717.svg?style=for-the-badge&logo=GitHub&logoColor=white" alt="Wiki"></a>

This project aims to compile [OpenAI gpt-oss](https://github.com/openai/gpt-oss) model using [Apache TVM](https://github.com/apache/tvm) and run it on the target device.


## Project Goals
<a href="https://github.com/brekkylab/gpt-oss-tvm/wiki"><img src="https://img.shields.io/badge/Wiki-181717.svg?style=for-the-badge&logo=GitHub&logoColor=white" alt="Wiki"></a>

Visit [Wiki Home](https://github.com/brekkylab/gpt-oss-tvm/wiki) or [Design Philosophy](https://github.com/brekkylab/gpt-oss-tvm/wiki/Design-Philosophy) page to read more for the project goal and objectives!

## Setup
To support gpt-oss _correctly_, TVM & MLC LLM needs to be built with a few patches.

Please refer to our [Wiki - Setup & Run](https://github.com/brekkylab/gpt-oss-tvm/wiki/Setup-%26-Run) page for setup instructions.

## Download model

> [!NOTE]
> While TVM supports multiple hardware backends, this project has been mainly tested with the metal target on macOS.
> As the model uses the original mxfp4 and bfloat16 weights without further quantization, an Apple Silicon Mac with **24 GB or more of unified memory** is recommended.

### Files for gpt-oss reference torch implementation
```bash
pip install huggingface_hub  # to use `hf` command
hf download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/
```

## Compile & Run
> [!IMPORTANT]
> To ensure equivalence with _**gpt-oss**_, please confirm that a TVM built with the [patches](https://github.com/brekkylab/gpt-oss-tvm/blob/main/tvm_fix.patch) applied.  
> You can install the desired **TVM & MLC LLM** by referring to the [Wiki Setup page](https://github.com/brekkylab/gpt-oss-tvm/wiki/Setup-%26-Run#tvm--mlc-llm).

### Basic single-turn test
```bash
python run_gpt_oss.py
```

### Multi-turn chat
```bash
python chat.py
```

### Use other target devices
The target device can be changed by modifying the following line in the scripts:
```diff
- engine = Engine(model_path, target="metal")
+ engine = Engine(model_path, target="<YOUR DEVICE TYPE>")
```

Supported device types are determined by [TVM target](https://tvm.apache.org/docs/reference/api/python/target.html) support.


## License
This project follows the [Apache License 2.0](https://github.com/brekkylab/gpt-oss-tvm/blob/main/LICENSE), in line with the licenses of [gpt-oss](https://github.com/openai/gpt-oss?tab=Apache-2.0-1-ov-file#readme) and [TVM](https://github.com/apache/tvm?tab=Apache-2.0-1-ov-file#readme).

## Authors
- @Liberatedwinner
- @grf53
- @jhlee525
- @khj809
