# gpt-oss-tvm
Compile &amp; Run OpenAI's gpt-oss through TVM

## Install dependencies
### TVM & MLC-LLM
#### Update submodule
```
git submodule update --init --recursive
```

#### Apply patches
```bash
cd 3rdparty/mlc-llm/3rdparty/tvm
git apply ../../../../tvm_fix.patch
cd -
```

#### Build from source
- [TVM](https://llm.mlc.ai/docs/install/mlc_llm.html#option-2-build-from-source)
- [MLC-LLM](https://llm.mlc.ai/docs/install/tvm.html#option-2-build-from-source)

#### Install Python bindings
```bash
# apache-tvm-ffi 
cd 3rdparty/mlc-llm/3rdparty/tvm/3rdparty/tvm-ffi
pip install -e .
cd -

# tvm
cd 3rdparty/mlc-llm/3rdparty/tvm/python
pip install -e .
cd -

# mlc-llm
cd 3rdparty/mlc-llm/python
# if you use macOS, you need to exclude `flashinfer` dependency from `./requirements.txt`
pip install -e .
cd -
```

### Python dependencies
```bash
pip install -r requirements.txt
```

## Download model
### Files for gpt-oss reference torch implementation
```bash
pip install -y huggingface_hub  # to use `hf` command
hf download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/
```

## Authors
- @Liberatedwinner
- @grf53
- @jhlee525
- @khj809
