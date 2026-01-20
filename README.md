# gpt-oss-tvm
Compile &amp; Run OpenAI's gpt-oss through TVM

## Install dependencies
### TVM & MLC-LLM
#### Update submodule
```bash
git submodule update --init --recursive
```

#### Apply patches
```bash
cd 3rdparty/mlc-llm/3rdparty/tvm
git apply ../../../../tvm_fix.patch
cd -
```

#### Build from source
Follow the official documentations below to build TVM & MLC-LLM.
Cloning repository is already done through the git submodule command above.
- [TVM](https://tvm.apache.org/docs/install/from_source.html) (in `3rdparty/mlc-llm/3rdparty/tvm`): 
    You will perform the tasks similar to this:
    ```bash
    mkdir build && cp cmake/config.cmake build/ && cd build
    # Now, edit build/config.cmake refer to the document
    cmake .. && cmake --build . --parallel $(nproc)
    ```
- [MLC-LLM](https://llm.mlc.ai/docs/install/tvm.html#option-2-build-from-source) (in `3rdparty/mlc-llm`): 
    You will perform the tasks similar to this:
    ```bash
    mkdir build && cd build
    python ../cmake/gen_cmake_config.py      # Answer this script to generate configuration
    export CMAKE_POLICY_VERSION_MINIMUM=3.5  # Recommended to avoid the CMake error on `tokenizer-cpp`
    cmake .. && cmake --build . --parallel $(nproc)
    ```

#### Install Python bindings
```bash
# << apache-tvm-ffi >>
# Install from source (recommended)
cd 3rdparty/mlc-llm/3rdparty/tvm/3rdparty/tvm-ffi
pip install -e .
cd -
# ... Or just use PyPI version
pip install apache-tvm-ffi<=0.1.7

# << tvm >>
cd 3rdparty/mlc-llm/3rdparty/tvm/python
pip install -e .
cd -
# Extra dependencies for tvm
# c.f. https://tvm.apache.org/docs/install/from_source.html#step-5-extra-python-dependencies
pip install psutil

# << mlc-llm >>
cd 3rdparty/mlc-llm/python
# Notes on handling `flashinfer-python` in `requirements.txt`
#  - exclude installation on unsupported platforms (e.g. macOS)
#  - use version constraint `>=0.5.0` for better dependency resolution
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
pip install huggingface_hub  # to use `hf` command
hf download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/
```

## Authors
- @Liberatedwinner
- @grf53
- @jhlee525
- @khj809
