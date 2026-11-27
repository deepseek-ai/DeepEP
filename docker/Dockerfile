# Official DeepEP image: PyTorch + CUDA devel toolkit + NCCL + built deep_ep.
#
# Build from the repository root (DeepJIT submodule must already be checked out):
#   git submodule update --init --recursive
#   docker build -f docker/Dockerfile -t deepep:local .
#
# Run on a Linux host with NVIDIA Container Toolkit:
#   docker run --rm -it --gpus all --ipc=host deepep:local

# CUDA Toolkit 13.1+ is required by DeepEP's DeepJIT path; use a matching devel image
# so nvcc remains available at runtime for JIT compilation.
ARG BASE_IMAGE=pytorch/pytorch:2.14.1-cuda13.2-cudnn9-devel
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="DeepEP" \
      org.opencontainers.image.description="DeepEP built on the official PyTorch CUDA devel image" \
      org.opencontainers.image.source="https://github.com/deepseek-ai/DeepEP"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # Keep the toolkit discoverable for DeepJIT at import/runtime.
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH}

# Minimal host packages beyond the PyTorch devel image.
# build-essential provides a C++20-capable toolchain on current Ubuntu bases.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/DeepEP

# Expect a clean build context that already contains third-party/deep_jit sources.
COPY . .

RUN test -f third-party/deep_jit/include/deep_jit/python_api.hpp \
    || (echo "DeepJIT submodule missing. Run: git submodule update --init --recursive" >&2 && exit 1)

# Match the README floor. --no-deps avoids pulling an older torch pin transitively.
# find_pkgs / EP_NCCL_ROOT_DIR will resolve this pip package at build and import time.
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir "nvidia-nccl-cu13>=2.32.3" --no-deps \
    && python -m pip install --no-cache-dir numpy \
    && bash install.sh \
    && python -c "import deep_ep; print('deep_ep', deep_ep.__version__)"

WORKDIR /workspace
CMD ["bash"]
