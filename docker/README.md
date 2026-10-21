# DeepEP Docker image

Minimal image based on the official PyTorch CUDA **devel** image, with DeepEP and its NCCL floor installed.

## Requirements

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on a Linux host
- NVIDIA Hopper (or newer) GPU for running DeepEP kernels
- Repository checkout with the DeepJIT submodule initialized

The image installs the host extension at build time. DeepJIT still needs `nvcc` from the CUDA toolkit at runtime, which is why the base image is `*-devel` rather than `*-runtime`.

## Build

From the repository root:

```bash
git submodule update --init --recursive
# On Apple Silicon / other non-amd64 hosts, pin the platform to match the NVIDIA image:
docker build --platform=linux/amd64 -f docker/Dockerfile -t deepep:local .
```

Optional base-image override (must remain a CUDA 13.1+ devel image with PyTorch ≥ 2.10):

```bash
docker build --platform=linux/amd64 -f docker/Dockerfile \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.14.1-cuda13.2-cudnn9-devel \
  -t deepep:local .
```

## Run

```bash
docker run --rm -it --gpus all --ipc=host deepep:local
```

`--ipc=host` (or a larger `--shm-size`) is recommended for PyTorch multiprocessing and multi-process tests.

Smoke check inside the container:

```bash
python -c "import deep_ep, torch; print(deep_ep.__version__, torch.cuda.is_available())"
```

Example single-node test (needs a visible GPU):

```bash
python /opt/DeepEP/tests/ep/test_ep.py
```

## What this image includes

| Component | Source |
| --- | --- |
| PyTorch + CUDA toolkit (`nvcc`) | `pytorch/pytorch:*-cuda13.2-cudnn9-devel` |
| NCCL ≥ 2.32.3 | `pip install nvidia-nccl-cu13>=2.32.3 --no-deps` |
| DeepEP | built with `bash install.sh` into the image Python env |
| NumPy | test dependency |

## Limits

- Inter-node RDMA / multi-node tests need matching host IB/RoCE setup and usually `--network=host` plus device mounts for RDMA; those are site-specific and are not baked into this Dockerfile.
- Building the image on macOS/arm64 without an NVIDIA GPU can produce an amd64 image via buildx, but you still need a Linux NVIDIA host to run DeepEP.
- The Docker build context excludes `.git`, so the installed package version suffix is typically `+local` (see `setup.py`).
