rm -rf build
rm -rf dist
rm -rf deep_ep_cpp.cpython-38-x86_64-linux-gnu.so
export TORCH_CUDA_ARCH_LIST="10.0"
export PADDLE_CUDA_ARCH_LIST=10.0
export CUDA_HOME="/usr/local/cuda"
NVSHMEM_DIR=/root/paddlejob/share-storage/gpfs/system-public/lzy/bzz2/nvshmem python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall