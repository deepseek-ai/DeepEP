#!/bin/bash

# Run with: sbatch --requeue --parsable launch_modified_high_throughput_4_nodes.sh

#SBATCH --account=network_research_swarch
#SBATCH -J network_research_swarch-deepep.modified_high_throughput_test
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --switches=1
#SBATCH --open-mode=append
#SBATCH --partition=batch
#SBATCH --time=00:15:00

set -evx

# Storage directory - change this to match your system
export STORAGE_DIR="/home/gtheodorakis/storage/shared-data"

# Check if required files and directories exist
export MOUNT_DIR="${STORAGE_DIR}"
export MOUNT_DEST="/mnt/shared-data"
export CONTAINER_IMAGE="${STORAGE_DIR}/new_image.sqsh"

if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
    echo "ERROR: Container image not found: ${CONTAINER_IMAGE}"
    exit 1
fi

if [[ ! -d "${MOUNT_DIR}" ]]; then
    echo "ERROR: Mount directory not found: ${MOUNT_DIR}"
    exit 1
fi

export WORK_DIR=/root
export NVSHMEM_DIR=${WORK_DIR}/nvshmem/install
export DEEP_EP_DIR=${MOUNT_DEST}/gin-deepep

# Use dynamic port assignment to avoid conflicts
export MASTER_PORT=$(($SLURM_JOB_ID % 50000 + 10000))

# Backend configuration
export ENABLE_NCCL_GIN=1  # Set to 1 to use NCCL GIN backend, 0 for NVSHMEM

# NVSHMEM InfiniBand Service Level setting
# NVSHMEM_IB_SL=0: Default, no adaptive routing
# NVSHMEM_IB_SL=1: Enable adaptive routing for better performance
export NVSHMEM_IB_SL=0

# Number of test iterations to run
export NUM_ITERATIONS=3

# Generate log file names with timestamp
export LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export LOG_DIR="${STORAGE_DIR}/deepep-logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"
if [[ ! -d "${LOG_DIR}" ]]; then
    echo "ERROR: Cannot create log directory: ${LOG_DIR}"
    exit 1
fi

echo "Running ${NUM_ITERATIONS} iteration(s)..."

# Run multiple iterations if specified
for ((i=1; i<=${NUM_ITERATIONS}; i++)); do
    echo "=== Starting iteration ${i}/${NUM_ITERATIONS} ==="
    
    # Generate iteration-specific log file names
    export ITER_LOG_OUTPUT="${LOG_DIR}/deepep.modified_internode_test_n${SLURM_NNODES}_t${SLURM_NTASKS_PER_NODE}_salloc_${SLURM_JOB_ID}_${LOG_TIMESTAMP}_iter${i}.out"
    export ITER_LOG_ERROR="${LOG_DIR}/deepep.modified_internode_test_n${SLURM_NNODES}_t${SLURM_NTASKS_PER_NODE}_salloc_${SLURM_JOB_ID}_${LOG_TIMESTAMP}_iter${i}.err"
    
    echo "  Iteration ${i} logs:"
    echo "    Output: ${ITER_LOG_OUTPUT}"
    echo "    Error:  ${ITER_LOG_ERROR}"
    
    # Set MASTER_ADDR and MASTER_PORT for distributed training
    srun                                                                          \
    --output="${ITER_LOG_OUTPUT}"                                                 \
    --error="${ITER_LOG_ERROR}"                                                   \
    --container-image="${CONTAINER_IMAGE}"                                        \
    --mpi=pmix                                                                    \
    --container-mounts=${MOUNT_DIR}:${MOUNT_DEST}                                 \
    --export=NVSHMEM_IB_SL=${NVSHMEM_IB_SL},MASTER_ADDR=$(hostname),MASTER_PORT=${MASTER_PORT},ENABLE_NCCL_GIN=${ENABLE_NCCL_GIN}   \
    bash -c "
    cd ${WORK_DIR};

    echo '=== Environment Debug Info ===';
    echo \"Current working directory: \$(pwd)\";
    echo \"Current user: \$(whoami)\";
    echo \"Current PATH: \$PATH\";
    echo \"Current LD_LIBRARY_PATH: \$LD_LIBRARY_PATH\";
    echo \"MASTER_ADDR: \$MASTER_ADDR\";
    echo \"MASTER_PORT: \$MASTER_PORT\";
    echo \"SLURM_PROCID: \$SLURM_PROCID\";
    echo \"SLURM_NTASKS: \$SLURM_NTASKS\";
    echo \"SLURM_NNODES: \$SLURM_NNODES\";
    echo \"SLURM_NTASKS_PER_NODE: \$SLURM_NTASKS_PER_NODE\";
    echo 'Before source:';
    which nvshmem-info || echo 'nvshmem-info not found';
    
    # Setup HPCX environment with better debugging
    echo '=== HPCX Setup ===';
    export HPCX_DIR=/root/hpcx/hpcx-v2.21.2-gcc-inbox-ubuntu24.04-cuda12-x86_64
    echo \"HPCX_DIR set to: \$HPCX_DIR\";
    source \$HPCX_DIR/hpcx-mt-init.sh && hpcx_load
    
    # Check if directory exists
    if [[ -d \"\$HPCX_DIR\" ]]; then
        echo \"HPCX_DIR directory exists\";
    else
        echo \"WARNING: HPCX_DIR directory does not exist: \$HPCX_DIR\";
    fi
    
    echo \"HPCX_DIR: \$HPCX_DIR\";
    echo \"PATH: \$PATH\";
    echo \"LD_LIBRARY_PATH: \$LD_LIBRARY_PATH\";
    
    # Setup CUDA environment
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH=\$CUDA_HOME/bin:\$PATH
    export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
    
    # Setup NVSHMEM environment
    export NVSHMEM_DIR=/root/nvshmem
    source /root/setup_nvshmem_env.sh
    
    # Setup NCCL GIN environment
    export NCCL_NET_HOME=none
    export NCCL_GIN_HOME=/mnt/shared-data/nccl_gin_install/updated_nccl/nccl
    export LD_LIBRARY_PATH=\$NCCL_GIN_HOME/build/lib:\$LD_LIBRARY_PATH
    
    echo 'After source:';
    which nvshmem-info;
    cd ${DEEP_EP_DIR};
    
    # Force Python to use the pre-built deep_ep installation
    # The package is installed in dist-packages, not site-packages
    export PYTHONPATH=${DEEP_EP_DIR}/deepep_custom_install/local/lib/python3.12/dist-packages/deep_ep-1.2.1+b14df36-py3.12-linux-x86_64.egg:$PYTHONPATH
    
    echo '=== Using pre-built deep_ep installation ===';
    echo \"PYTHONPATH: \$PYTHONPATH\";
    echo \"deep_ep install location: ${DEEP_EP_DIR}/deepep_custom_install\";

    # Verify deep_ep is available
    python3 -c \"
import sys
print('Python executable:', sys.executable)
print('Python version:', sys.version)
print('Python path:')
for p in sys.path:
    print('  ', p)

try:
    from mpi4py import MPI
    print('✓ mpi4py imported successfully')
    import deep_ep
    print('✓ deep_ep imported successfully')
    print('deep_ep location:', deep_ep.__file__)
except ImportError as e:
    print('✗ Failed to import:', e)
    sys.exit(1)
\";
    
    # Choose backend based on ENABLE_NCCL_GIN flag
    if [[ \"\${ENABLE_NCCL_GIN}\" == \"1\" ]]; then
        echo '=== Running with NCCL GIN backend ===';

        DEEP_EP_GIN_NUM_COMMS=12 LD_LIBRARY_PATH=\$MPI_HOME/lib/:\$HPCX_UCX_DIR/lib/:\$NCCL_GIN_HOME/build/lib:\$NCCL_GIN_HOME/src/gin/transport/gdaki/doca-gpunetio-lite/lib:\$LD_LIBRARY_PATH DEEP_EP_BACKEND=nccl_gin NCCL_GIN_TYPE=3 NCCL_GIN_ENABLE=1 UCX_IB_DM_COUNT=0 NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_NET_PLUGIN=none NCCL_DEBUG=0 DOCA_GPUNETIO_LITE_DEBUG=0 python tests/test_internode.py;

    else
        echo '=== Running with NVSHMEM backend ===';
        export DEEP_EP_BACKEND=nvshmem;
        python tests/test_internode.py;
    fi"
    
    # Add delay between iterations (except for the last one)
    if [[ ${i} -lt ${NUM_ITERATIONS} ]]; then
        echo "=== Iteration ${i} completed. Waiting 10 seconds before next iteration... ==="
        sleep 10
    fi
done

echo "=== All ${NUM_ITERATIONS} iterations completed ===" 
