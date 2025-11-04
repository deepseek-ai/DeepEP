#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -J coreai_devtech_all-FW:hybrid_ep
#SBATCH -p batch
#SBATCH -t 00:10:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
set -x

export RDMA_CORE_HOME=/lustre/fsw/coreai_devtech_all/tongliu/rdma-core/build
HYBRID_EP_PATH=/lustre/fsw/coreai_devtech_all/tongliu/DeepEP
OUTPUT_PATH=/lustre/fsw/coreai_devtech_all/tongliu/DeepEP/output
mkdir -p $OUTPUT_PATH/

NTASKS_PER_NODE=8
export NVLINK_DOMAIN_SIZE=$NTASKS_PER_NODE
export WORLD_SIZE=$SLURM_NNODES
export HIDDEN_DIM=7168
export MAX_NUM_OF_TOKENS_PER_RANK=4096
export NUM_TOKENS_PER_RANK=4096
export NUM_LOCAL_EXPERTS=8
export NUM_OF_RANKS_PER_NODE=$NTASKS_PER_NODE
export TOPK=6
export PAD_MULTIPLE=32
export ITERATIONS=100
export SEED=42
export NUM_OF_NODES=$SLURM_NNODES
export PYTHONPATH=$HYBRID_EP_PATH:$PYTHONPATH
CONT=/lustre/fsw/coreai_devtech_all/tongliu/Sqsh/pytorch.sqsh

if [ "${PROFILE}" = 1 ]; then
    PROFILE_CMD="nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-memory-usage true -f true -x true -o ${OUTPUT_PATH}/report_hybrid_ep"
else
    PROFILE_CMD=""
fi

run_cmd="${PROFILE_CMD} python ${HYBRID_EP_PATH}/tests/test_hybrid_ep.py --num-processes $NTASKS_PER_NODE --ib-dev-name-list mlx5_0 mlx5_3 mlx5_4 mlx5_5 mlx5_6 mlx5_9 mlx5_10 mlx5_11"

set -x

srun --mpi=pmix --container-image=${CONT} \
    --container-mounts=/lustre/:/lustre/ \
    --output="$OUTPUT_PATH/log/dispatcher_%j.log" \
    --no-container-mount-home ${run_cmd}