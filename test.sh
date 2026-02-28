unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
export LD_LIBRARY_PATH=/root/paddlejob/share-storage/gpfs/system-public/lzy/eb5_tool/miniconda3/envs/lzy/lib/python3.10/site-packages/torch/lib/:$LD_LIBRARY_PATH

rm -rf log
rm -rf core.*
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH="/root/paddlejob/share-storage/gpfs/system-public/lzy/bzz2/hybrid_ep/bingoo_deepep/DeepEP":$PYTHONPATH

python -m paddle.distributed.launch \
            tests/test_hybrid_ep.py