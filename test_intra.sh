unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS

rm -rf log
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH="/root/paddlejob/share-storage/gpfs/system-public/lzy/bzz2/hybrid_ep/DeepEP":$PYTHONPATH

python -m paddle.distributed.launch \
            tests/test_intranode.py