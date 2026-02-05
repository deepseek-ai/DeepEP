unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS

rm -rf log_rank*
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH="/root/paddlejob/share-storage/gpfs/system-public/lzy/bzz2/hybrid_ep/DeepEP":$PYTHONPATH
# export CUDA_VISIBLE_DEVICES="4,5,6,7"
python tests/test_hybrid_ep.py --num-processes 8

# torchrun --nproc_per_node=8 tests/test_hybrid_ep.py