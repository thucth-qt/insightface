
CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch \
--nproc_per_node=3 \
--nnodes=1 \
--node_rank=0 \
--master_addr="127.0.0.1" \
--master_port=12347 train.py $@ &>>log.log