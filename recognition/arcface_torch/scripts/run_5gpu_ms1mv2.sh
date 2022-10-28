
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch \
--nproc_per_node=4 \
--nnodes=1 \
--node_rank=0 \
--master_addr="127.0.0.1" \
--master_port=12346 train.py configs/ms1mv2_ir50_thucth_adaface1.py

ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
