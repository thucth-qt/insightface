
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nnodes=1 --nproc-per-node=6 --node_rank=0 --master_addr="127.0.0.1" --master_port=12346 train.py configs/combine_pfc10_ir201_adaface_5gnn &> train1.log

ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
