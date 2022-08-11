
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5 python -m torch.distributed.launch \
--nproc_per_node=5 \
--nnodes=1 \
--node_rank=0 \
--master_addr="127.0.0.1" \
--master_port=12346 train.py configs/wf42m_pfc03_ir101_thucth_adaface &> log.log

ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
