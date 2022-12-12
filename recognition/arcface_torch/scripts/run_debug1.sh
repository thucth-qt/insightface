
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 mprof run --multiprocess  train.py $@ &> mem_track_dataset.log

# ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh


# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 mprof run --multiprocess  train.py $@ &> mem_track_dataset.log


CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=4,5 mprof run --multiprocess python -m torch.distributed.launch \
--nproc_per_node=2 \
--nnodes=1 \
--node_rank=0 \
--master_addr="127.0.0.1" \
--master_port=12345 train.py configs/wf42m_debug_ada &> mem_track_dataset_1733.log

ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh

