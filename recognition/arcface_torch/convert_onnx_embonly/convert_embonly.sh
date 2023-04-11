
CUDA_VISIBLE_DEVICES=3 python torch2onnx.py \
--input "/home/thucth/Biometrics/insightface/recognition/arcface_torch/work_dirs/finetune_363kIden_1306kImage/model_10.pt" \
--output "/home/thucth/Biometrics/insightface/recognition/arcface_torch/work_dirs/finetune_363kIden_1306kImage/model_10.onnx" \
--network "ir_101"



ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
