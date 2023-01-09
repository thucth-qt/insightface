
CUDA_VISIBLE_DEVICES=3 python torch2onnx.py --input "/home/thucth/thucth/DAL/face-recognition/insightface/recognition/arcface_torch/official_releases/wf42m10faces_pfc10_ir101_adaface_cont/model_3.pt" --output "/home/thucth/thucth/DAL/face-recognition/insightface/recognition/arcface_torch/official_releases/wf42m10faces_pfc10_ir101_adaface_cont/adaface_model3.onnx" --network "ir_101"

ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
