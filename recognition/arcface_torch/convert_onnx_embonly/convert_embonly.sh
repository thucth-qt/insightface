
CUDA_VISIBLE_DEVICES=3 python torch2onnx.py --input "/home/thucth/thucth/project/NIST_source/nist_frvt_11/11/config/models/model_9.pt" --output "/home/thucth/thucth/project/NIST_source/nist_frvt_11/11/config/models/ada2_webface42m10faces_epoch9_double.onnx" --network "ir_101"

ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
