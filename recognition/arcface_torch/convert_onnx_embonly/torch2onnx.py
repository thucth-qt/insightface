import numpy as np
import onnx
import torch
from collections import namedtuple
import torch
import torch.nn as nn
from torch.nn import Dropout
from torch.nn import MaxPool2d
from torch.nn import Sequential
from torch.nn import Conv2d, Linear
from torch.nn import BatchNorm1d, BatchNorm2d
from torch.nn import ReLU, Sigmoid
from torch.nn import Module
from torch.nn import PReLU
from backbone_adaface_embonly import build_model
    
def convert_onnx(net, path_module, output, opset=11, simplify=False):
    assert isinstance(net, torch.nn.Module)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(np.float)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()

    weight = torch.load(path_module, map_location=torch.device('cpu'))
    net.load_state_dict(weight, strict=True)
    net.eval()
    torch.onnx.export(net, \
                        img, \
                        output, \
                        input_names=["data"], \
                        output_names=["embedding"], \
                        keep_initializers_as_inputs=False, \
                        verbose=False, \
                        dynamic_axes={'data': {0: 'batch_size'}, 
                        'embedding': {0: 'batch_size'}}, \
                        opset_version=opset)
    model = onnx.load(output)
    graph = model.graph
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output)

    #export torch jit 
    net_jit = torch.jit.script(net)
    torch.jit.save(net_jit,output.replace(".onnx",".jit.pt"))
    
if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch to onnx')
    parser.add_argument('--input', type=str, default="/mnt/data/weights/adaface/wf42m10faces_pfc10_ir101_adaface_originalloss/model_5.pt", help='input backbone.pth file or path')
    parser.add_argument('--output', type=str, default="/mnt/data/weights/adaface/wf42m10faces_pfc10_ir101_adaface_originalloss/model_5_embonly.pt", help='output onnx path')
    parser.add_argument('--simplify', type=bool, default=False, help='onnx simplify')
    parser.add_argument('--network', type=str, default="ir_101", help='backbone network')

    args = parser.parse_args()
    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "model.pt")
    assert os.path.exists(input_file)

    backbone_torch_only_emb= build_model(model_name=args.network, fp16=False)
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "model.onnx")
    convert_onnx(backbone_torch_only_emb, input_file, args.output, simplify=args.simplify)