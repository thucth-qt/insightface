import torch 
from torch import nn
# gpus = [1,2]
gpus = range(0,6)
values=[]
for gpu in gpus:
    device = torch.device(gpu)
    a = torch.ones((10000,10000))
    a = a.to(device)
    values.append(a)
import time
while True:
    time.sleep(10)
    for gpu in gpus:
        device = torch.device(gpu)
        a = torch.ones((10000,10000))
        a = a.to(device)