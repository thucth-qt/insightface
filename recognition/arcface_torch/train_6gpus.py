import torch 
from torch import nn
import resource
import sys

# gpus = [1,2]
def main():
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
    # container = []
    # for _ in range(10):
    #     container.append(torch.ones((26843, 10000), dtype=torch.float32))
    # while True:
    #     pass

def memory_limit(percent):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    total_mem =get_memory()*1024
    limit_mem = int(percent*total_mem)
    print("Total mem: %d (bytes)"%total_mem)
    print("limit_mem: %d (bytes)"%limit_mem)
    resource.setrlimit(resource.RLIMIT_AS, (limit_mem, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

if __name__ == '__main__':
    memory_limit(0.2) # Limitates maximun memory usage to half
    try:
        main()
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)