import torch 
from memory_profiler import profile
import sys
from threading import Thread

class Child(Thread):
    @profile
    def run(self):
        a_child = torch.ones((1000,100))
        b_child = torch.ones((1000,1000))
        c_child = torch.ones((1000,10000))
        d_child = torch.ones((1000,100000))    
@profile
def main():
    a = torch.ones((1000,100))
    b = torch.ones((1000,1000))
    c = torch.ones((1000,10000))
    d = torch.ones((1000,100000))

    child = Child()
    child.start()
    child.join()

    return
if __name__ == "__main__":
    main()