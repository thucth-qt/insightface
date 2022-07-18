## What did I change in this repo to run Adaface

0. add backbones from adaface repo (to make sure the exact backbone)

1. install dali for processing on GPU
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

2. define loss and head for new adaface
- ref config:
    - configs/ms1mv2_ir50_thucth_arcface01.py  : Ada backbone but Acrface head.
    - (update)  configs/ms1mv2_ir50_thucth_adaface1.py  : Ada backbone + Ada head - single GPU.
    - (new) ada fullflow - multiple GPUs (todo: load_state_dict, sample_rate): ./scritps/run_3gpu_ms1mv2.sh ms1mv2_ir100_thucth_adaface1.py





# FYI

To set breakpoints for debugging in multiprocessing:

```
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

ForkedPdb.set_trace()
 ```

