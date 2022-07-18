## What did I change in this repo to run Adaface

0. add backbones from adaface repo (to make sure the exact backbone)

1. install dali for processing on GPU
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

2. define loss and head for new adaface
- ref config:
    - configs/ms1mv2_ir50_thucth_arcface01.py  : Ada backbone but Acrface head.
    - (update)  configs/ms1mv2_ir50_thucth_adaface1.py  : Ada backbone + Ada head.
