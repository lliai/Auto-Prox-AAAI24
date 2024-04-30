#!/bin/bash
CUDA_VISIBLE_DEVICES=0  python search_model_shell.py  --trial_num 500  --gpu_idx 0 --refer_cfg configs/auto/rank/pit/pit-other-flowers.yaml  --other_zc nwot
