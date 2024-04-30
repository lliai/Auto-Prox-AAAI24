#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified from pycls.
https://github.com/facebookresearch/pycls/blob/main/tools/run_net.py
"""

import argparse
import sys
import os

import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.autoformer_trainer as trainer
from pycls.core.config import cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Run a model.")
    help_s, choices = "Run mode", [ "train", "time", "test"]
    parser.add_argument("--mode", help=help_s, choices=choices, required=True, type=str)
    help_s = "Config file location"
    parser.add_argument("--cfg", help=help_s, required=True, type=str)
    help_s = "See pycls/core/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode
    config.load_cfg(args.cfg)
    cfg.merge_from_list(args.opts)
    if cfg.OUT_DIR is None:
        out_dir = os.path.join('work_dirs', os.path.splitext(os.path.basename(args.cfg))[0])
        cfg.OUT_DIR = out_dir
    config.assert_cfg()
    cfg.freeze()
    if not os.path.exists(cfg.OUT_DIR):
        os.makedirs(cfg.OUT_DIR)
    if mode == 'train':
        dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.train_model)
    elif mode == "test":
        arch_cfg = {'mlp_ratio': [4.0, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 4.0, 4.0, 3.5, 3.5, 3.5],
                    'num_heads': [4, 4, 3, 3, 4, 3, 4, 4, 3, 3, 4, 3, 3], 'hidden_dim': 192, 'depth': 13}
        dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.test_model(arch_cfg))
    elif mode == "time":
        dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.time_model)


if __name__ == "__main__":
    main()
