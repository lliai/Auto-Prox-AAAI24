import argparse
import copy
import gc
import csv
import shutil
import pycls.core.config as config
from pycls.core.config import cfg
from pycls.datasets.loader import _DATASETS
import pycls.core.logging as logging
import yaml
from typing import Union
import time
import os
import subprocess
import torch.nn.functional as F
from pycls.models.build import MODEL
import numpy as np
import pycls.datasets.loader as data_loader
import torch
from torch import Tensor
from pycls.predictor.pruners.predictive import find_measures
from autozc.structures import GraphStructure, LinearStructure, TreeStructure
from pycls.models.build import MODEL
from autozc.utils.rank_consistency import kendalltau, pearson, spearman



logger = logging.get_logger(__name__)



def obtain_gt(load_results):
    data_num = len(load_results[0].keys()) // 2
    arch_pop = []
    acc_pop = [[] for _ in range(data_num)]
    for item in load_results:
        arch_pop.append(item['arch'])
        for i in range(data_num):
            if i != 0:
                key = list(load_results[0].keys())[2 * i]
                acc_pop[i-1].append(item['{}'.format(key)])
            if i == data_num-1:
                key = list(load_results[0].keys())[-1]
                acc_pop[-1].append(item['{}'.format(key)])
    return arch_pop, acc_pop


def obtain_zc(cfg, csv_path):
    res = []
    classes = None
    if cfg.PROXY_DATASET == 'all':
        if cfg.MODEL.TYPE == 'PiT':
            classes = [100, 102, 4]
        elif cfg.MODEL.TYPE == 'AutoFormerSub':
            classes = [100, 102, 4, 1000]
        header = list(_DATASETS.keys())[:len(classes)]
        for item in header:
            with open(csv_path, 'r') as file:
                reader = csv.DictReader(file)
                temp = [eval(row[item]) for row in reader]
                res.append(temp)
    else:
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            temp = [eval(row[cfg.PROXY_DATASET]) for row in reader]
            res.extend(temp)
    return res




def prepare_trials(cfg, arch_pop, xargs, log_dir, temp_dir):
    bash_file = ['#!/bin/bash']

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)

    for idx, cand in enumerate(arch_pop):

        trial_name = '{}-{}'.format(cfg.MODEL.TYPE, idx)


        with open(xargs.refer_cfg) as f:
            refer_data = yaml.safe_load(f)
        trial_data = copy.deepcopy(refer_data)

        if cfg.MODEL.TYPE =='PIT':
            trial_data['PIT_SUBNET']['BASE_DIM']= cand['base_dim']
            trial_data['PIT_SUBNET']['MLP_RATIO'] = cand['mlp_ratio']
            trial_data['PIT_SUBNET']['DEPTH'] = cand['depth']
            trial_data['PIT_SUBNET']['NUM_HEADS'] = cand['num_heads']

        elif cfg.MODEL.TYPE =='AutoFormerSub':
            trial_data['AUTOFORMER_SUBNET']['HIDDEN_DIM'] = cand['hidden_dim']
            trial_data['AUTOFORMER_SUBNET']['MLP_RATIO'] = cand['mlp_ratio']
            trial_data['AUTOFORMER_SUBNET']['DEPTH'] = cand['depth']
            trial_data['AUTOFORMER_SUBNET']['NUM_HEADS'] = cand['num_heads']

        with open(temp_dir+'/{}.yaml'.format(trial_name), 'w') as f:
            yaml.safe_dump(trial_data, f, default_flow_style=False)
        if cfg.AUTO_PROX.type == None:
            execution_line = "CUDA_VISIBLE_DEVICES={}  python proxy_zc.py --save_dir {}  --csv  --csv_dir {} --refer_cfg {}/{}.yaml --other_zc {} ".format(
            xargs.gpu_idx, temp_dir , log_dir, temp_dir, trial_name, xargs.other_zc)
        else:
            execution_line = "CUDA_VISIBLE_DEVICES={}  python proxy_zc.py --save_dir {}  --csv  --csv_dir {} --refer_cfg {}/{}.yaml".format(
                xargs.gpu_idx, temp_dir, log_dir, temp_dir, trial_name)
        bash_file.append(execution_line)
    with open(os.path.join(temp_dir, 'run_bash.sh'), 'w') as handle:
        for line in bash_file:
            handle.write(line + os.linesep)
    subprocess.call("sh {}/run_bash.sh".format(temp_dir), shell=True)





def obtain_rank(cfg, args,  acc_pop, log_dir):

    if cfg.AUTO_PROX.type == None:
        csv_path = os.path.join(log_dir, '{}_{}_{}.csv'.format(cfg.MODEL.TYPE, args.other_zc, cfg.PROXY_DATASET))
    else:
        csv_path = os.path.join(log_dir,
                                '{}_{}_{}.csv'.format(cfg.MODEL.TYPE, cfg.AUTO_PROX.type, cfg.PROXY_DATASET))
    zc_score = obtain_zc(cfg, csv_path)


    if cfg.PROXY_DATASET == 'all':
        for i in range(len(acc_pop)):
            ken = kendalltau(acc_pop[i], zc_score[i])
            sp = spearman(acc_pop[i], zc_score[i])
            ps = pearson(acc_pop[i], zc_score[i])
            logger.info(f'Rank consistency on dataset {list(_DATASETS.keys())[i]}: kendall: {ken}, spearman: {sp}, pearson: {ps}')


    else:
        index = list(_DATASETS.keys()).index(cfg.PROXY_DATASET)
        # print('index is:', index)
        # print('acc_pop[index] is:', acc_pop[index])
        # print('zc score:', zc_score)
        ken = kendalltau(acc_pop[index], zc_score)
        sp = spearman(acc_pop[index], zc_score)
        ps = pearson(acc_pop[index], zc_score)
        logger.info(
            f'Rank consistency on dataset {list(_DATASETS.keys())[index]}: kendall: {ken}, spearman: {sp}, pearson: {ps}')






if __name__ == '__main__':

    parser = argparse.ArgumentParser( description='evo search zc',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--save_dir', type=str, default='work_dirs/rank_by_shell')
    parser.add_argument('--gt_path', type=str, default=None, help='ground truth path')
    parser.add_argument('--gpu_idx', type=int, default=None)

    parser.add_argument(
        '--refer_cfg', default='./configs/auto/autoformer/autoformer-ti-subnet_c100_base.yaml', type=str,
        help='save output path')
    parser.add_argument(
        '--other_zc',
        default=None,
        type=str,
        help=  'size, epe_nas, nwot, grasp, snip, ntk, fisher, synflow, dss'
    )

    args = parser.parse_args()
    config.load_cfg(args.refer_cfg)
    config.assert_cfg()




    logging.setup_logging()
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))

    save_path = os.path.join(args.save_dir, cfg.MODEL.TYPE)

    if cfg.AUTO_PROX.type == None:
        log_dir = os.path.join(args.save_dir,
                               '{}_{}_{}_{}'.format(cfg.MODEL.TYPE, args.other_zc, cfg.PROXY_DATASET, time_str))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir,
                                'rank_{}_{}_{}_{}.txt'.format(cfg.MODEL.TYPE, args.other_zc, cfg.PROXY_DATASET,
                                                                  time_str))
    else:
        log_dir = os.path.join(args.save_dir,
                               '{}_{}_{}_{}'.format(cfg.MODEL.TYPE, cfg.AUTO_PROX.type, cfg.PROXY_DATASET, time_str))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir,
                                'rank_{}_{}_{}_{}.txt'.format(cfg.MODEL.TYPE, cfg.AUTO_PROX.type, cfg.PROXY_DATASET,
                                                                  time_str))
    temp_dir = log_dir + '/temp'

    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    message = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
    logger.info(message)

    gt_results = torch.load(args.gt_path)
    arch_pop, acc_pop = obtain_gt(gt_results)

    data_loader = data_loader.construct_proxy_loader()

    t1 = time.time()
    prepare_trials(cfg, arch_pop, args, log_dir ,temp_dir)
    # log_dir = os.path.join(args.save_dir,'AutoFormerSub_ntk_imagenet_2023-03-01_19-44-00')
    obtain_rank(cfg, args,  acc_pop, log_dir)
    t2 = time.time()
    logger.info('Finished, time cost: {} hour.'.format((t2 - t1) / 3600))
    # shutil.rmtree(temp_dir)









