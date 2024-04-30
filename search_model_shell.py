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
import random
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



def autoformer_configs(trial_num):
    trial_configs = []
    choices = {'num_heads': [3, 4], 'mlp_ratio': [3.5, 4.0],
               'hidden_dim': [192, 216, 240], 'depth': [12, 13, 14]}
    dimensions = ['mlp_ratio', 'num_heads']

    for idx in range(trial_num):
        flag = False
        while not flag:
            depth = random.choice(choices['depth'])
            config = {
                dimension: [
                    random.choice(choices[dimension]) for _ in range(depth)
                ]
                for dimension in dimensions
            }
            config['hidden_dim'] = random.choice(choices['hidden_dim'])
            config['depth'] = depth

            if config not in trial_configs:
                flag = True
                trial_configs.append(config)
                # logger.info(f'generate {idx}-th config: {config}')

    return trial_configs



def pit_configs(trial_num, args):

    trial_configs = []
    logger.info('Pit param limit: {}--{}'.format(args.pit_low, args.pit_up))
    choices = {'base_dim': [16, 24, 32, 40], 'mlp_ratio': [2, 4, 6, 8],
               'num_heads': [[2,2,2], [2,2,4], [2,2,8], [2,4,4], [2,4,8], [2,8,8], [4,4,4], [4,4,8], [4,8,8], [8,8,8]],
               'depth': [[1,6,6], [1,8,4], [2,4,6], [2,6,4], [2,6,6], [2,8,2], [2,8,4], [3,4,6], [3,6,4], [3,8,2]]}
    for idx in range(trial_num):

        flag = False
        while not flag:
            config = {}
            dimensions = ['mlp_ratio', 'num_heads', 'base_dim', 'depth']
            for dimension in dimensions:
                config[dimension] = random.choice(choices[dimension])
            temp_model = MODEL.get('PiT')(arch_config = config)
            temp_params= sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
            if config not in trial_configs and args.pit_low <= round(temp_params/1e6) <= args.pit_up:
                flag = True
                trial_configs.append(config)
                logger.info('generate {}-th config: {}, param: {} M'.format(idx, config, round(temp_params / 1e6)))
            else:
                logger.info('not suitable, param is:{} M'.format(round(temp_params/1e6)))

    return trial_configs



def sample_trial_configs(model_type, args):
    pop= None
    trial_num = args.trial_num
    if model_type =='AutoFormerSub':
        pop = autoformer_configs(trial_num)
    elif model_type == 'PiT':
        pop = pit_configs(trial_num, args)
    return pop


def obtain_zc(cfg, csv_path):
    res = []

    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        temp = [eval(row[cfg.PROXY_DATASET]) for row in reader]
        res.append(temp)
    return res[0]




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





def obtain_optimal_model(cfg, args,  arch_pop, log_dir):

    if cfg.AUTO_PROX.type == None:
        csv_path = os.path.join(log_dir, '{}_{}_{}.csv'.format(cfg.MODEL.TYPE, args.other_zc, cfg.PROXY_DATASET))
        zc_name = args.other_zc
    else:
        csv_path = os.path.join(log_dir,
                                '{}_{}_{}.csv'.format(cfg.MODEL.TYPE, cfg.AUTO_PROX.type, cfg.PROXY_DATASET))
        zc_name = cfg.AUTO_PROX.type

    zc_score = obtain_zc(cfg, csv_path)
    best_index = zc_score.index(max(zc_score))
    best_score = zc_score[best_index]
    best_cfg = arch_pop[best_index]
    for arch_id, arch_cfg in enumerate(arch_pop):
        logger.info(f'config: {arch_cfg} ')
        logger.info(f'{zc_name} score: {zc_score[arch_id]}')

    return best_cfg, best_score








if __name__ == '__main__':

    parser = argparse.ArgumentParser( description='evo search zc',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--trial_num', default=1000,
                        type=int, help='number of mutate')

    parser.add_argument('--save_dir', type=str, default='work_dirs/search_model_shell')
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
    parser.add_argument("--pit_up", default=22, type=float, help="pit param upper limit")
    parser.add_argument("--pit_low", default=4, type=float, help="pit param lower limit")

    args = parser.parse_args()
    config.load_cfg(args.refer_cfg)
    config.assert_cfg()




    logging.setup_logging()
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))


    if cfg.AUTO_PROX.type == None:
        log_dir =  os.path.join(args.save_dir,
                            '{}_{}_{}_{}'.format(cfg.MODEL.TYPE, args.other_zc, cfg.PROXY_DATASET, time_str))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir,
                            'rs_model_{}_{}_{}_{}.txt'.format(cfg.MODEL.TYPE, args.other_zc, cfg.PROXY_DATASET, time_str))
    else:
        log_dir = os.path.join(args.save_dir,
                               '{}_{}_{}_{}'.format(cfg.MODEL.TYPE, cfg.AUTO_PROX.type, cfg.PROXY_DATASET, time_str))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir ,
                                'rs_model_{}_{}_{}_{}.txt'.format(cfg.MODEL.TYPE, cfg.AUTO_PROX.type, cfg.PROXY_DATASET,
                                                              time_str))
    temp_dir = log_dir+'/temp'

    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    message = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
    logger.info(message)

    arch_pop = sample_trial_configs(cfg.MODEL.TYPE, args)

    data_loader = data_loader.construct_proxy_loader()

    t1 = time.time()
    prepare_trials(cfg, arch_pop, args, log_dir, temp_dir)
    # log_dir = os.path.join(args.save_dir,'AutoFormerSub_nwot_cifar100_2023-03-01_00-11-53')
    best_cfg, best_score = obtain_optimal_model(cfg, args,  arch_pop, log_dir)
    t2 = time.time()
    logger.info('Finished, time cost: {} hour.'.format((t2 - t1) / 3600))
    logger.info(f'best arch config: {best_cfg}, best score: {best_score}')
    shutil.rmtree(temp_dir)









