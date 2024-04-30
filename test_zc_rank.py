import argparse
import copy
import gc
import math
import pycls.core.config as config
from pycls.core.config import cfg
from pycls.datasets.loader import _DATASETS
import pycls.core.logging as logging
import random
from typing import Union
import time
import os
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



def all_same(items):
    return all(x == items[0] for x in items)


def build_model(arch_config, cfg, num_classes):
    model = MODEL.get(cfg.MODEL.TYPE)(arch_config=arch_config, num_classes=num_classes)
    return model


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



def is_anomaly(zc_score: Union[torch.Tensor, float, int] = None) -> bool:
    """filter the score with -1,0,nan,inf"""
    if isinstance(zc_score, Tensor):
        zc_score = zc_score.item()
    if zc_score is None or zc_score == -1 or math.isnan(
            zc_score) or math.isinf(zc_score) or zc_score == 0:
        return True
    return False



def is_anomaly_group(zc_group) -> bool:
    """filter the score with -1,0,nan,inf"""
    for item in zc_group:
        if is_anomaly(item):
            return True
    return False



def auto_prox_fitness(cfg, data_loader, arch_pop, acc_pop, structure, num_classes):
    """structure is belong to popultion."""

    gt_score = acc_pop
    zc_score = []

    data_iter = iter(data_loader)
    try:
        img, label, _ = next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        img, label, _ = next(data_iter)

    for arch_idx, arch_cfg in enumerate(arch_pop):
        temp_model = build_model(arch_cfg, cfg, num_classes)
        if torch.cuda.is_available():
            temp_model.cuda()

        zc = structure(img, label, temp_model)
        logger.info(f'The {arch_idx}-th {cfg.MODEL.TYPE} arch:, sturcture zc score: {zc}')
        if is_anomaly(zc):
            return -1
        # early exit
        if len(zc_score) > 3 and all_same(zc_score):
            return -1
        zc_score.append(zc)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    # TODO add inf check
    if len(zc_score) <= 1 or np.isnan(kendalltau(gt_score, zc_score)):
        return -1

    # release memory
    del img, label, temp_model
    torch.cuda.empty_cache()
    gc.collect()

    ken = kendalltau(gt_score, zc_score)
    sp = spearman( gt_score, zc_score)
    ps = pearson( gt_score, zc_score)

    return ken, sp, ps



def other_fitness(cfg, data_loader, arch_pop, acc_pop, zc_name, num_classes):
    gt_score = acc_pop
    zc_score = []
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    for arch_idx, arch_cfg in enumerate(arch_pop):
        temp_model = build_model(arch_cfg, cfg, num_classes)
        if torch.cuda.is_available():
            temp_model.cuda()

        if zc_name == 'grasp':
            dataload_info = ['grasp', 1, num_classes]
        else:
            dataload_info = ['random', 1, num_classes]

        zc = find_measures(temp_model, data_loader,
                           dataload_info=dataload_info,
                           device=device,
                           loss_fn=F.cross_entropy,
                           measure_names=[zc_name])

        logger.info(f'The {arch_idx}-th {cfg.MODEL.TYPE} arch:, {zc_name} score: {zc}')
        if is_anomaly(zc):
            return -1

        zc_score.append(zc)

        del dataload_info, temp_model
        torch.cuda.empty_cache()
        gc.collect()

    ken = kendalltau(gt_score, zc_score)
    sp = spearman( gt_score, zc_score)
    ps = pearson( gt_score, zc_score)

    return ken, sp, ps





def auto_prox_rank(cfg, arch_pop, acc_pop, data_loader, struct):

    if cfg.PROXY_DATASET == 'all':
        logger.info(f'len(data_loader): {len(data_loader)}')
        for i in range(len(acc_pop)):
            classes =[100, 102, 4, 1000]

            single_kendall, single_spearman, single_pearson = auto_prox_fitness(cfg, data_loader[i], arch_pop, acc_pop[i], struct, num_classes=classes[i])
            logger.info(f'Rank consistency on dataset {list(_DATASETS.keys())[i]}: kendall: {single_kendall}, spearman: {single_spearman}, pearson: {single_pearson}')

    else:
        index = list(_DATASETS.keys()).index(cfg.PROXY_DATASET)
        kendall_score, spearman_score, pearson_score = auto_prox_fitness(cfg, data_loader, arch_pop, acc_pop[index], struct, num_classes=cfg.MODEL.NUM_CLASSES)
        logger.info(f'Rank consistency on dataset {cfg.PROXY_DATASET}: kendall: {kendall_score}, spearman: {spearman_score}, pearson: {pearson_score}')




def other_rank(cfg, data_loader, arch_pop, acc_pop, zc_name):

    if cfg.PROXY_DATASET == 'all':
        for i in range(len(acc_pop)):
            classes =[100, 102, 4, 1000]

            single_kendall, single_spearman, single_pearson = other_fitness(cfg, data_loader[i], arch_pop, acc_pop[i], zc_name, classes[i])
            logger.info(f'Rank consistency on dataset {list(_DATASETS.keys())[i]}: kendall: {single_kendall}, spearman: {single_spearman}, pearson: {single_pearson}')


    else:
        index = list(_DATASETS.keys()).index(cfg.PROXY_DATASET)
        kendall_score, spearman_score, pearson_score = other_fitness(cfg, data_loader, arch_pop, acc_pop[index], zc_name, num_classes=cfg.MODEL.NUM_CLASSES)

        logger.info(f'Rank consistency on dataset {cfg.PROXY_DATASET}: kendall: {kendall_score}, spearman: {spearman_score}, pearson: {pearson_score}')






if __name__ == '__main__':

    parser = argparse.ArgumentParser( description='evo search zc',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--save_dir', type=str, default='./work_dirs/rank')
    parser.add_argument('--gt_path', type=str, default=None, help='ground truth path')
    parser.add_argument(
        '--refer_cfg', default='./configs/auto/autoformer/autoformer-ti-subnet_c100_base.yaml', type=str,
        help='save output path')

    parser.add_argument(
        '--other_zc',
        default=None,
        type=str,
        help=  'size, epe_nas, grasp, snip, ntk, fisher, synflow, dss, (nwot is not included with memory oom)'
    )



    args = parser.parse_args()
    config.load_cfg(args.refer_cfg)
    config.assert_cfg()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)



    logging.setup_logging()
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    if cfg.AUTO_PROX.type == None:
        log_file = os.path.join(args.save_dir,
                            'rank_{}_{}_{}_{}.txt'.format(cfg.MODEL.TYPE, args.other_zc, cfg.PROXY_DATASET, time_str))
    else:
        log_file = os.path.join(args.save_dir,
                                'rank_{}_{}_{}_{}.txt'.format(cfg.MODEL.TYPE, cfg.AUTO_PROX.type, cfg.PROXY_DATASET,
                                                              time_str))
    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    message = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
    logger.info(message)

    gt_results = torch.load(args.gt_path)
    arch_pop, acc_pop = obtain_gt(gt_results)

    data_loader = data_loader.construct_proxy_loader()

    # preprocess search space structure

    structure = None

    if cfg.AUTO_PROX.type == None:
        logger.info(f'Build {args.other_zc} ....')
        t1 = time.time()
        other_rank(cfg, data_loader, arch_pop, acc_pop, args.other_zc)
        t2 = time.time()
        logger.info('Finished, time cost: {} hour.'.format((t2 - t1) / 3600))
    else:

        if cfg.AUTO_PROX.type == 'linear':
            structure = LinearStructure()
        elif cfg.AUTO_PROX.type == 'tree':
            structure = TreeStructure()
        elif cfg.AUTO_PROX.type == 'graph':
            structure = GraphStructure()

        logger.info('Build zc structure....')
        structure._genotype['input_geno'] = cfg.AUTO_PROX.input_geno
        structure._genotype['op_geno'] = cfg.AUTO_PROX.op_geno
        structure._genotype['repr_geno'] = cfg.AUTO_PROX.repr_geno

        t1 = time.time()
        auto_prox_rank(cfg, arch_pop, acc_pop, data_loader, structure)
        t2 = time.time()
        logger.info('Finished, time cost: {} hour.'.format((t2 - t1) / 3600))






