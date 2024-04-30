import argparse
import random
import os
import torch.nn.functional as F
import pycls.core.config as config
import pycls.core.logging as logging
import pycls.datasets.loader as data_loader
from pycls.core.config import cfg
from pycls.predictor.pruners.predictive import find_measures
from pycls.models.build import MODEL
from pycls.models.distill import DistillationWrapper
from autozc.structures import GraphStructure, LinearStructure, TreeStructure
from proxy_zc import auto_prox_fitness, other_fitness
import torch
import time
import gc
import yaml
import copy

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

            temp_model = MODEL.get('AutoFormerSub')(arch_config=config, num_classes=cfg.MODEL.NUM_CLASSES)
            temp_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)

            if config not in trial_configs and args.af_low <= round(temp_params/1e6) <= args.af_up:
                flag = True
                trial_configs.append(config)
                logger.info('generate {}-th AF config: {}, param: {} M'.format(idx, config, round(temp_params / 1e6)))
            else:
                logger.info('not suitable, AF param is:{} M'.format(round(temp_params/1e6)))

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
            temp_model = MODEL.get('PiT')(arch_config = config, num_classes=cfg.MODEL.NUM_CLASSES)
            temp_params= sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
            if config not in trial_configs and args.pit_low <= round(temp_params/1e6) <= args.pit_up:
                flag = True
                trial_configs.append(config)
                logger.info('generate {}-th PIT config: {}, param: {} M'.format(idx, config, round(temp_params / 1e6)))
            else:
                logger.info('not suitable, PIT param is:{} M'.format(round(temp_params/1e6)))

    return trial_configs



def sample_trial_configs(model_type, args):
    pop= None
    trial_num = args.trial_num
    if model_type =='AutoFormerSub':
        pop = autoformer_configs(trial_num)
    elif model_type == 'PiT':
        pop = pit_configs(trial_num, args)
    return pop


def standrd(data):
    min_value = torch.min(data)
    max_value = torch.max(data)
    res = (data-min_value)/(max_value-min_value)
    return res



def save_cfg(refer_path, searched_cfg ,exp_name, cfg):
    with open(refer_path) as f:
        refer_data = yaml.safe_load(f)
    trial_data = copy.deepcopy(refer_data)

    if cfg.MODEL.TYPE == 'PiT':
        trial_data['PIT_SUBNET']['BASE_DIM'] = searched_cfg['base_dim']
        trial_data['PIT_SUBNET']['MLP_RATIO'] = searched_cfg['mlp_ratio']
        trial_data['PIT_SUBNET']['DEPTH'] = searched_cfg['depth']
        trial_data['PIT_SUBNET']['NUM_HEADS'] = searched_cfg['num_heads']

    elif cfg.MODEL.TYPE == 'AutoFormerSub':
        trial_data['AUTOFORMER_SUBNET']['HIDDEN_DIM'] = searched_cfg['hidden_dim']
        trial_data['AUTOFORMER_SUBNET']['MLP_RATIO'] = searched_cfg['mlp_ratio']
        trial_data['AUTOFORMER_SUBNET']['DEPTH'] = searched_cfg['depth']
        trial_data['AUTOFORMER_SUBNET']['NUM_HEADS'] = searched_cfg['num_heads']

    yaml_dir = 'configs/auto/retrain/' +  cfg.MODEL.TYPE
    if not os.path.exists(yaml_dir):
        os.makedirs(yaml_dir, exist_ok=True)

    with open(yaml_dir + '/{}.yaml'.format(exp_name), 'w') as f:
        yaml.safe_dump(trial_data, f, default_flow_style=False)





def rs_by_other(cfg, arch_pop, data_loader, num_classes, zc_name='dss'):
    best_score = float("-inf")
    best_cfg = None

    for arch_idx, arch_cfg in enumerate(arch_pop):
        # build model
        other_zc = other_fitness(cfg, data_loader, zc_name, num_classes)

        # logger.info(f'The {arch_idx}-th arch , {zc_name} Score: {other_zc}')
        logger.info(f'config: {arch_cfg}')
        logger.info(f'The {arch_idx}-th arch , {zc_name} Score: {other_zc}')
        if other_zc > best_score:
            best_score = other_zc
            best_cfg = arch_cfg

        gc.collect()
        torch.cuda.empty_cache()

    return best_cfg, best_score


def rs_by_autoprox(cfg, arch_pop, data_loader, num_classes, structure):
    best_score = float("-inf")
    best_cfg = None

    for arch_idx, arch_cfg in enumerate(arch_pop):
        # build model
        autoprox_zc = auto_prox_fitness(cfg, data_loader, structure, num_classes)

        # logger.info(f'The {arch_idx}-th arch , autoprox  {cfg.AUTO_PROX.type} Score: {autoprox_zc}')
        logger.info(f'config: {arch_cfg} ')
        logger.info(f'Auto_prox  {cfg.AUTO_PROX.type} Score: {autoprox_zc}')
        if autoprox_zc > best_score:
            best_score = autoprox_zc
            best_cfg = arch_cfg

        gc.collect()
        torch.cuda.empty_cache()

    return best_cfg, best_score



def parse_args():
    parser = argparse.ArgumentParser(
        description='parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)



    # random
    parser.add_argument('--trial_num', default=500,
                        type=int, help='number of mutate')

    parser.add_argument('--save_dir', default='work_dirs/search_model', type=str)

    parser.add_argument(
        '--refer_cfg', default='./configs/auto/autoformer/autoformer-ti-subnet_c100_base.yaml', type=str,
        help='save output path')

    # parser.add_argument(
    #     '--save_refer', default=None, type=str,
    #     help='save output path')

    parser.add_argument("--other_zc", type=str, help="zero cost metric name", default=None)


    parser.add_argument("--pit_up", default=22, type=float, help="pit param upper limit")
    parser.add_argument("--pit_low", default=4, type=float, help="pit param lower limit")

    parser.add_argument("--af_up", default=9, type=float, help="autoformer param upper limit")
    parser.add_argument("--af_low", default=4, type=float, help="autoformer param lower limit")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config.load_cfg(args.refer_cfg)
    config.assert_cfg()

    logging.setup_logging()
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    if cfg.AUTO_PROX.type == None:
        log_file = os.path.join(args.save_dir,
                            'rs_model_{}_{}_{}_{}.txt'.format(cfg.MODEL.TYPE, args.other_zc, cfg.PROXY_DATASET, time_str))
        exp_name = '{}_{}_{}_{}'.format(cfg.MODEL.TYPE, args.other_zc, cfg.PROXY_DATASET, time_str)
    else:
        log_file = os.path.join(args.save_dir,
                                'rs_model_{}_{}_{}_{}.txt'.format(cfg.MODEL.TYPE, cfg.AUTO_PROX.type, cfg.PROXY_DATASET,
                                                              time_str))
        exp_name = '{}_{}_{}_{}'.format(cfg.MODEL.TYPE, cfg.AUTO_PROX.type, cfg.PROXY_DATASET, time_str)
    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    message = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
    logger.info(message)

    data_loader = data_loader.construct_proxy_loader()

    # get arch_configs
    arch_pop = sample_trial_configs(cfg.MODEL.TYPE, args)

    structure = None

    if cfg.AUTO_PROX.type == None:
        logger.info(f'Build {args.other_zc} ....')
        t1 = time.time()
        best_cfg, best_score = rs_by_other(cfg, arch_pop, data_loader, cfg.MODEL.NUM_CLASSES, zc_name=args.other_zc)
        t2 = time.time()
        logger.info('Finished, time cost: {} hour.'.format((t2 - t1) / 3600))
        logger.info(f'best arch config: {best_cfg}, best score: {best_score}')

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
        best_cfg, best_score = rs_by_autoprox(cfg, arch_pop, data_loader, cfg.MODEL.NUM_CLASSES, structure)
        t2 = time.time()
        logger.info('Finished, time cost: {} hour.'.format((t2 - t1) / 3600))
        logger.info(f'best arch config: {best_cfg}, best score: {best_score}')

    searched_model = MODEL.get(cfg.MODEL.TYPE)(arch_config=best_cfg, num_classes=cfg.MODEL.NUM_CLASSES)
    print('searched model:', searched_model)
    temp_params = sum(p.numel() for p in searched_model.parameters() if p.requires_grad)
    logger.info(f'params: {(temp_params/1e6)} M.')
    save_cfg(args.refer_cfg, best_cfg, exp_name, cfg)



