import sys
sys.path.append('SegGPT/SegGPT_inference')

import os
import argparse
import json
import torch as T
import torch.multiprocessing as mp
from agent import Agent
from typing import Dict
from utils import *
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from SegGPT.SegGPT_inference.models_seggpt import seggpt_vit_large_patch16_input896x448
from data import BaseDataset

def ddp_setup(rank: int, world_size: int, port:int=None):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(np.random.randint(10000, 60000)) if port is None else str(port)
    T.cuda.set_device(rank)
    T.cuda.empty_cache()
    init_process_group('nccl', rank=rank, world_size=world_size)


def main(rank: int, world_size: int, train_args: Dict, port: int):
    ddp_setup(rank, world_size, port)

    setup_logging()
    logger = get_logger(__name__, rank)

    logger.info('Preparing dataset')
    train_dataset = BaseDataset(
        root = train_args['train_dataset_dir'], 
        n_classes = train_args['n_classes'],
        mean = train_args['image_mean'],
        std = train_args['image_std'],
        mask_ratio = train_args['mask_ratio'],
        resize = (1024, 1024),
        is_train=True,
    )
    val_dataset = BaseDataset(
        root = train_args['val_dataset_dir'], 
        n_classes = train_args['n_classes'],
        mean = train_args['image_mean'],
        std = train_args['image_std'],
        mask_ratio = train_args['mask_ratio'],
        resize = (448, 448),
        is_train = False,
    )

    logger.info('Instantiating model and trainer agent')

    model = seggpt_vit_large_patch16_input896x448()
    initial_ckpt = T.load('seggpt_vit_large.pth', map_location='cpu')
    model.load_state_dict(initial_ckpt['model'], strict=False)
    logger.info('Initial checkpoint loaded')

    trainer = Agent(model, rank, train_args)
    logger.info(f'Using {T.cuda.device_count()} GPU(s)')
    if 'model_path' in train_args:
        trainer.load_checkpoint(train_args['model_path'])

    logger.info('Instantiating dataloader')
    train_dataloader = T.utils.data.DataLoader(
        train_dataset,
        batch_size=train_args['batch_size'],
        shuffle=False,
        num_workers=train_args['num_workers'],
        pin_memory=True,
        sampler=DistributedSampler(train_dataset),
    )
    val_dataloader = T.utils.data.DataLoader(
        val_dataset,
        batch_size=train_args['batch_size'],
        shuffle=False,
        num_workers=train_args['num_workers'],
        pin_memory=True,
        sampler=DistributedSampler(val_dataset),
    )

    trainer.do_training(train_dataloader, val_dataloader, train_args['eval_per_epoch'])
    destroy_process_group()

def get_args_parser():
    parser = argparse.ArgumentParser('SegGPT train', add_help=False)
    parser.add_argument('--uid', type=str, help='unique id for the run', default=None)
    parser.add_argument('--port', type=int, help='DDP port', default=None)
    parser.add_argument('--config', type=str, help='path to json config', default='configs/base.json')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args_parser()
    train_args = json.load(open(args.config, 'r'))
    train_args['uid'] = args.uid
    world_size = T.cuda.device_count()
    mp.spawn(main, nprocs=world_size, args=(world_size, train_args, args.port))
