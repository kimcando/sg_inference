import os
import argparse
import numpy as np
from glob import glob

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

from lib.config import cfg
from lib.model import build_model
from lib.scene_parser.rcnn.utils.miscellaneous import mkdir, save_config, get_timestamp
from lib.scene_parser.rcnn.utils.comm import synchronize, get_rank
from lib.scene_parser.rcnn.utils.logger import setup_logger
from data_handler import Dataset

class SGInference(object):
    def __init__(self, cg, args):
        self.cg = cg
        self.args = args

    def prediction(self, cfg, args, loader, model=None):
        if model is None:
            arguments = {}
            arguments["iteration"] = 0
            model = build_model(cfg, args, arguments, args.local_rank, args.distributed)
        model.test(loader, output_folder='results/bbox/', visualize=args.visualize, live=args.live)

    def load_test_loader(self, img_dir = "/home/ncl/ADD_sy/inference/sg_inference/data"):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        dataset = Dataset(root=img_dir,
                          transform=None)
        test_loader = DataLoader(dataset=dataset, batch_size=1,
                                 shuffle=True,
                                 num_workers=0)
        return test_loader

    def post_process(self):
        pass


def prediction(cg, args, loader, model=None):
    """
    save bbox_result + triplet result in assigned directory
    test scene graph generation model
    """
    if model is None:
        arguments = {}
        arguments["iteration"] = 0
        model = build_model(cfg, args, arguments, args.local_rank, args.distributed)
    model.test(loader, output_folder = 'results/bbox/',visualize=args.visualize, live=args.live)

def load_test_loader(img_dir="/home/ncl/ADD_sy/inference/sg_inference/data"):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = Dataset(root=img_dir,
                      transform=None)
    test_loader = DataLoader(dataset = dataset, batch_size=1,
                             shuffle=True,
                             num_workers=0)
    return test_loader


def main():
    # TODO
    # single test inference
    ''' parse config file '''
    parser = argparse.ArgumentParser(description="Scene Graph Generation")
    parser.add_argument("--config-file", default="configs/baseline_res101.yaml")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--batchsize", type=int, default=0)
    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--instance", type=int, default=-1)
    parser.add_argument("--use_freq_prior", action='store_true')
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--algorithm", type=str, default='sg_baseline')
    
    parser.add_argument("--live", action='store_true')
    parser.add_argument("--raw_img", action='store_true')
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    # import pdb; pdb.set_trace()
    cfg.merge_from_file(args.config_file)
    cfg.resume = args.resume
    cfg.instance = args.instance
    cfg.inference = args.inference
    cfg.MODEL.USE_FREQ_PRIOR = args.use_freq_prior
    cfg.MODEL.ALGORITHM = args.algorithm
    if args.batchsize > 0:
        cfg.DATASET.TRAIN_BATCH_SIZE = args.batchsize
    if args.session > 0:
        cfg.MODEL.SESSION = str(args.session)
    # cfg.freeze()

    if not os.path.exists("logs") and get_rank() == 0:
        os.mkdir("logs")
    logger = setup_logger("scene_graph_generation", "logs", get_rank(),
                          filename="{}_{}.txt".format(args.algorithm, get_timestamp()))

    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    output_config_path = os.path.join("logs", 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)
    model_inference = SGInference(cfg, args)
    data_loader = model_inference.load_test_loader(img_dir = '/home/ncl/ADD_sy/inference/sg_inference/data')
    model_inference.prediction(cfg, args, data_loader)

    # img_dir = '/home/ncl/ADD_sy/inference/sg_inference/data'
    # test_loader = load_test_loader(img_dir=img_dir)
    # prediction(cfg, args, test_loader)

if __name__=='__main__':
    print('inferenceing start')
    ''' parse config file '''
    main()