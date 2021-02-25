import io
import os
import json
import argparse
import numpy as np
import torch
import datetime
from PIL import Image
from glob import glob

from lib.config import cfg
from lib.model import build_model
from lib.scene_parser.rcnn.utils.miscellaneous import mkdir, save_config, get_timestamp
from lib.scene_parser.rcnn.utils.comm import synchronize, get_rank
from lib.scene_parser.rcnn.utils.logger import setup_logger

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from test_build import Dataset

# from torchvision.datasets import ImageFolder
# from torchvision.transforms import ToTensor
from torchvision.utils import save_image

def load_test_data(img_dir, test_save_img = False):
    img_list = glob(f'{img_dir}/*')
    # for img_file in img_list:
    img_pil = Image.open(img_list[0])
    img_pil = np.array(img_pil).transpose(2,0,1)
    # img = ToTensor()(img_pil).unsqueeze(0) # ToTensor 하면 normalize 도 자동으로 됨
    img = torch.from_numpy(img_pil).unsqueeze(0).float()
    file_name = img_dir.split('/')[-1].split('.')[0]
    if test_save_img:
        save_image(img_dir)
    return img, file_name

def load_test_loader(img_dir="/home/ncl/ADD_sy/inference/sg_inference/"):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = Dataset(root='/home/ncl/ADD_sy/inference/sg_inference/data',
                      transform=None)
    test_loader = DataLoader(dataset = dataset, batch_size=1,
                             shuffle=True,
                             num_workers=0)
    return test_loader
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
    # model.test_0220(output_folder='results/bbox/', visualize=args.visualize, live=args.live)


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
    parser.add_argument("--single_test", action='store_true')
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

    img_dir = "/home/ncl/ADD_sy/inference/sg_inference/input_data/"
    bbox_json_dir = ""
    triplet_json_dir = ""
    final_result_dir = ""

    # single
    # img, file_name = load_test_data(img_dir)
    # prediction(cfg, args, img)
    test_loader = load_test_loader()
    prediction(cfg, args, test_loader)
    # prediction(img, bbox_json_dir, triplet_json_dir)

    #
    # if not os.path.exists(final_result_dir):
    #     os.mkdir(final_result_dir)
    # with open(final_result_dir+'/'+file_name+'.json', 'w') as outfile:
    #     json.dump(final_json, outfile)


if __name__=='__main__':
    print('inferenceing start')
    ''' parse config file '''
    main()

