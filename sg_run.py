import io
import os
import json
import argparse
import numpy as np
import torch
import datetime


from lib.config import cfg
from lib.model import build_model
from lib.scene_parser.rcnn.utils.miscellaneous import mkdir, save_config, get_timestamp
from lib.scene_parser.rcnn.utils.comm import synchronize, get_rank
from lib.scene_parser.rcnn.utils.logger import setup_logger

def load_test_data():
    pass
    return img, file_name

def prediction(cg, args, model=None):
    """
    save bbox_result + triplet result in assigned directory
    test scene graph generation model
    """
    if model is None:
        arguments = {}
        arguments["iteration"] = 0
        model = build_model(cfg, arguments, args.local_rank, args.distributed)
    model.test(visualize=args.visualize)


def merge_json(bbx_json_dir, triplet_json_dir):
    """merge final json form to be sent"""
    pass
    return final_json

def main(cfg, args):
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
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

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

    img_dir = ""
    bbox_json_dir = ""
    triplet_json_dir = ""
    final_result_dir = ""

    img, file_name = load_test_data(img_dir)
    prediction(cfg, args)
    prediction(img, bbox_json_dir, triplet_json_dir)
    final_json = merge_json(bbox_json_dir, triplet_json_dir)

    if not os.path.exists(final_result_dir):
        os.mkdir(final_result_dir)
    with open(final_result_dir+'/'+file_name+'.json', 'w') as outfile:
        json.dump(final_json, outfile)


if __name__=='__main__':
    print('inferenceing start')
    ''' parse config file '''

