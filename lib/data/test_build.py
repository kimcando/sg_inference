import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as F
import numpy as np
import os, sys
from glob import glob
from lib.config import cfg

def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_horizontal_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = T.Compose(
        [
            color_jitter,
            T.Resize(min_size, max_size),
            # T.RandomHorizontalFlip(flip_horizontal_prob), # NOTE: mute this since spatial repations is snesible to this
            # T.RandomVerticalFlip(flip_vertical_prob), # NOTE: mute this since spatial repations is snesible to this
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform

def build_data_loader(cfg, data_dir="/home/ncl/ADD_sy/inference/sg_inference/input_data", split="test"):
    "single image per batch"
    transforms = build_transforms(cfg, is_train=True if split == "train" else False)
    dataset = ImageFolder(root=data_dir, transform = transforms)
    data_loader = DataLoader(dataset = dataset, batch_size=1,
                             shuffle=True,
                             num_workers=0)
    return data_loader

if __name__=='__main__':
    cfg.merge_from_file("configs/baseline_res101.yaml")

    d = build_data_loader()