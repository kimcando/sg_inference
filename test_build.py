import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as F
from torchvision import transforms
import numpy as np
import os, sys
from glob import glob
from lib.config import cfg
import argparse
###
import torch.utils.data as data
import os
import re
import torch
from PIL import Image

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    if class_to_idx is None:
        class_to_idx = dict()
        build_class_idx = True
    else:
        build_class_idx = False
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        if build_class_idx and not subdirs:
            class_to_idx[label] = None
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
    if build_class_idx:
        classes = sorted(class_to_idx.keys(), key=natural_key)
        for idx, c in enumerate(classes):
            class_to_idx[c] = idx
    images_and_targets = zip(filenames, [class_to_idx[l] for l in labels])
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    if build_class_idx:
        return images_and_targets, classes, class_to_idx
    else:
        return images_and_targets


class Dataset(data.Dataset):
    def __init__(
            self,
            root,
            transform=None):

        imgs, _, _ = find_images_and_targets(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = np.array(Image.open(path).convert('RGB')).transpose(2,0,1)
        img = torch.from_numpy(img).float()
        # img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.imgs)

    def set_transform(self, transform):
        self.transform = transform

    def filenames(self, indices=[], basename=False):
        if indices:
            if basename:
                return [os.path.basename(self.imgs[i][0]) for i in indices]
            else:
                return [self.imgs[i][0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.imgs]
            else:
                return [x[0] for x in self.imgs]




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
    # import pdb; pdb.set_trace()
    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = T.Compose(
        [
            # color_jitter,
            T.Resize(min_size, max_size),
            # # T.RandomHorizontalFlip(flip_horizontal_prob), # NOTE: mute this since spatial repations is snesible to this
            # # T.RandomVerticalFlip(flip_vertical_prob), # NOTE: mute this since spatial repations is snesible to this
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform

def build_data_loader(cfg, data_dir="/home/ncl/ADD_sy/inference/sg_inference/", split="test"):
    """single image per batch
    data will be loaded containing in subfolder of data_dir+'data/'
    I manually configured first depth folder 'data',"""
    transforms = build_transforms(cfg, is_train=True if split == "train" else False)
    dataset = ImageFolder(root=data_dir, transform = transforms)
    data_loader = DataLoader(dataset = dataset, batch_size=1,
                             shuffle=True,
                             num_workers=0)
    return data_loader

if __name__=='__main__':

    # import pdb; pdb.set_trace()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = Dataset(root= '/home/ncl/ADD_sy/inference/sg_inference/data',
            transform=transform)

    dataloader = DataLoader(dataset = dataset, batch_size=1,
                             shuffle=True,
                             num_workers=0)

    t = []
    img_shape = []
    for img, targets in dataloader:
        img, targets = img, targets
        t.append(targets)
        img_shape.append(img.shape)
    print(t)
    print(img_shape)

    # parser = argparse.ArgumentParser(description="Scene Graph Generation")
    # parser.add_argument("--config-file", default="configs/sgg_res101_step.yaml")
    # parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument("--session", type=int, default=0)
    # parser.add_argument("--resume", type=int, default=0)
    # parser.add_argument("--batchsize", type=int, default=0)
    # parser.add_argument("--inference", action='store_true')
    # parser.add_argument("--instance", type=int, default=-1)
    # parser.add_argument("--use_freq_prior", action='store_true')
    # parser.add_argument("--visualize", action='store_true')
    # parser.add_argument("--algorithm", type=str, default='sg_baseline')
    # parser.add_argument("--live", action='store_true')
    # args = parser.parse_args()
    #
    # cfg.merge_from_file(args.config_file)
    # d = build_data_loader(cfg)
    # import pdb; pdb.set_trace()
    # for data in d:
    #     img, tar = data

