import os
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile

import torch
import torch.utils.data as data
from torchvision import datasets


ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_set(args, split, transform=None):
    if args.dataset_name == 'cifar10':
        ds = datasets.CIFAR10(root=args.dataset_root_path,
                              train=True if split == 'train' else False,
                              transform=transform, download=True)
        ds.num_classes = 10
    elif args.dataset_name == 'cifar100':
        ds = datasets.CIFAR100(root=args.dataset_root_path,
                               train=True if split == 'train' else False,
                               transform=transform, download=True)
        ds.num_classes = 100
    else:
        if args.cal_cm and split == 'train':
            ds = DatasetImgTargetTwoSameClass(args, split=split, transform=transform)
        else:
            ds = DatasetImgTarget(args, split=split, transform=transform)
        args.num_classes = ds.num_classes

    setattr(args, f'num_images_{split}', ds.__len__())
    print(f"{args.dataset_name} {split} split. N={ds.__len__()}, K={ds.num_classes}.")
    return ds


class DatasetImgTarget(data.Dataset):
    def __init__(self, args, split, transform=None):
        self.root = os.path.abspath(args.dataset_root_path)
        self.transform = transform

        if split == 'train':
            if args.train_trainval:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_trainval
            else:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_train
        elif split == 'val':
            if args.train_trainval:
                self.images_folder = args.folder_test
                self.df_file_name = args.df_test
            else:
                self.images_folder = args.folder_val
                self.df_file_name = args.df_val
        else:
            self.images_folder = args.folder_test
            self.df_file_name = args.df_test

        assert os.path.isfile(os.path.join(self.root, self.df_file_name)), \
            f'{os.path.join(self.root, self.df_file_name)} is not a file.'

        self.df = pd.read_csv(os.path.join(self.root, self.df_file_name), sep=',')
        self.targets = self.df['class_id'].to_numpy()
        self.data = self.df['dir'].to_numpy()

        self.num_classes = len(np.unique(self.targets))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir, target = self.data[idx], self.targets[idx]
        full_img_dir = os.path.join(self.root, self.images_folder, img_dir)
        img = Image.open(full_img_dir)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.targets)


class DatasetImgTargetTwoSameClass(DatasetImgTarget):
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir, target = self.data[idx], self.targets[idx]
        full_img_dir = os.path.join(self.root, self.images_folder, img_dir)
        img = Image.open(full_img_dir)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Find indices of images with the same class as the current image
        same_class_indices = np.where(self.targets == target)[0]
        # Remove the current index from the list
        same_class_indices = same_class_indices[same_class_indices != idx]

        # Randomly select one index from the list of indices with the same class
        if len(same_class_indices) > 0:
            same_class_idx = np.random.choice(same_class_indices)
        # If there are no other images with the same class, return the current image twice
        else:
            same_class_idx = idx

        # load same class image
        same_class_img_dir = self.data[same_class_idx]
        full_same_class_img_dir = os.path.join(self.root, self.images_folder, same_class_img_dir)
        same_class_img = Image.open(full_same_class_img_dir)
        if same_class_img.mode != 'RGB':
            same_class_img = same_class_img.convert('RGB')

        target = np.repeat(target, 2)

        if self.transform:
            img = self.transform(img)
            same_class_img = self.transform(same_class_img)
            imgs = torch.stack([img, same_class_img], dim=0)
            return imgs, target

        return img, same_class_img, target
