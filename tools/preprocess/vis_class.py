
import os
import yaml
import argparse
from math import sqrt
from ast import literal_eval

import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    instead of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            fp = cfg.get("defaults").get(d)
            cf = os.path.join(os.path.dirname(config_file), fp)
            with open(cf) as f:
                val = yaml.safe_load(f)
                print(val)
                cfg.update(val)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


def make_img_grid(args):

    if args.split_test:
        images_folder = args.folder_test
        df_file_name = args.df_test
    else:
        images_folder = args.folder_train
        df_file_name = args.df_trainval

    df = pd.read_csv(os.path.join(args.dataset_root_path, df_file_name))

    df = df[df['class_id'] == args.class_id]

    df = df.iloc[:args.num_images]

    if args.filter_random:
        df = df.sample(frac=1)

    num_images = len(df)
    bh = int(sqrt(num_images))
    bw = int(num_images / bh)
    num_images = bh * bw

    img_list = []
    for i in range(num_images):
        img_dir = df.iloc[i]['dir']
        full_img_dir = os.path.join(args.dataset_root_path, images_folder, img_dir)

        img = Image.open(full_img_dir)
        img = img.resize((args.image_size, args.image_size))

        img = np.asarray(img)

        img_list.append(img)

    img_array = rearrange(img_list, '(bh bw) h w c -> (bh h) (bw w) c', bh=bh, bw=bw)
    img_array = Image.fromarray(img_array)

    img_array.save(os.path.join(args.results_dir, f'{args.output_file}.png'))

    return 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', default=None, type=str, help='dataset name')
    parser.add_argument('--dataset_root_path', type=str, default=None,
                        help='the root directory for where the data/feature/label files are')
    # folders with images (can be same: those where it's all stored in 'data')
    parser.add_argument('--folder_train', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/train/')
    parser.add_argument('--folder_val', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/val/')
    parser.add_argument('--folder_test', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/test/')
    # df files with img_dir, class_id
    parser.add_argument('--df_train', type=str, default='train.csv',
                        help='the df csv with img_dirs, targets, def: train.csv')
    parser.add_argument('--df_trainval', type=str, default='train_val.csv',
                        help='the df csv with img_dirs, targets, def: train_val.csv')
    parser.add_argument('--df_val', type=str, default='val.csv',
                        help='the df csv with img_dirs, targets, def: val.csv')
    parser.add_argument('--df_test', type=str, default='test.csv',
                        help='the df csv with img_dirs, targets, root/test.csv')
    parser.add_argument('--df_classid_classname', type=str, default='classid_classname.csv',
                        help='the df csv with classnames and class ids, root/classid_classname.csv')

    parser.add_argument('--train_trainval', action='store_false',
                        help='when true uses trainval for train and evaluates on test \
                        otherwise use train for train and evaluates on val')

    parser.add_argument("--cfg", type=str,
                        help="If using it overwrites args and reads yaml file in given path")

    parser.add_argument('--split_test', action='store_true',
                        help='by def visualizes train split, if use this flag then vis test')
    parser.add_argument('--num_images', type=int, default=16,
                        help='number of images to visualize in grid')
    parser.add_argument('--filter_random', action='store_true',
                        help='if used then returns random num_images rather than first')
    parser.add_argument('--class_id', type=int, default=0,
                        help='class id for class to visualize')

    parser.add_argument('--image_size', type=int, default=224)

    parser.add_argument('--output_file', default=None, type=str,
                        help='output file name')
    parser.add_argument('--results_dir', default='results_inference', type=str,
                        help='The directory where results will be stored')

    args = parser.parse_args()

    if args.cfg:
        config = yaml_config_hook(os.path.abspath(args.cfg))
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)

    if args.output_file is None:
        split = 'test' if args.split_test else 'train'
        args.output_file = f'{args.dataset_name}_{split}_{args.class_id}'

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    make_img_grid(args)

    return 0


main()

