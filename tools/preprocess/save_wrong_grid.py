import os
import argparse
from math import sqrt
from ast import literal_eval

import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange


def read_filter_df(args):
    ind_preds = pd.read_csv(args.preds_path)

    # filter by confidently wrong
    if args.filter_prob_th:
        print('Before filtering with confidence threshold: ', len(ind_preds))

        ind_preds = ind_preds[ind_preds['prob'] >= args.filter_prob_th]

        print('After filtering with confidence threshold: ', len(ind_preds))

    if args.filter_manual_index:
        for i in ind_preds.index:
            class_gt = ind_preds.loc[i]['class_name']
            class_pred = ind_preds.loc[i]['pred_class_name']
            prob = ind_preds.loc[i]['prob']
            print(f'Index {i}, GT class: {class_gt}, predicted class: {class_pred} ({prob})')

        filt_idx = 'Input indexes to filter in format [i1, i2, i3]: '
        filt_idx = literal_eval(filt_idx)        
        ind_preds = ind_preds.loc[~filt_idx]

        print('After manually filtering indexes: ', len(ind_preds))

    if args.filter_random:
        ind_preds = ind_preds.sample(frac=1)

    if args.filter_num_images:
        ind_preds = ind_preds.iloc[:args.filter_num_images]
        print('After filtering by first number of images: ', len(ind_preds))

    for i in range(len(ind_preds)):
        class_gt = ind_preds.iloc[i]['class_name']
        class_pred = ind_preds.iloc[i]['pred_class_name']
        prob = ind_preds.iloc[i]['prob']
        dir = ind_preds.iloc[i]['dir']
        print(f'Index {i}, GT class: {class_gt}, predicted class: {class_pred} ({prob})\n{dir}')

    return ind_preds


def make_img_grid(args, ind_preds):

    num_images = len(ind_preds)
    bh = int(sqrt(num_images))
    bw = int(num_images / bh)
    num_images = bh * bw

    img_list = []
    for i in range(num_images):
        fp = ind_preds.iloc[i]['dir']

        img = Image.open(fp)
        img = img.resize((args.image_size, args.image_size))

        img = np.asarray(img)

        img_list.append(img)

    img_array = rearrange(img_list, '(bh bw) h w c -> (bh h) (bw w) c', bh=bh, bw=bw)
    img_array = Image.fromarray(img_array)

    img_array.save(os.path.join(args.results_dir, f'{args.output_file}.png'))

    return 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--preds_path', type=str, required=True,
                        help='path to ind_preds.csv (results_train/dataset_model/ind_preds.csv)')

    parser.add_argument('--filter_prob_th', type=float, default=90,
                        help='filter confidently wrong')
    parser.add_argument('--filter_manual_index', action='store_true',
                        help='manually filter certain indexes')

    parser.add_argument('--filter_num_images', type=int, default=36,
                        help='number of images to visualize in grid')
    parser.add_argument('--filter_random', action='store_true',
                        help='if used then returns random filter_num_images rather than first')

    parser.add_argument('--image_size', type=int, default=224)

    parser.add_argument('--output_file', default=None, type=str,
                        help='output file name')
    parser.add_argument('--results_dir', default='results_inference', type=str,
                        help='The directory where results will be stored')

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = os.path.splitext(os.path.split(args.preds_path)[1])[0]

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    ind_preds = read_filter_df(args)

    make_img_grid(args, ind_preds)

    return 0


main()