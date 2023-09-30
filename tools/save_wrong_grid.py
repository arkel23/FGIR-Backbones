import os
import yaml
import argparse
from math import sqrt
from ast import literal_eval

import torch
import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange

from fgir_backbones.model_utils.build_model import build_model
from fgir_backbones.train_utils.misc_utils import set_random_seed
from fgir_backbones.data_utils.build_transform import build_transform
from heatmap import make_heatmaps


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


def read_df_filter_wrong_prob_class(args):
    # read ind preds, val/test.csv, and classid_classname.csv files
    ind_preds = pd.read_csv(args.preds_path)
    print('Original dataframe: ', len(ind_preds))

    # keep all ind_preds or filter by only wrong
    if args.wrong_preds_only:
        ind_preds = ind_preds[ind_preds['class_id'] != ind_preds['pred_id']]

        print('After filtering wrong only: ', len(ind_preds))

    # filter by confidently wrong
    if args.prob_th:
        ind_preds = ind_preds[ind_preds['prob'] >= args.prob_th]
        print('After filtering with confidence threshold: ', len(ind_preds))

    if args.class_id:
        ind_preds = ind_preds[ind_preds['class_id'] == args.class_id]
        print('After filtering by class id: ', len(ind_preds))

    return ind_preds


def update_ind_preds_class_names(args, ind_preds):
    fp = os.path.join(args.dataset_root_path, args.df_classid_classname)
    dic_classid_classname = pd.read_csv(fp, index_col='class_id')['class_name'].to_dict()
    args.num_classes = len(dic_classid_classname)

    fn_test = args.df_test if args.train_trainval else args.df_val
    folder_test = args.folder_test if args.train_trainval else args.folder_val
    fp = os.path.join(args.dataset_root_path, fn_test)
    df_test = pd.read_csv(fp)

    # update ind_preds with class_name and pred_class_name columns
    ind_preds['class_name'] = ind_preds['class_id'].apply(lambda x: dic_classid_classname[x])
    ind_preds['pred_class_name'] = ind_preds['pred_id'].apply(lambda x: dic_classid_classname[x])

    # update with full dir
    ind_preds['dir'] = df_test['dir'].apply(lambda x: os.path.join(args.dataset_root_path, folder_test, x))

    return ind_preds


def filter_manual_random_number(args, ind_preds):
    if args.select_manual_index:
        for i in ind_preds.index:
            class_gt = ind_preds.loc[i]['class_name']
            class_pred = ind_preds.loc[i]['pred_class_name']
            prob = ind_preds.loc[i]['prob']
            print(f'Index {i}, GT class: {class_gt}, predicted class: {class_pred} ({prob})')

        select_idx = 'Input indexes to select in format [i1, i2, i3]: '
        select_idx = literal_eval(select_idx)        
        ind_preds = ind_preds.loc[select_idx]

        print('After manually selecting indexes: ', len(ind_preds))

    if args.random:
        ind_preds = ind_preds.sample(frac=1)

    if args.num_images:
        ind_preds = ind_preds.iloc[:args.num_images]
        print('After filtering by first number of images: ', len(ind_preds))

    return ind_preds


def prepare_inference(args):
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if not args.resize_size:
        args.resize_size = int(args.image_size * 1.34)

    if not args.test_resize_size:
        args.test_resize_size = args.resize_size

    if args.ckpt_path:
        args_temp = vars(torch.load(args.ckpt_path, map_location=torch.device('cpu'))['config'])
        for k, v in args_temp.items():
            if k not in ('ckpt_path', 'results_dir'):
                if ((k == 'dataset_root_path' and getattr(args, k, None) is not None) or
                    (k == 'test_resize_size' and (
                        getattr(args, k) >= args_temp['image_size'] and
                        getattr(args, k) < v))):
                    pass
                # checkpoints with previous codebase (cal_model_name eg cal_vit_b16)
                elif (k == 'model_name' and 'vit' in v and 'cal' in v):
                    new_model_name = v.replace('cal_', '')
                    setattr(args, k, new_model_name)
                    args_temp['selector'] = 'cal'
                else:
                    setattr(args, k, v)

    set_random_seed(0, numpy=False)

    transform = build_transform(args=args, split='test')

    model = build_model(args)
    model.eval()

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    return transform, model


def prepare_img(fn, args, transform):
    # open img
    img = Image.open(fn).convert('RGB')
    # Preprocess image
    img = transform(img).unsqueeze(0).to(args.device)
    return img


def inference_single(args, model, img, dic_classid_classname=None, scores=None):

    with torch.no_grad():
        outputs = model(img, ret_inter=True)

    if isinstance(outputs, tuple) and len(outputs) == 2 and args.selector == 'cal':
        outputs, crops = outputs
    elif isinstance(outputs, tuple) and len(outputs) == 2:
        outputs, scores = outputs

    outputs = outputs.squeeze(0)
    for i, idx in enumerate(torch.topk(outputs, k=3).indices.tolist()):
        prob = torch.softmax(outputs, -1)[idx].item()
        if dic_classid_classname is not None:
            classname = dic_classid_classname[idx]
            out_text = '[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=classname, p=prob*100)
        else:
            out_text = '[{idx}] ({p:.2f}%)'.format(idx=idx, p=prob*100)

    masked_image = make_heatmaps(args, None, img, model, scores, False, save=False)

    return masked_image


def make_img_grid(args, ind_preds, transform=None, model=None):

    num_images = len(ind_preds)
    bh = int(sqrt(num_images))
    bw = int(num_images / bh)
    num_images = bh * bw

    img_list = []
    for i in range(num_images):
        fp = ind_preds.iloc[i]['dir']

        if args.vis_mask:
            img = prepare_img(fp, args, transform)
            img = inference_single(args, model, img)
            output_grid_name = f'{args.output_name}_{args.vis_mask}'
        else:
            img = Image.open(fp)
            img = img.resize((args.image_size, args.image_size))
            output_grid_name = args.output_name

        img = np.asarray(img)

        img_list.append(img)

    img_array = rearrange(img_list, '(bh bw) h w c -> (bh h) (bw w) c', bh=bh, bw=bw)
    img_array = Image.fromarray(img_array)

    img_array.save(os.path.join(args.results_dir, f'{output_grid_name}.png'))

    # save only certain columns
    ind_preds = ind_preds[['dir', 'class_id', 'pred_id', 'class_name', 'pred_class_name', 'prob']]
    fp_out = os.path.join(args.results_dir, f'{args.output_name}.csv')
    ind_preds.to_csv(fp_out, sep=',', header=True, index=False)

    print(f'Saved to {fp_out}', ind_preds.head())

    return 0


def main(transform=None, model=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--preds_path', type=str, required=True,
                        help='path to ind_preds.csv (results_train/dataset_model/ind_preds.csv)')

    parser.add_argument('--wrong_preds_only', action='store_false',
                        help='by default only saves wrong preds (if use flag saves all)')
    parser.add_argument('--prob_th', type=float, default=None,
                        help='filter confidently wrong')

    parser.add_argument('--class_id', type=int, default=None,
                        help='class id for class to visualize')
    parser.add_argument('--select_manual_index', action='store_true',
                        help='manually filter certain indexes')

    parser.add_argument('--random', action='store_true',
                        help='if used then returns random num_images rather than first')
    parser.add_argument('--num_images', type=int, default=16,
                        help='number of images to visualize in grid')

    parser.add_argument('--image_size', type=int, default=224)

    parser.add_argument('--output_name', default=None, type=str,
                        help='output file name')
    parser.add_argument('--results_dir', default='results_inference', type=str,
                        help='The directory where results will be stored')

    # inference with model
    parser.add_argument('--vis_mask', type=str, default=None,
                        help='which layer/mechanism to visualize: gradcam')
    parser.add_argument('--vis_mask_color', action='store_false',
                        help='if true then uses color heat map otherwise attn (white/black) map')
    parser.add_argument('--vis_th_topk', action='store_true',
                        help='for heatmaps use threshold of top-k largest')
    parser.add_argument('--vis_mask_sq', action='store_true',
                        help='square masks when applying heatmap')

    # model related
    parser.add_argument('--model_name', type=str, default='vit_b16')  # , choices=MODELS)
    parser.add_argument('--ckpt_path', type=str, default=None, help='path to custom pretrained ckpt')
    parser.add_argument('--ppt', action='store_true', help='patch prompt tuning')

    # augs
    parser.add_argument('--resize_size', type=int, default=None, help='resize_size before cropping')
    parser.add_argument('--test_resize_size', type=int, default=None, help='test resize_size before cropping')
    parser.add_argument('--custom_mean_std', action='store_true', help='custom mean/std')

    # dataset
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

    args = parser.parse_args()

    if args.cfg:
        config = yaml_config_hook(os.path.abspath(args.cfg))
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)

    if args.output_name is None:
        args.output_name = os.path.splitext(os.path.split(args.preds_path)[1])[0]

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    ind_preds = read_df_filter_wrong_prob_class(args)

    ind_preds = update_ind_preds_class_names(args, ind_preds)

    ind_preds = filter_manual_random_number(args, ind_preds)

    if args.vis_mask and args.ckpt_path:
        transform, model = prepare_inference(args)

    make_img_grid(args, ind_preds, transform, model)

    return 0


main()