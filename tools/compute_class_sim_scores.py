import os
import json
import glob
import random
import argparse

import timm
import numpy as np
import pandas as pd
from PIL import Image
from einops import reduce
from einops.layers.torch import Rearrange, Reduce
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except:
    from PIL.Image import BICUBIC as BICUBIC


IGNORE = ('ckpt_path', 'transfer_learning', 'dataset_root_path', 'folder_test')


class TIMMNets(nn.Module):
    def __init__(self, args):
        super(TIMMNets, self).__init__()
        # init default config

        if 'vgg' in args.model_name:
            self.model = timm.create_model(args.model_name, pretrained=args.pretrained,
                                           num_classes=0, global_pool='')
        elif any(model in args.model_name for model in ['resnet', 'convnext']):
            self.model = timm.create_model(
                args.model_name, pretrained=args.pretrained, num_classes=0,
                drop_path_rate=args.sd, global_pool='')
        else:
            self.model = timm.create_model(
                args.model_name, pretrained=args.pretrained, num_classes=0,
                img_size=args.image_size, drop_path_rate=args.sd, global_pool='')
        # to return intermediate features, features_only=True (only works for some)

        out_features, s, self.rearrange = self.get_out_features(args.image_size)

        self.pool = Reduce('b s d -> b d', 'mean')

        self.head = nn.Linear(out_features, args.num_classes)

    @torch.no_grad()
    def get_out_features(self, image_size):
        x = torch.rand(2, 3, image_size, image_size)
        x = self.model(x)

        if len(x.shape) == 2:
            b, d = x.shape
            s = 1
            rearrange = Rearrange('b d -> b 1 d')
        elif len(x.shape) == 3:
            b, s, d = x.shape
            rearrange = nn.Identity()
        elif len(x.shape) == 4:
            b, d, h, w = x.shape
            s = h * w
            rearrange = Rearrange('b d h w -> b (h w) d')

        return d, s, rearrange

    def forward(self, images, targets=None, ret_dist=False):
        features = self.model(images)
        features = self.rearrange(features)

        out = self.pool(features)
        out = self.head(out)

        if ret_dist:
            return out, features
        return out


def set_random_seed(seed=0, numpy=True):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if numpy:
        np.random.seed(seed)
    return 0


def build_transform(args):
    resize_size = int(args.image_size / 0.875)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    t = []

    t.append(
        transforms.Resize(
            (resize_size, resize_size), interpolation=BICUBIC))
    t.append(transforms.CenterCrop(args.image_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=mean, std=std))

    transform = transforms.Compose(t)
    return transform


def build_model(args):
    if args.model_name in timm.list_models(pretrained=True):
        model = TIMMNets(args)

    if args.ckpt_path:
        state_dict = torch.load(
            args.ckpt_path, map_location=torch.device('cpu'))['model']
        expected_missing_keys = []
        if args.transfer_learning:
            # modifications to load partial state dict
            if ('model.head.weight' in state_dict):
                expected_missing_keys += ['model.head.weight', 'model.head.bias']
            for key in expected_missing_keys:
                state_dict.pop(key)
        ret = model.load_state_dict(state_dict, strict=False)
        print('''Missing keys when loading pretrained weights: {}
              Expected missing keys: {}'''.format(ret.missing_keys, expected_missing_keys))
        print('Unexpected keys when loading pretrained weights: {}'.format(
            ret.unexpected_keys))
        print('Loaded from custom checkpoint.')

    model.to(args.device)

    print(f'Initialized classifier: {args.model_name}')
    return model


def parse_inference_args():
    parser = argparse.ArgumentParser('Arguments for inference')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model_name', type=str, default='convnext_base_in22k')  # , choices=MODELS)
    parser.add_argument('--pretrained', action='store_true', help='pretrained model on imagenet')
    parser.add_argument('--ckpt_path', type=str, default=None, help='path to custom pretrained ckpt')
    parser.add_argument('--transfer_learning', action='store_true',
                        help='not load fc layer when using custom ckpt')
    parser.add_argument('--image_size', type=int, default=224, help='image_size')
    parser.add_argument('--images_path', type=str, default='samples',
                        help='path to folder (with images) or image')
    parser.add_argument('--results_inference', type=str, default='results_inference',
                        help='path to folder to save result crops')
    parser.add_argument('--dataset_root_path', type=str, default=None,
                        help='the root directory for where the data/feature/label files are')
    parser.add_argument('--json_test', type=str, default='movienet_4_cropped_val.json',
                        help='the json with the source images to compare similarity')
    parser.add_argument('--folder_test', type=str, default='images',
                        help='the directory where images are stored, ex: dataset_root_path/test/')
    args = parser.parse_args()
    return args


def set_src_dic(args):

    with open(os.path.join(args.dataset_root_path, args.json_test), 'rt') as f:
        json_test = json.load(f)

    src_dic = {}

    seen_ids = []

    for row in json_test:
        movie_id = int(row['film_label'])
        if movie_id == 0:
            movie_name = 'oz'
        elif movie_id == 96:
            movie_name = 'stalker'
        elif movie_id == 113:
            movie_name = 'bladerunner'
        elif movie_id == 923:
            movie_name = 'moonrise'

        if movie_id not in seen_ids:
            src_idx = 0
            seen_ids.append(movie_id)
            src_dic.update({movie_name: {}})

        src_dic[movie_name].update({src_idx: row['style_path']})

        src_idx += 1

    return src_dic


def setup_env():
    args = parse_inference_args()

    set_random_seed(args.seed, numpy=False)

    if args.ckpt_path:
        args_temp = vars(torch.load(args.ckpt_path, map_location=torch.device('cpu'))['config'])
        for k, v in args_temp.items():
            if k not in IGNORE:
                setattr(args, k, v)
    else:
        args.num_classes = 1000

    model = build_model(args)
    model.eval()

    transform = build_transform(args)

    if args.results_inference:
        args.results_dir = os.path.join(args.results_inference, args.model_name)
        os.makedirs(args.results_dir, exist_ok=True)

    src_dic = set_src_dic(args)

    return args, model, transform, src_dic


def search_images(args):
    # if path is a file
    if os.path.isfile(args.images_path):
        return [args.images_path]
    # else if directory
    # the tuple of file types
    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    files_all = []
    for file_type in types:
        # files_all is the list of files
        path = os.path.join(args.images_path, '**', file_type)
        files_curr_type = glob.glob(path, recursive=True)
        files_all.extend(files_curr_type)

        print(file_type, len(files_curr_type))

    print('Total image files pre-filtering', len(files_all))
    return files_all


def prepare_img(fn, args, transform):
    # open img
    img = Image.open(fn).convert('RGB')
    # Preprocess image
    img = transform(img).unsqueeze(0).to(args.device)
    return img


def inference_single(model, img, gt_idx=None):

    with torch.no_grad():
        outputs, features = model(img, ret_dist=True)
        features = reduce(features, '1 s d -> d', 'mean')
    #1, k -> k = 4
    outputs = torch.softmax(outputs.squeeze(0), -1)

    if gt_idx is not None:
        prob = outputs[gt_idx].item()
    else:
        for idx in torch.topk(outputs, k=1).indices.tolist():
            prob = outputs[idx].item()

    return prob, features


def compute_sim(features_src, features_gen):
    sim = F.cosine_similarity(features_src, features_gen, dim=-1).item()
    return sim


def main():

    args, model, transform, src_dic = setup_env()

    # requires images_path
    files_all = search_images(args)

    results_list = []

    print(src_dic, len(files_all))

    for i, fp_gen in enumerate(files_all):
        # print(file)
        img = prepare_img(fp_gen, args, transform)

        prob, features_gen = inference_single(model, img)

        fp_rest, fn = os.path.split(os.path.abspath(fp_gen))
        fp_rest, iter = os.path.split(fp_rest)
        movie_name = os.path.split(fp_rest)[-1].split('_')[-1]

        idx_src = int(fn.split('_')[1])

        fp_src = os.path.join(args.dataset_root_path, args.folder_test, src_dic[movie_name][idx_src])

        img_src = prepare_img(fp_src, args, transform)
        _, features_src = inference_single(model, img_src)

        sim = compute_sim(features_src, features_gen)

        folder_name = os.path.split(os.path.normpath(fp_gen))[0]
        results_list.append([folder_name, fp_gen, prob, sim])

        if i % 100 == 0:
            print(f'{i}/{len(files_all)}: {results_list[i]}')

    print('Finished: ', len(results_list), results_list[0], results_list[-1])

    return 0


if __name__ == '__main__':
    main()
