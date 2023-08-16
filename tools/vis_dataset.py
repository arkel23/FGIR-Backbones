import os
import math

import torch
from torchvision.utils import save_image

from fgir_backbones.data_utils.build_dataloaders import build_dataloaders
from fgir_backbones.other_utils.build_args import parse_inference_args
from fgir_backbones.train_utils.misc_utils import set_random_seed
from fgir_backbones.train_utils.save_vis_images import inverse_normalize


def adjust_args_general(args):
    args.run_name = '{}_{}'.format(args.dataset_name, args.serial)
    args.results_dir = os.path.join(args.results_dir, args.run_name)
    os.makedirs(args.results_dir, exist_ok=True)
    return args


def vis_dataset(args):

    set_random_seed(args.seed, numpy=False)

    # dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(args)

    args = adjust_args_general(args)
    print(args)

    for split, loader in zip(['test', 'train'], [test_loader, train_loader]):
    # for split, loader in zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader]):
        for idx, (images, _) in enumerate(loader):
            images = inverse_normalize(images.data, norm_custom=args.custom_mean_std)
            fp = os.path.join(args.results_dir, f'{split}_{idx}.png')
            save_image(images, fp, nrow=int(math.sqrt(images.shape[0])))

            if idx % args.log_freq == 0:
                print(f'{split} ({idx} / {len(loader)}): {fp}')

            if not args.vis_dataset_all:
                break

    return 0


def main():
    args = parse_inference_args()

    vis_dataset(args)

    return 0


if __name__ == '__main__':
    main()