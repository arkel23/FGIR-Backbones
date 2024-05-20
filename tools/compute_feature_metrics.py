import os
import math

import wandb
import numpy as np
import torch
from torchvision.utils import save_image

from fgir_backbones.data_utils.build_dataloaders import build_dataloaders
from fgir_backbones.other_utils.build_args import parse_train_args
from fgir_backbones.model_utils.build_model import build_model
from fgir_backbones.train_utils.misc_utils import set_random_seed
from fgir_backbones.train_utils.save_vis_images import inverse_normalize

import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from warnings import warn
from typing import List, Dict
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from einops import reduce, rearrange


EPS = 1e-5


def adjust_args_general(args):
    freeze = '_fz' if args.freeze_backbone else ''
    selector = f'_{args.selector}' if args.selector else ''
    classifier = f'_{args.classifier}' if args.classifier in ('blp', 'iblp') else ''

    args.run_name = '{}_{}{}{}{}_{}'.format(
        args.dataset_name, args.model_name, classifier, selector, freeze, args.serial
    )

    args.results_dir = os.path.join(args.results_dir, args.run_name)
    os.makedirs(args.results_dir, exist_ok=True)
    return args


def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


class Distances:
    def __init__(self,
                 model1: nn.Module,
                 model1_name: str = None,
                 model1_layers: List[str] = None,
                 device: str ='cpu',
                 image_size: int = 224,
                 out_size: int = 4,
                 debugging: bool = False):
        """

        :param model1: (nn.Module) Neural Network 1
        :param model1_name: (str) Name of model 1
        :param model1_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model1 = model1

        self.device = device

        self.model1_info = {}

        if model1_name is None:
            self.model1_info['Name'] = model1.__repr__().split('(')[0]
        else:
            self.model1_info['Name'] = model1_name

        self.model1_info['Layers'] = []

        self.model1_features = {}

        if len(list(model1.modules())) > 150 and model1_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model1_layers' parameter. Your CPU/GPU will thank you :)")

        self.model1_layers = model1_layers

        self._insert_hooks()
        self.model1 = self.model1.to(self.device)

        self.model1.eval()

        self._check_shape(image_size)

        self.pool = nn.AdaptiveAvgPool2d((out_size, out_size)).to(self.device)

        self.debugging = debugging

        print(self.model1_info)

    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model1":
            self.model1_features[name] = out
        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model1.named_modules():
            if self.model1_layers is not None:
                if name in self.model1_layers:
                    self.model1_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model1", name))
            else:
                self.model1_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model1", name))

    def _check_shape(self, image_size):
        with torch.no_grad():
            x = torch.rand(2, 3, image_size, image_size).to(self.device)
            _ = self.model1(x)

            # -1 in certain cases corresponds to classification layer
            last = self.model1_info['Layers'][-2]
            feat_out = self.model1_features[last]

            if len(feat_out.shape) == 4:
                b, c, h, w = feat_out.shape
                if h == w:
                    self.bchw = True
                    h = feat_out.shape[-1]
                else:
                    self.bchw = False
                    h = feat_out.shape[1]
            elif len(feat_out.shape) == 3:
                h = int(feat_out.shape[1] ** 0.5)
                self.cls = False if h ** 2 == feat_out.shape[1] else True
            else:
                pass

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        result = ((1 / (N * (N - 3))) * result).item()
        return result

    def _pool_features(self, feat, pool=True):
        if hasattr(self, 'pool') and pool:
            if len(feat.shape) == 4:
                if not self.bchw:
                    feat = rearrange(feat, 'b h w c -> b c h w')
                pooled = self.pool(feat)
                pooled = rearrange(pooled, 'b c h w -> (b h w) c')
            elif len(feat.shape) == 3:
                h = int(feat.shape[1] ** 0.5)
                if self.cls:
                    x_cls, x_others = torch.split(feat, [1, int(h**2)], dim=1)
                    x_others = rearrange(x_others, 'b (h w) d -> b d h w', h=h)
                    x_others = self.pool(x_others)
                    x_others = rearrange(x_others, 'b d h w -> b (h w) d')
                    pooled = torch.cat([x_cls, x_others], dim=1)
                    pooled = rearrange(pooled, 'b s d -> (b s) d')
                else:
                    pooled = rearrange(feat, 'b (h w) d -> b d h w', h=h)
                    pooled = self.pool(pooled)
                    pooled = rearrange(pooled, 'b c h w -> (b h w) c')                    
        else:
            pooled = feat.flatten(1)
 
        return pooled

    def compare(self,
                dataloader1: DataLoader) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        """

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))

        self.hsic_matrix = torch.zeros(N, N, 3)
        self.hsic_matrix_pooled = torch.zeros(N, N, 3)
        self.dist_cum = torch.zeros(N, device=self.device)
        self.dist_cum_norm = torch.zeros(N, device=self.device)
        self.l2_norm = torch.zeros(N, device=self.device)

        num_batches = min(len(dataloader1), len(dataloader1))

        for (x1, *_) in tqdm(dataloader1, desc="| Comparing features |", total=num_batches):

            self.model1_features = {}
            x1 = x1.to(self.device)
            _ = self.model1(x1)

            for i, (name1, feat1) in enumerate(self.model1_features.items()):

                X = self._pool_features(feat1, pool=False)
                X_pooled = self._pool_features(feat1, pool=True)

                # frobenius norm
                self.l2_norm[i] += torch.norm(X_pooled, p='fro', dim=-1).mean() / num_batches

                dist = torch.cdist(X_pooled, X_pooled, p=2.0)

                dist_avg = (torch.sum(dist) / torch.nonzero(dist).size(0))
                self.dist_cum[i] += dist_avg / num_batches

                dist = (dist - dist.min()) / (dist.max() - dist.min())
                dist_avg_norm = (torch.sum(dist) / torch.nonzero(dist).size(0))
                self.dist_cum_norm[i] += dist_avg_norm / num_batches

                K = X @ X.t()
                K.fill_diagonal_(0.0)
                try:
                    self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches
                except:
                    self.hsic_matrix[i, :, 0] += 0

                K_pooled = X_pooled @ X_pooled.t()
                K_pooled.fill_diagonal_(0.0)
                self.hsic_matrix_pooled[i, :, 0] += self._HSIC(K_pooled, K_pooled) / num_batches

                for j, (name2, feat2) in enumerate(self.model1_features.items()):
                    Y = self._pool_features(feat2, pool=False)
                    Y_pooled = self._pool_features(feat2, pool=True)

                    L = Y @ Y.t()
                    L.fill_diagonal_(0)

                    L_pooled = Y_pooled @ Y_pooled.t()
                    L_pooled.fill_diagonal_(0)

                    assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"
                    assert K_pooled.shape == L_pooled.shape, f"Feature shape mistach! {K_pooled.shape}, {L_pooled.shape}"

                    try:
                        self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                        self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches
                    except:
                        self.hsic_matrix[i, j, 1] += 0
                        self.hsic_matrix[i, j, 2] += 0

                    self.hsic_matrix_pooled[i, j, 1] += self._HSIC(K_pooled, L_pooled) / num_batches
                    self.hsic_matrix_pooled[i, j, 2] += self._HSIC(L_pooled, L_pooled) / num_batches

        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                        self.hsic_matrix[:, :, 2].sqrt())
        self.hsic_matrix_pooled = self.hsic_matrix_pooled[:, :, 1] / (self.hsic_matrix_pooled[:, :, 0].sqrt() *
                                                        self.hsic_matrix_pooled[:, :, 2].sqrt())


    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": self.model1_info['Name'],
            'l2_norm': self.l2_norm,
            "CKA": self.hsic_matrix,
            "CKA_pooled": self.hsic_matrix_pooled,
            "dist": self.dist_cum,
            "dist_norm": self.dist_cum_norm,
            "model1_layers": self.model1_info['Layers'],
            "dataset1_name": self.model1_info['Dataset'],

        }

    def plot_cka(self,
                 save_path: str = None,
                 title: str = None,
                 show: bool = False,
                 pooled: bool = False):
        fig, ax = plt.subplots()
        if pooled:
            im = ax.imshow(self.hsic_matrix_pooled, origin='lower', cmap='magma')
        else:
            im = ax.imshow(self.hsic_matrix, origin='lower', cmap='magma')
        ax.set_xlabel(f"Layers {self.model1_info['Name']}", fontsize=15)
        ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)

        if title is not None:
            ax.set_title(f"{title}", fontsize=18)
        else:
            ax.set_title(f"{self.model1_info['Name']}", fontsize=18)

        add_colorbar(im)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        if not self.debugging:
            fn = os.path.splitext(os.path.split(save_path)[-1])[0]
            wandb.log({fn: wandb.Image(fig)})

        if show:
            plt.show()

    def plot_norms(self,
                   save_path: str = None,
                   title: str = None,
                   show: bool = False):
        fig, ax = plt.subplots()

        labels = range(self.l2_norm.shape[0])

        ax.bar(labels, self.l2_norm.cpu())
        ax.set_xlabel("Layer", fontsize=15)
        ax.set_ylabel("L2-Norm", fontsize=15)

        if title is not None:
            ax.set_title(f"{title}", fontsize=18)
        else:
            ax.set_title(f"L2-Norm Distribution As Function of Layer", fontsize=18)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        if not self.debugging:
            fn = os.path.splitext(os.path.split(save_path)[-1])[0]
            wandb.log({fn: wandb.Image(fig)})

        if show:
            plt.show()


def calc_cka(results, pooled=False):
    name = 'CKA_pooled' if pooled else 'CKA'
    cka_first = torch.mean(results[name][0, 1:].flatten()).item()

    num_layers = results[name].shape[0]
    mask = torch.triu(torch.ones(num_layers, num_layers))
    mask -= torch.eye(num_layers)
    masked = results[name] * mask
    cka_avg = (torch.sum(masked) / torch.nonzero(masked).size(0)).item()

    cka_last = torch.mean(results[name][-1, :-1]).item()

    return cka_first, cka_last, cka_avg


def calc_distances(results):
    dist_first = results['dist'][0].item()
    dist_first_norm = results['dist_norm'][0].item()
    dist_last = results['dist'][-1].item()
    dist_last_norm = results['dist_norm'][-1].item()
    dist_avg = torch.mean(results['dist']).item()
    dist_avg_norm = torch.mean(results['dist_norm']).item()

    return dist_first, dist_first_norm, dist_last, dist_last_norm, dist_avg, dist_avg_norm


def calc_l2_norm(results):
    l2_norm_first = results['l2_norm'][0].item()
    l2_norm_last = results['l2_norm'][-1].item()
    l2_norm_avg = torch.mean(results['l2_norm']).item()
    return l2_norm_first, l2_norm_last, l2_norm_avg


def compute_cka_dataset(args):

    set_random_seed(args.seed, numpy=True)

    # dataloaders
    args.shuffle_test = True
    train_loader, val_loader, test_loader = build_dataloaders(args)

    model = build_model(args)

    args = adjust_args_general(args)
    results_fp = os.path.join(args.results_dir, 'feature_metrics.csv')

    if not args.debugging:
        wandb.init(config=args, project=args.project_name, entity=args.entity)
        wandb.run.name = args.run_name

    if args.model_name == 'vgg19_bn':
        layers = ['model.features.11', 'model.features.24', 'model.features.37', 'model.features.50']
    elif args.model_name == 'resnet101': 
        layers = ['model.layer1.2.bn3', 'model.layer2.3.bn3', 'model.layer3.22.bn3', 'model.layer4.2.bn3']
    elif 'vit_b' in args.model_name:
        layers = ['model.encoder.blocks.2.norm2', 'model.encoder.blocks.5.norm2', 'model.encoder.blocks.8.norm2', 'model.encoder.blocks.11.norm2']
    elif 'beitv2_base_patch16_224_in22k' in args.model_name or 'deit3_base_patch16_224' in args.model_name:
        layers = ['model.blocks.2.norm2', 'model.blocks.5.norm2', 'model.blocks.8.norm2', 'model.blocks.11.norm2']
    elif 'deit3_large_patch16_224' in args.model_name:
        layers = ['model.blocks.5.norm2', 'model.blocks.11.norm2', 'model.blocks.17.norm2', 'model.blocks.23.norm2']
    elif args.model_name == 'van_b3':
        layers = ['model.norm1', 'model.norm2', 'model.norm3', 'model.norm4']
    elif 'convnext' in args.model_name:
        layers = ['model.stages.0.blocks.2.norm', 'model.stages.1.blocks.2.norm', 'model.stages.2.blocks.26.norm', 'model.stages.3.blocks.2.norm']
    elif 'convnext_base' in args.model_name:
        layers = ['model.stages.0.blocks.2.norm', 'model.stages.1.blocks.2.norm', 'model.stages.2.blocks.26.norm', 'model.stages.3.blocks.2.norm']
    elif 'swin' in args.model_name:
        layers = ['model.layers.0.blocks.1.norm2', 'model.layers.1.blocks.1.norm2', 'model.layers.2.blocks.17.norm2', 'model.layers.3.blocks.1.norm2']
    elif 'swin_base' in args.model_name:
        layers = ['model.layers.0.blocks.1.norm2', 'model.layers.1.blocks.1.norm2', 'model.layers.2.blocks.17.norm2', 'model.layers.3.blocks.1.norm2']
    elif 'resnetv2_101' in args.model_name:
        layers = ['model.stages.0.blocks.2.norm3', 'model.stages.1.blocks.3.norm3', 'model.stages.2.blocks.22.norm3', 'model.stages.3.blocks.2.norm3']
    else:
        raise NotImplementedError

    if args.selector == 'cal' and any([kw in args.model_name for kw in ('vit', 'deit', 'beit', 'swin', 'van')]):
        layers = [layer.replace('model.', 'model.encoder.0.') for layer in layers]
    elif args.selector == 'cal':
        layers = [layer.replace('model.', 'model.encoder.') for layer in layers]

    distances = Distances(model, args.model_name, layers, args.device,
                          args.image_size, debugging=args.debugging)

    with torch.no_grad():
        distances.compare(train_loader)

        results_train = distances.export()
        distances.plot_cka(os.path.join(args.results_dir, 'cka_train.png'))
        distances.plot_cka(os.path.join(args.results_dir, 'cka_pooled_train.png'), pooled=True)
        distances.plot_norms(os.path.join(args.results_dir, 'norms_train.png'))

        cka_first_train, cka_last_train, cka_avg_train = calc_cka(results_train)
        cka_pooled_first_train, cka_pooled_last_train, cka_pooled_avg_train = calc_cka(results_train, pooled=True)

        (dist_first_train, dist_first_norm_train, dist_last_train, dist_last_norm_train, 
         dist_avg_train, dist_avg_norm_train) = calc_distances(results_train)

        l2_norm_first_train, l2_norm_last_train, l2_norm_avg_train = calc_l2_norm(results_train)


        distances.compare(test_loader)

        results_test = distances.export()
        distances.plot_cka(os.path.join(args.results_dir, 'cka_test.png'))
        distances.plot_cka(os.path.join(args.results_dir, 'cka_pooled_test.png'), pooled=True)
        distances.plot_norms(os.path.join(args.results_dir, 'norms_test.png'))

        cka_first_test, cka_last_test, cka_avg_test = calc_cka(results_test)
        cka_pooled_first_test, cka_pooled_last_test, cka_pooled_avg_test = calc_cka(results_test, pooled=True)

        (dist_first_test, dist_first_norm_test, dist_last_test, dist_last_norm_test
         , dist_avg_test, dist_avg_norm_test) = calc_distances(results_test)

        l2_norm_first_test, l2_norm_last_test, l2_norm_avg_test = calc_l2_norm(results_test)


    title = '''dataset_name,model,setting,cka_avg_train,cka_first_train,cka_last_train,cka_pooled_avg_train,cka_pooled_first_train,cka_pooled_last_train,dist_avg_train,dist_avg_norm_train,dist_first_train,dist_first_norm_train,dist_last_train,dist_last_norm_train,l2_norm_avg_train,l2_norm_first_train,l2_norm_first_train,cka_avg_test,cka_first_test,cka_last_test,cka_pooled_avg_test,cka_pooled_first_test,cka_pooled_last_test,dist_avg_test,dist_avg_norm_test,dist_first_test,dist_first_norm_test,dist_last_test,dist_last_norm_test,l2_norm_avg_test,l2_norm_first_test,l2_norm_last_test\n'''
    if args.selector == 'cal':
        setting = 'cal'
    elif args.freeze_backbone:
        setting = 'fz'
    else:
        setting = 'ft'
    setting = f'{setting}_{args.image_size}'

    values = f'''{args.dataset_name},{args.model_name},{setting},{cka_avg_train},{cka_first_train},{cka_last_train},{dist_avg_train},{dist_avg_norm_train},{dist_first_train},{dist_first_norm_train},{dist_last_train},{dist_last_norm_train},{l2_norm_avg_train},{l2_norm_first_train},{l2_norm_last_train},{cka_avg_test},{cka_first_test},{cka_last_test},{dist_avg_test},{dist_avg_norm_test},{dist_first_test},{dist_first_norm_test},{dist_last_test},{dist_last_norm_test},{l2_norm_avg_test},{l2_norm_first_test},{l2_norm_last_test}\n'''
    print(title, values)

    values = f'''{args.dataset_name},{args.model_name},{setting},{cka_avg_train},{cka_first_train},{cka_last_train},{cka_pooled_avg_train},{cka_pooled_first_train},{cka_pooled_last_train},{dist_avg_train},{dist_avg_norm_train},{dist_first_train},{dist_first_norm_train},{dist_last_train},{dist_last_norm_train},{l2_norm_avg_train},{l2_norm_first_train},{l2_norm_last_train},{cka_avg_test},{cka_first_test},{cka_last_test},{cka_pooled_avg_test},{cka_pooled_first_test},{cka_pooled_last_test},{dist_avg_test},{dist_avg_norm_test},{dist_first_test},{dist_first_norm_test},{dist_last_test},{dist_last_norm_test},{l2_norm_avg_test},{l2_norm_first_test},{l2_norm_last_test}\n'''
    print(title, values)

    with open(results_fp, 'w') as file:
        file.write(title)
        file.write(values)

    if not args.debugging:
        log_dic = {
            'setting': setting,
            'cka_avg_train': cka_avg_train,
            'cka_first_train': cka_first_train,
            'cka_last_train': cka_last_train,
            'cka_pooled_avg_train': cka_pooled_avg_train,
            'cka_pooled_first_train': cka_pooled_first_train,
            'cka_pooled_last_train': cka_pooled_last_train,
            'dist_avg_train': dist_avg_train,
            'dist_avg_norm_train': dist_avg_norm_train,
            'dist_first_train': dist_first_train,
            'dist_first_norm_train': dist_first_norm_train,
            'dist_last_train': dist_last_train,
            'dist_last_norm_train': dist_last_norm_train,
            'l2_norm_avg_train': l2_norm_avg_train,
            'l2_norm_first_train': l2_norm_first_train,
            'l2_norm_last_train': l2_norm_last_train,
            'cka_avg_test': cka_avg_test,
            'cka_first_test': cka_first_test,
            'cka_last_test': cka_last_test,
            'cka_pooled_avg_test': cka_pooled_avg_test,
            'cka_pooled_first_test': cka_pooled_first_test,
            'cka_pooled_last_test': cka_pooled_last_test,
            'dist_avg_test': dist_avg_test,
            'dist_avg_norm_test': dist_avg_norm_test,
            'dist_first_test': dist_first_test,
            'dist_first_norm_test': dist_first_norm_test,
            'dist_last_test': dist_last_test,
            'dist_last_norm_test': dist_last_norm_test,
            'l2_norm_avg_test': l2_norm_avg_test,
            'l2_norm_first_test': l2_norm_first_test,
            'l2_norm_last_test': l2_norm_last_test,
        }
        wandb.log(log_dic)
        wandb.finish()

    return 0


def main():
    args = parse_train_args()

    compute_cka_dataset(args)

    return 0


if __name__ == '__main__':
    main()