'''
https://github.com/raoyongming/CAL/tree/master/fgvc
https://github.com/raoyongming/CAL/blob/master/fgvc/train_distributed.py
https://github.com/raoyongming/CAL/blob/master/fgvc/models/cal.py
https://github.com/raoyongming/CAL/blob/master/fgvc/infer.py
https://github.com/raoyongming/CAL/blob/master/fgvc/utils.py
'''
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ml_collections
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce

from fgir_backbones.train_utils.mix import rand_bbox


EPSILON = 1e-6


def get_cal_config():
    """Returns the CAL configuration."""
    config = ml_collections.ConfigDict()
    config.num_attention_maps = 32
    config.beta = 5e-2
    config.single_crop = False
    return config

'''
# augment function
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1, percent_max=True):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                if percent_max:
                    theta_c = random.uniform(*theta) * atten_map.max()
                else:
                    theta_c = random.uniform(*theta) * atten_map.mean()
            else:
                if percent_max:
                    theta_c = theta * atten_map.max()
                else:
                    theta_c = theta * atten_map.mean()

            # 0 / 1 mask based on if attention at x,y is higher than max value * threshold percentage
            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c

            # x, y indices for 1 values in mask
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])

            # select highest/min height/width
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(
                    images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                if percent_max:
                    theta_d = random.uniform(*theta) * atten_map.max()
                else:
                    theta_c = random.uniform(*theta) * atten_map.mean()
            else:
                if percent_max:
                    theta_c = theta * atten_map.max()
                else:
                    theta_c = theta * atten_map.mean()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], \
            but received unsupported augmentation method %s' % mode)
'''

def get_theta(atten_map, theta=0.5, percent_max=True, top_k=None):
    if top_k:
        atten_1d = rearrange(atten_map, 'b 1 h w -> b 1 (h w)')
        values, _ = atten_1d.topk(top_k, dim=-1, largest=True)
        theta_c = values[0, 0, -1]
    elif isinstance(theta, tuple):
        if percent_max:
            theta_c = random.uniform(*theta) * atten_map.max()
        else:
            theta_c = random.uniform(*theta) * atten_map.mean()
    else:
        if percent_max:
            theta_c = theta * atten_map.max()
        else:
            theta_c = theta * atten_map.mean()

    return theta_c



def get_width_height(atten_map, imgH=224, imgW=224, theta_c=0.5, padding_ratio=0.1):
    # 0 / 1 mask based on if attention at x,y is higher than max value * threshold percentage
    crop_mask = F.interpolate(atten_map, size=(imgH, imgW), mode='bicubic') >= theta_c

    # x, y indices for 1 values in mask
    nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])

    # select highest/min height/width
    try:
        height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
    except:
        height_min = 0
    try:
        height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
    except:
        height_max = imgH
    try:
        width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
    except:
        width_min = 0
    try:
        width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
    except:
        width_max = imgW

    return height_min, height_max, width_min, width_max


# augment function
def batch_augment(images, attention_map, mode='crop', theta=0.5,
                  padding_ratio=0.1, percent_max=True, top_k=None):
    batches, _, imgH, imgW = images.size()

    if mode == 'cm':
        attention_map = rearrange(attention_map, '(b r) 1 fh fw -> b r 1 fh fw', r=2)
        batches = batches // 2
        cloned = images.clone()

        cm_images = []
        for batch_index in range(batches):
            atten_map_1 = attention_map[batch_index:batch_index + 1, 0]
            theta_c_1 = get_theta(atten_map_1, theta, percent_max, top_k)
            height_min_1, height_max_1, width_min_1, width_max_1 = get_width_height(atten_map_1, imgH, imgW, theta_c_1, padding_ratio)

            atten_map_2 = attention_map[batch_index:batch_index + 1, 1]
            theta_c_2 = get_theta(atten_map_2, theta, percent_max, top_k)
            height_min_2, height_max_2, width_min_2, width_max_2 = get_width_height(atten_map_2, imgH, imgW, theta_c_2, padding_ratio)

            img_1 = cloned[batch_index*2:batch_index*2+1].clone()
            img_2 = cloned[batch_index*2+1:batch_index*2+2].clone()

            crop_1 = img_1[:, :, height_min_1:height_max_1, width_min_1:width_max_1].clone()
            crop_2 = img_2[:, :, height_min_2:height_max_2, width_min_2:width_max_2].clone()

            _, _, h_1, w_1 = crop_1.shape
            _, _, h_2, w_2 = crop_2.shape

            img_1[:, :, height_min_1:height_max_1, width_min_1:width_max_1] = F.upsample_bilinear(crop_2, size=(h_1, w_1))
            img_2[:, :, height_min_2:height_max_2, width_min_2:width_max_2] = F.upsample_bilinear(crop_1, size=(h_2, w_2))

            cm_images.append(img_1)
            cm_images.append(img_2)
        cm_images = torch.cat(cm_images, dim=0)
        return cm_images

    elif mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]

            theta_c = get_theta(atten_map, theta, percent_max, top_k)

            height_min, height_max, width_min, width_max = get_width_height(atten_map, imgH, imgW, theta_c, padding_ratio)

            crop_images.append(
                F.upsample_bilinear(
                    images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                    size=(imgH, imgW)))

        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        if top_k:
            _, _, h, w = attention_map.shape
            atten_1d = rearrange(attention_map, 'b 1 h w -> b 1 (h w)')
            values, _ = atten_1d.topk(top_k, dim=-1, largest=True)
            mask_masks = atten_1d[:, 0, :] < values[:, :, -1]
            mask_masks = rearrange(mask_masks.float(), 'b (h w) -> b 1 h w', h=h, w=w)
            mask_masks = F.interpolate(mask_masks, size=(imgH, imgW), mode='bicubic')

            mask_images = images * mask_masks
            return mask_images

        mask_masks = []

        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]

            theta_m = get_theta(atten_map, theta, percent_max)

            mask_masks.append(F.interpolate(atten_map, size=(imgH, imgW), mode='bicubic') < theta_m)
        mask_masks = torch.cat(mask_masks, dim=0)
        mask_images = images * mask_masks.float()
        return mask_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'mask\'], \
            but received unsupported augmentation method %s' % mode)


class BasicConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# Bilinear Attention Pooling
class BAP_Counterfactual(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP_Counterfactual, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        if self.training:
            fake_att = torch.zeros_like(attentions).uniform_(0, 2)
        else:
            fake_att = torch.ones_like(attentions)
        counterfactual_feature = (torch.einsum('imjk,injk->imn', (fake_att, features)) / float(H * W)).view(B, -1)

        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(
            torch.abs(counterfactual_feature) + EPSILON)

        counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)
        return feature_matrix, counterfactual_feature


class WSDAN_CAL(nn.Module):
    """
    WS-DAN models
    Hu et al.,
    "See Better Before Looking Closer: Weakly Supervised Data Augmentation Network
    for Fine-Grained Visual Classification",
    arXiv:1901.09891
    """
    def __init__(self, num_classes, num_features=2048, num_attention_maps=32, cm=False):
        super(WSDAN_CAL, self).__init__()
        # Attention Maps
        self.num_attention_maps = num_attention_maps
        self.attentions = BasicConv2D(num_features, num_attention_maps, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAP_Counterfactual(pool='GAP')

        # Classification Layer
        self.fc = nn.Linear(num_attention_maps * num_features, num_classes, bias=False)

        self.cm = cm

    def visualize(self, feature_maps):
        # Feature Maps, Attention Maps and Feature Matrix
        attention_maps = self.attentions(feature_maps)

        feature_matrix, _ = self.bap(feature_maps, attention_maps)
        p = self.fc(feature_matrix * 100.)

        return p, attention_maps

    def forward(self, feature_maps):
        # Feature Maps, Attention Maps and Feature Matrix
        batch_size = feature_maps.size(0)
        attention_maps = self.attentions(feature_maps)

        feature_matrix, feature_matrix_hat = self.bap(feature_maps, attention_maps)

        # Classification
        p = self.fc(feature_matrix * 100.)

        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                if self.cm:
                    k_index = np.random.choice(self.num_attention_maps, 3, p=attention_weights.cpu().numpy())
                else:
                    k_index = np.random.choice(self.num_attention_maps, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        return p, p - self.fc(feature_matrix_hat * 100.), feature_matrix, attention_map


class CAL(nn.Module):
    def __init__(self, model, seq_len=196, hidden_size=768, num_classes=1000,
                 bsd=False, device='cpu', topk_crop=0, da_pool=0, ap_only=False, voting=0, cm=False):
        super(CAL, self).__init__()

        config = get_cal_config()
        self.beta = config.beta
        self.single_crop = config.single_crop
        self.topk_crop = topk_crop
        self.ap_only = ap_only
        self.voting = voting
        self.cm = cm

        # Network Initialization
        if bsd:
            ph = int(math.sqrt(seq_len))
            self.encoder = nn.Sequential(
                model,
                Rearrange('b (h w) d -> b d h w', h=ph)
            )
        else:
            self.encoder = model

        # discriminative feature selection mechanism
        self.dfsm = WSDAN_CAL(num_classes, hidden_size, config.num_attention_maps, self.cm)

        self.feature_center = torch.zeros(
            num_classes, config.num_attention_maps * hidden_size, device=device)

        if da_pool:
            self.da_pool = nn.AvgPool2d(2, 2)

        print('WSDAN: num_attention_maps: {}'.format(config.num_attention_maps))

    def forward(self, x, y=None):
        if self.training and self.cm and y is not None:
            # raw image
            feature_maps = self.encoder(x)
            y_pred_raw, y_pred_aux, feature_matrix, attention_map = self.dfsm(feature_maps)
            if hasattr(self, 'da_pool'):
                attention_map = self.da_pool(attention_map)

            # Update Feature Center
            feature_center_batch = F.normalize(self.feature_center[y], dim=-1)
            self.feature_center[y] += self.beta * (feature_matrix.detach() - feature_center_batch)

            # attention cropping
            with torch.no_grad():
                crop_images = batch_augment(
                    x, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6))
                drop_images = batch_augment(
                    x, attention_map[:, 1:2, :, :], mode='drop', theta=(0.2, 0.5))
                cm_images = batch_augment(
                    x, attention_map[:, 2:, :, :], mode='cm', theta=(0.4, 0.6), padding_ratio=0)

            aug_images = torch.cat([crop_images, cm_images, drop_images], dim=0)

            # crop images forward
            feature_maps = self.encoder(aug_images)
            y_pred_aug, y_pred_aux_aug, _, _ = self.dfsm(feature_maps)

            y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)

            # final prediction
            bs = x.shape[0]
            y_pred_aug_crops, _, _ = torch.split(y_pred_aug, [bs, bs, bs], dim=0)
            y_pred = (y_pred_raw + y_pred_aug_crops) / 2.

            return (y_pred, y_pred_raw, y_pred_aux, feature_matrix, feature_center_batch,
                    y_pred_aug, aug_images)

        elif self.training and y is not None:
            # raw image
            feature_maps = self.encoder(x)
            y_pred_raw, y_pred_aux, feature_matrix, attention_map = self.dfsm(feature_maps)
            if hasattr(self, 'da_pool'):
                attention_map = self.da_pool(attention_map)

            # Update Feature Center
            feature_center_batch = F.normalize(self.feature_center[y], dim=-1)
            self.feature_center[y] += self.beta * (feature_matrix.detach() - feature_center_batch)

            # attention cropping
            with torch.no_grad():
                crop_images = batch_augment(
                    x, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6))
                drop_images = batch_augment(x, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
            aug_images = torch.cat([crop_images, drop_images], dim=0)

            # crop images forward
            feature_maps = self.encoder(aug_images)
            y_pred_aug, y_pred_aux_aug, _, _ = self.dfsm(feature_maps)

            y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)

            # final prediction
            y_pred_aug_crops, _ = torch.split(y_pred_aug, [x.shape[0], x.shape[0]], dim=0)
            y_pred = (y_pred_raw + y_pred_aug_crops) / 2.

            return (y_pred, y_pred_raw, y_pred_aux, feature_matrix, feature_center_batch,
                    y_pred_aug, aug_images)


        elif self.ap_only:
            feature_maps = self.encoder(x)
            y_pred, _, _, _ = self.dfsm(feature_maps)
            return y_pred


        elif self.single_crop:
            feature_maps = self.encoder(x)
            y_pred_raw, _, _, attention_map = self.dfsm(feature_maps)
            if hasattr(self, 'da_pool'):
                attention_map = self.da_pool(attention_map)

            crop_images3 = batch_augment(x, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            feature_maps = self.encoder(crop_images3)
            y_pred_crop3, _, _, _ = self.dfsm(feature_maps)

            # final prediction
            y_pred = (y_pred_raw + y_pred_crop3) / 2.

            return y_pred, crop_images3


        elif self.topk_crop:
            k = self.topk_crop
            x_m = torch.flip(x, [3])

            # Raw Image
            feature_maps = self.encoder(x)
            y_pred_raw, _, _, attention_map = self.dfsm(feature_maps)
            if hasattr(self, 'da_pool'):
                attention_map = self.da_pool(attention_map)

            feature_maps = self.encoder(x_m)
            y_pred_raw_m, _, _, attention_map_m = self.dfsm(feature_maps)
            if hasattr(self, 'da_pool'):
                attention_map_m = self.da_pool(attention_map_m)

            # Object Localization and Refinement
            crop_images = batch_augment(x, attention_map, mode='crop', top_k=k, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images)
            y_pred_crop, _, _, _ = self.dfsm(feature_maps)

            crop_images2 = batch_augment(x, attention_map, mode='crop', top_k=k+2, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images2)
            y_pred_crop2, _, _, _ = self.dfsm(feature_maps)

            crop_images3 = batch_augment(x, attention_map, mode='crop', top_k=k+4, padding_ratio=0.05)
            feature_maps = self.encoder(crop_images3)
            y_pred_crop3, _, _, _ = self.dfsm(feature_maps)

            crop_images_m = batch_augment(x_m, attention_map_m, mode='crop', top_k=k+1, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images_m)
            y_pred_crop_m, _, _, _ = self.dfsm(feature_maps)

            crop_images_m2 = batch_augment(x_m, attention_map_m, mode='crop', top_k=k+3, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images_m2)
            y_pred_crop_m2, _, _, _ = self.dfsm(feature_maps)

            crop_images_m3 = batch_augment(x_m, attention_map_m, mode='crop', top_k=k+5, padding_ratio=0.05)
            feature_maps = self.encoder(crop_images_m3)
            y_pred_crop_m3, _, _, _ = self.dfsm(feature_maps)

            y_pred = (y_pred_raw + y_pred_crop + y_pred_crop2 + y_pred_crop3) / 4.
            y_pred_m = (y_pred_raw_m + y_pred_crop_m + y_pred_crop_m2 + y_pred_crop_m3) / 4.
            y_pred = (y_pred + y_pred_m) / 2.

            samples = torch.cat([x, crop_images, crop_images2, crop_images3,
                               crop_images_m, crop_images_m2, crop_images_m3], dim=0)
            return y_pred, samples


        elif self.voting:
            k = self.topk_crop
            x_m = torch.flip(x, [3])

            # Raw Image
            feature_maps = self.encoder(x)
            y_pred_raw, _, _, attention_map = self.dfsm(feature_maps)
            if hasattr(self, 'da_pool'):
                attention_map = self.da_pool(attention_map)

            feature_maps = self.encoder(x_m)
            y_pred_raw_m, _, _, attention_map_m = self.dfsm(feature_maps)
            if hasattr(self, 'da_pool'):
                attention_map_m = self.da_pool(attention_map_m)

            # Object Localization and Refinement
            crop_images = batch_augment(x, attention_map, mode='crop', top_k=k, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images)
            y_pred_crop, _, _, _ = self.dfsm(feature_maps)

            crop_images2 = batch_augment(x, attention_map, mode='crop', top_k=k+2, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images2)
            y_pred_crop2, _, _, _ = self.dfsm(feature_maps)

            crop_images3 = batch_augment(x, attention_map, mode='crop', top_k=k+4, padding_ratio=0.05)
            feature_maps = self.encoder(crop_images3)
            y_pred_crop3, _, _, _ = self.dfsm(feature_maps)

            crop_images_m = batch_augment(x_m, attention_map_m, mode='crop', top_k=k+1, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images_m)
            y_pred_crop_m, _, _, _ = self.dfsm(feature_maps)

            crop_images_m2 = batch_augment(x_m, attention_map_m, mode='crop', top_k=k+3, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images_m2)
            y_pred_crop_m2, _, _, _ = self.dfsm(feature_maps)

            crop_images_m3 = batch_augment(x_m, attention_map_m, mode='crop', top_k=k+5, padding_ratio=0.05)
            feature_maps = self.encoder(crop_images_m3)
            y_pred_crop_m3, _, _, _ = self.dfsm(feature_maps)

            y_voting = torch.stack([y_pred_raw, y_pred_crop, y_pred_crop2, y_pred_crop3,
                                    y_pred_raw_m, y_pred_crop_m, y_pred_crop_m2, y_pred_crop_m3], dim=1)

            y_pred = (y_pred_raw + y_pred_crop + y_pred_crop2 + y_pred_crop3) / 4.
            y_pred_m = (y_pred_raw_m + y_pred_crop_m + y_pred_crop_m2 + y_pred_crop_m3) / 4.
            y_pred = (y_pred + y_pred_m) / 2.

            samples = torch.cat([x, crop_images, crop_images2, crop_images3,
                               crop_images_m, crop_images_m2, crop_images_m3], dim=0)
            return y_pred, y_voting, samples


        else:
            x_m = torch.flip(x, [3])

            # Raw Image
            feature_maps = self.encoder(x)
            y_pred_raw, _, _, attention_map = self.dfsm(feature_maps)
            if hasattr(self, 'da_pool'):
                attention_map = self.da_pool(attention_map)

            feature_maps = self.encoder(x_m)
            y_pred_raw_m, _, _, attention_map_m = self.dfsm(feature_maps)
            if hasattr(self, 'da_pool'):
                attention_map_m = self.da_pool(attention_map_m)

            # Object Localization and Refinement
            crop_images = batch_augment(x, attention_map, mode='crop', theta=0.3, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images)
            y_pred_crop, _, _, _ = self.dfsm(feature_maps)

            crop_images2 = batch_augment(x, attention_map, mode='crop', theta=0.2, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images2)
            y_pred_crop2, _, _, _ = self.dfsm(feature_maps)

            crop_images3 = batch_augment(x, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            feature_maps = self.encoder(crop_images3)
            y_pred_crop3, _, _, _ = self.dfsm(feature_maps)

            crop_images_m = batch_augment(x_m, attention_map_m, mode='crop', theta=0.3, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images_m)
            y_pred_crop_m, _, _, _ = self.dfsm(feature_maps)

            crop_images_m2 = batch_augment(x_m, attention_map_m, mode='crop', theta=0.2, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images_m2)
            y_pred_crop_m2, _, _, _ = self.dfsm(feature_maps)

            crop_images_m3 = batch_augment(x_m, attention_map_m, mode='crop', theta=0.1, padding_ratio=0.05)
            feature_maps = self.encoder(crop_images_m3)
            y_pred_crop_m3, _, _, _ = self.dfsm(feature_maps)

            y_pred = (y_pred_raw + y_pred_crop + y_pred_crop2 + y_pred_crop3) / 4.
            y_pred_m = (y_pred_raw_m + y_pred_crop_m + y_pred_crop_m2 + y_pred_crop_m3) / 4.
            y_pred = (y_pred + y_pred_m) / 2.

            samples = torch.cat([x, crop_images, crop_images2, crop_images3,
                               crop_images_m, crop_images_m2, crop_images_m3], dim=0)
            return y_pred, samples
