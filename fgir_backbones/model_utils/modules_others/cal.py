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
from timm.models.layers import trunc_normal_, DropPath
from scipy.stats import kurtosis

try:
    from torch.nn.functional import scaled_dot_product_attention as eff_attention
    EFF_ATTENTION_AVAILABLE = True
except ImportError:
    print('PyTorch2.0 not available')
    EFF_ATTENTION_AVAILABLE = False

EPSILON = 1e-6


def get_cal_config():
    """Returns the CAL configuration."""
    config = ml_collections.ConfigDict()
    config.num_attention_maps = 32
    config.beta = 5e-2
    config.single_crop = False
    return config


# augment function
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1, percent_max=True, kur_adjust=False):
    batches, _, imgH, imgW = images.size()

    '''
    if kur_adjust:
        pool_x = Reduce('b 1 h w -> b w', 'mean')
        pool_y = Reduce('b 1 h w -> b h', 'mean')
        pooled_x = pool_x(attention_map)
        pooled_y = pool_y(attention_map)
        pooled = reduce(torch.stack([pooled_x, pooled_y], dim=1), 'b c hw -> b hw', 'mean')
        kur = kurtosis(pooled.cpu(), axis=1, fisher=False)
        kur = np.nan_to_num(kur, copy=True, nan=0, posinf=0, neginf=0)
        kur_max = reduce(kur, 'b -> 1', 'max')
        kur_min = reduce(kur, 'b -> 1', 'min')
        kur = (kur - kur_min) / (kur_max - kur_min)
    '''

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            # if kur_adjust:
            #    kr = kur[batch_index]
            #    if not np.isfinite(kr):
            #        kr = 0
            if isinstance(theta, tuple):
                '''
                if kur_adjust:
                    theta_c = random.uniform(*theta)
                    if theta_c > 0.5:
                        theta_c = theta_c - 0.2 * math.log(theta_c) * kr
                    else:
                        theta_c = theta_c + 0.2 * theta_c*kr
                    if percent_max:
                        theta_c = theta_c * atten_map.max()
                    else:
                        theta_c = theta_c * atten_map.mean()
                '''
                if percent_max:
                    theta_c = random.uniform(*theta) * atten_map.max()
                else:
                    theta_c = random.uniform(*theta) * atten_map.mean()
            else:
                # if kur_adjust and percent_max:
                #    theta_c = (theta + theta*kr) * atten_map.max()
                # elif kur_adjust:
                #    theta_c = (theta + theta*kr) * atten_map.mean()
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


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class TSMPool(nn.Module):
    # Inspired by https://github.com/Sense-X/SiT/blob/main/models/lvvit.py
    def __init__(self, dim, drop_path=0.1, init_values=1e-5):
        super().__init__()

        self.weight = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(dim / 2)),
            nn.GELU(),
            nn.Linear(int(dim / 2), 1),
        )

        # self.scale = nn.Parameter(torch.ones(1, 1, 1))

        self.soft = nn.Softmax(dim=1)

        # layerscale code (init_values: 1e-5 for dinov2) taken from
        # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
        self.ls = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        x: B, S_in, D_in ; context: B, S_c, D_c
        """
        weight = self.weight(x)
        # weight = self.soft(weight * self.scale)
        weight = self.soft(weight)
        weight = rearrange(weight, 'b s 1 -> b 1 s')

        h = torch.matmul(weight, x)
        # x = x + self.drop_path(self.ls(h))

        return h[:, 0, :]


class AttentionPool(nn.Module):
    # Inspired by https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
    def __init__(self, dim_in, heads=8, dim_head=64, drop_prob=0.,
                 drop_path=0.1, init_values=1e-5, eff_attention=True):
        super().__init__()

        dim_inner = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.drop_prob = drop_prob

        self.cls_token = nn.Parameter(torch.rand(1, 1, dim_inner))

        self.norm1 = nn.LayerNorm(dim_in)
        # self.proj_qkv = nn.Linear(dim_in, dim_inner * 3, bias=True)
        self.proj_q = nn.Linear(dim_in, dim_inner, bias=True)
        self.proj_kv = nn.Linear(dim_in, dim_inner * 2, bias=True)

        if EFF_ATTENTION_AVAILABLE and eff_attention:
            self.eff_attention = True

        self.soft = nn.Softmax(dim=-1)
        self.drop_attention = nn.Dropout(drop_prob)

        self.proj_out = nn.Linear(dim_inner, dim_in)

        self.norm2 = nn.LayerNorm(dim_in)
        self.pwffn = nn.Sequential(
            nn.Linear(dim_in, dim_in * 4),
            nn.GELU(),
            nn.Linear(dim_in * 4, dim_in)
        )

        self.norm3 = nn.LayerNorm(dim_in)
        # self.classifier = nn.Linear(dim_in, dim_in)

        # layerscale code (init_values: 1e-5 for dinov2) taken from
        # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
        self.ls = LayerScale(dim_in, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim_in, init_values=init_values) if init_values else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # self.init_weights()

    # @torch.no_grad()
    # def init_weights(self):
        # def _init(m):
        #    if isinstance(m, nn.Linear):
        #        trunc_normal_(m.weight, std=.02)
        #        if m.bias is not None:
        #            nn.init.zeros_(m.bias)
        # trunc_normal_(self.cls_token, std=.02)
        # self.apply(_init)


    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        x: B, S_in, D_in ; context: B, S_c, D_c
        """

        # (B, S_in, D_in) -proj-> (B, S_in, D_inner) -rearrange/split-> (B, H, S_in, W)
        # (B, S_c, D_c) -proj-> (B, S_c, D_inner) -rearrange/split-> (B, H, S_c, W)
        # q = self.cls_token
        # k, v = self.proj_kv(x).chunk(2, dim=-1)
        h = self.norm1(x)
        # q = self.proj_q(reduce(h, 'b s k -> b 1 k', 'mean'))
        q = self.proj_q(h[:, :1, :])
        k, v = self.proj_kv(h).chunk(2, dim=-1)
        # q, k, v = self.proj_qkv(h).chunk(3, dim=-1)
        
        q, k, v = map(lambda t: rearrange(t, 'b s (h w) -> b h s w', h=self.heads), [q, k, v])

        if hasattr(self, 'eff_attention'):
            out = eff_attention(q, k, v, dropout_p=self.drop_prob)
        else:
            # (B, H, S_in, W) @ (B, H, W, S_c) -> (B, H, S_in, S_c)
            sim = torch.matmul(q, rearrange(k, 'b h s w -> b h w s')) * self.scale

            # rescaled and normalized similarity (all you need)
            attn = self.drop_attention(self.soft(sim))

            # (B, H, S_in, S_c) @ (B, H, S_c, W) -> (B, H, S_in, W)
            out = torch.matmul(attn, v)

        out = rearrange(out, 'b h s w -> b s (h w)')
        # out = reduce(x, 'b s k -> b 1 k', 'mean') + self.drop_path(self.ls(self.proj_out(out)))
        x = reduce(x, 'b s k -> b 1 k', 'mean') + self.drop_path(self.ls(self.proj_out(out)))
        # x = x + self.drop_path(self.ls(self.proj_out(out)))
        # out = F.softmax(self.proj_out(out), dim=-1)
        # out = self.cls_token + self.drop_path(self.ls(self.proj_out(out)))

        x = x[:, 0, :]
        x = x + self.drop_path2(self.ls2(self.pwffn(self.norm2(x))))

        x = self.norm3(x)
        # x = reduce(x, 'b s d -> b d', 'mean')
        # x = self.classifier(x)
        # return self.norm(out[:, 0, :])
        return x


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
    def __init__(self, num_classes, num_features=2048, num_attention_maps=32):
        super(WSDAN_CAL, self).__init__()
        # Attention Maps
        self.num_attention_maps = num_attention_maps
        self.attentions = BasicConv2D(num_features, num_attention_maps, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAP_Counterfactual(pool='GAP')

        # Classification Layer
        self.fc = nn.Linear(num_attention_maps * num_features, num_classes, bias=False)

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
                k_index = np.random.choice(self.num_attention_maps, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        return p, p - self.fc(feature_matrix_hat * 100.), feature_matrix, attention_map


class CAL(nn.Module):
    def __init__(self, model, seq_len=196, hidden_size=768, num_classes=1000,
                 bsd=False, device='cpu', cal_ap_only=False, cal_voting=None,
                 attention_pool=False):
        super(CAL, self).__init__()

        config = get_cal_config()
        self.beta = config.beta
        self.single_crop = config.single_crop
        self.cal_ap_only = cal_ap_only
        self.cal_voting = cal_voting

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
        self.dfsm = WSDAN_CAL(num_classes, hidden_size, config.num_attention_maps)

        self.feature_center = torch.zeros(
            num_classes, config.num_attention_maps * hidden_size, device=device)

        if attention_pool:
            self.attention_pool = AttentionPool(
                num_classes, num_classes, 4, drop_prob=0.0, drop_path=0.1, init_values=1e-5)
            # self.attention_pool = TSMPool(num_classes, drop_path=0.1, init_values=1e-5)

        print('WSDAN: num_attention_maps: {}'.format(config.num_attention_maps))

    def forward(self, x, y=None):
        if self.training and y is not None and hasattr(self, 'attention_pool'):
            # raw image
            feature_maps = self.encoder(x)
            y_pred_raw, y_pred_aux, feature_matrix, attention_map = self.dfsm(feature_maps)

            # Update Feature Center
            feature_center_batch = F.normalize(self.feature_center[y], dim=-1)
            self.feature_center[y] += self.beta * (feature_matrix.detach() - feature_center_batch)

            # attention cropping
            with torch.no_grad():
                crop_images = batch_augment(
                    x, attention_map[:, :1, :, :], mode='crop', theta=(0.3, 0.7), padding_ratio=0.1)
                drop_images = batch_augment(x, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
            aug_images = torch.cat([crop_images, drop_images], dim=0)

            # crop images forward
            feature_maps = self.encoder(aug_images)
            y_pred_aug, y_pred_aux_aug, _, _ = self.dfsm(feature_maps)

            y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)

            # final prediction
            y_pred_aug_crops, _ = torch.split(y_pred_aug, [x.shape[0], x.shape[0]], dim=0)

            y_pred = torch.stack([y_pred_raw, y_pred_aug_crops], dim=1)
            y_pred = self.attention_pool(y_pred)

            return (y_pred, y_pred_raw, y_pred_aux, feature_matrix, feature_center_batch,
                    y_pred_aug, crop_images)

        elif self.training and y is not None:
            # raw image
            feature_maps = self.encoder(x)
            y_pred_raw, y_pred_aux, feature_matrix, attention_map = self.dfsm(feature_maps)

            # Update Feature Center
            feature_center_batch = F.normalize(self.feature_center[y], dim=-1)
            self.feature_center[y] += self.beta * (feature_matrix.detach() - feature_center_batch)

            # attention cropping
            with torch.no_grad():
                crop_images = batch_augment(
                    x, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
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
                    y_pred_aug, crop_images)

        elif self.cal_ap_only:
            feature_maps = self.encoder(x)
            y_pred, _, _, _ = self.dfsm(feature_maps)
            return y_pred

        elif self.single_crop:
            feature_maps = self.encoder(x)
            y_pred_raw, _, _, attention_map = self.dfsm(feature_maps)

            crop_images3 = batch_augment(x, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            feature_maps = self.encoder(crop_images3)
            y_pred_crop3, _, _, _ = self.dfsm(feature_maps)

            # final prediction
            # y_pred = (y_pred_raw + y_pred_crop3) / 2.
            y_pred = torch.stack([y_pred_raw, y_pred_crop3], dim=1)
            y_pred = self.attention_pool(y_pred)

            return y_pred, crop_images3

        elif hasattr(self, 'attention_pool'):
            x_m = torch.flip(x, [3])

            # Raw Image
            feature_maps = self.encoder(x)
            y_pred_raw, _, _, attention_map = self.dfsm(feature_maps)

            feature_maps = self.encoder(x_m)
            y_pred_raw_m, _, _, attention_map_m = self.dfsm(feature_maps)

            # Object Localization and Refinement
            crop_images = batch_augment(x, attention_map, mode='crop', theta=0.7, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images)
            y_pred_crop, _, _, _ = self.dfsm(feature_maps)

            crop_images2 = batch_augment(x, attention_map, mode='crop', theta=0.5, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images2)
            y_pred_crop2, _, _, _ = self.dfsm(feature_maps)

            crop_images3 = batch_augment(x, attention_map, mode='crop', theta=0.3, padding_ratio=0.05)
            feature_maps = self.encoder(crop_images3)
            y_pred_crop3, _, _, _ = self.dfsm(feature_maps)

            crop_images_m = batch_augment(x_m, attention_map_m, mode='crop', theta=0.6, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images_m)
            y_pred_crop_m, _, _, _ = self.dfsm(feature_maps)

            crop_images_m2 = batch_augment(x_m, attention_map_m, mode='crop', theta=0.4, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images_m2)
            y_pred_crop_m2, _, _, _ = self.dfsm(feature_maps)

            crop_images_m3 = batch_augment(x_m, attention_map_m, mode='crop', theta=0.2, padding_ratio=0.05)
            feature_maps = self.encoder(crop_images_m3)
            y_pred_crop_m3, _, _, _ = self.dfsm(feature_maps)

            y_pred = torch.stack([y_pred_raw, y_pred_crop, y_pred_crop2, y_pred_crop3,
                                    y_pred_raw_m, y_pred_crop_m, y_pred_crop_m2, y_pred_crop_m3], dim=1)
            y_pred = self.attention_pool(y_pred)

            return y_pred

        elif self.cal_voting:
            x_m = torch.flip(x, [3])

            # Raw Image
            feature_maps = self.encoder(x)
            y_pred_raw, _, _, attention_map = self.dfsm(feature_maps)

            feature_maps = self.encoder(x_m)
            y_pred_raw_m, _, _, attention_map_m = self.dfsm(feature_maps)

            # Object Localization and Refinement
            crop_images = batch_augment(x, attention_map, mode='crop', theta=0.7, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images)
            y_pred_crop, _, _, _ = self.dfsm(feature_maps)

            crop_images2 = batch_augment(x, attention_map, mode='crop', theta=0.5, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images2)
            y_pred_crop2, _, _, _ = self.dfsm(feature_maps)

            crop_images3 = batch_augment(x, attention_map, mode='crop', theta=0.3, padding_ratio=0.05)
            feature_maps = self.encoder(crop_images3)
            y_pred_crop3, _, _, _ = self.dfsm(feature_maps)

            crop_images_m = batch_augment(x_m, attention_map_m, mode='crop', theta=0.6, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images_m)
            y_pred_crop_m, _, _, _ = self.dfsm(feature_maps)

            crop_images_m2 = batch_augment(x_m, attention_map_m, mode='crop', theta=0.4, padding_ratio=0.1)
            feature_maps = self.encoder(crop_images_m2)
            y_pred_crop_m2, _, _, _ = self.dfsm(feature_maps)

            crop_images_m3 = batch_augment(x_m, attention_map_m, mode='crop', theta=0.2, padding_ratio=0.05)
            feature_maps = self.encoder(crop_images_m3)
            y_pred_crop_m3, _, _, _ = self.dfsm(feature_maps)

            y_voting = torch.stack([y_pred_raw, y_pred_crop, y_pred_crop2, y_pred_crop3,
                                    y_pred_raw_m, y_pred_crop_m, y_pred_crop_m2, y_pred_crop_m3], dim=1)

            y_pred = (y_pred_raw + y_pred_crop + y_pred_crop2 + y_pred_crop3) / 4.
            y_pred_m = (y_pred_raw_m + y_pred_crop_m + y_pred_crop_m2 + y_pred_crop_m3) / 4.
            y_pred = (y_pred + y_pred_m) / 2.

            return y_pred, y_voting, crop_images

        else:
            x_m = torch.flip(x, [3])

            # Raw Image
            feature_maps = self.encoder(x)
            y_pred_raw, _, _, attention_map = self.dfsm(feature_maps)

            feature_maps = self.encoder(x_m)
            y_pred_raw_m, _, _, attention_map_m = self.dfsm(feature_maps)

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

            return y_pred
            # return y_pred, crop_images
