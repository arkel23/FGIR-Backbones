from types import SimpleNamespace

import timm
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce

from .modules_others import van_dict, ViT, ViTConfig


VITS = (
    'vit_t4', 'vit_t8', 'vit_t16', 'vit_t32', 'vit_s8', 'vit_s16', 'vit_s32',
    'vit_b8', 'vit_b16', 'vit_b32', 'vit_l16', 'vit_l32', 'vit_h14')


def build_model(args):
    # initiates model and loss
    if args.model_name in VITS:
        model = VisionTransformer(args)
    elif 'van' in args.model_name or args.model_name in timm.list_models(pretrained=True):
        model = TIMMNets(args)
    else:
        raise NotImplementedError

    args.seq_len = model.cfg.seq_len

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

    if args.freeze_backbone:
        freeze_backbone(args, model)

    if args.distributed:
        model.cuda()
    else:
        model.to(args.device)

    print(f'Initialized classifier: {args.model_name}')
    return model


def freeze_backbone(args, model):
    keywords = ['head', 'prompt']

    for name, param in model.named_parameters():
        if any(kw in name for kw in keywords):
            param.requires_grad = True
        else:
            param.requires_grad = False

    print('Total parameters (M): ', sum([p.numel() for p in model.parameters()]) / (1e6))
    print('Trainable parameters (M): ', sum([p.numel() for p in model.parameters() if p.requires_grad]) / (1e6))


class VisionTransformer(nn.Module):
    def __init__(self, args):
        super(VisionTransformer, self).__init__()
        # init default config
        cfg = ViTConfig(model_name=args.model_name)
        # modify config if given an arg otherwise keep defaults
        args_temp = vars(args)
        for k, v in args_temp.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v if v is not None else getattr(cfg, k))
        cfg.calc_dims()
        # update the args with the final model config
        for attribute in vars(cfg):
            if hasattr(args, attribute):
                setattr(args, attribute, getattr(cfg, attribute))
        # init model
        self.model = ViT(cfg, pretrained=args.pretrained)
        self.cfg = cfg

    def forward(self, images, targets=None, ret_dist=False):
        out = self.model(images, ret_dist=ret_dist)
        return out


class TIMMNets(nn.Module):
    def __init__(self, args):
        super(TIMMNets, self).__init__()
        # init default config

        if 'van' in args.model_name:
            self.model = van_dict[args.model_name](
                pretrained=args.pretrained,img_size=args.image_size, drop_path_rate=args.sd)
        elif 'vgg' in args.model_name:
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

        self.cfg = SimpleNamespace(**{'seq_len': s})

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


class VisionTransformerDA(nn.Module):
    def __init__(self, args):
        super(VisionTransformerDA, self).__init__()
        # init default config
        cfg = ViTConfig(model_name=args.model_name)
        # modify config if given an arg otherwise keep defaults
        args_temp = vars(args)
        for k, v in args_temp.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v if v is not None else getattr(cfg, k))
        cfg.assertions_corrections()
        cfg.calc_dims()

        # update the args with the final model config
        for attribute in vars(cfg):
            if hasattr(args, attribute):
                setattr(args, attribute, getattr(cfg, attribute))
        # init model
        self.model = DAViT(cfg, pretrained=args.pretrained)
        self.cfg = cfg

    def forward(self, images, targets=None, ret_dist=False):
        out = self.model(images)
        return out
