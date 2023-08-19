import re
from types import SimpleNamespace

import timm
import torch
import torch.nn as nn

from .modules_others import beit_dict, van_dict, ViT, ViTConfig, PatchPromptTuning, Head, CAL


VITS = [
    'vit_t4', 'vit_t8', 'vit_t16', 'vit_t32', 'vit_s8', 'vit_s16', 'vit_s32',
    'vit_b8', 'vit_b16', 'vit_b32', 'vit_l16', 'vit_l32', 'vit_h14']


def build_model(args):
    # initiates model and loss
    if args.model_name in VITS or 'van' in args.model_name or args.model_name in timm.list_models(pretrained=True):
        model = ClassifierModel(args)
    else:
        raise NotImplementedError

    args.seq_len = model.cfg.seq_len

    if args.ckpt_path:
        load_model_compatibility_mode(args, model)

    if args.freeze_backbone:
        freeze_backbone(args, model)

    if args.distributed:
        model.cuda()
    else:
        model.to(args.device)

    print(f'Initialized classifier: {args.model_name}')
    return model


def freeze_backbone(args, model):
    keywords = ['head', 'prompt', 'dfsm', 'feature_center']

    if args.unfreeze_first_conv:
        if any(name in args.model_name for name in VITS + ['swin', 'beit']):
            keywords.append('patch_embed')
        elif 'vgg' in args.model_name:
            keywords.append('features.0')
        elif 'van' in args.model_name:
            keywords.append('patch_embed1')
        elif 'resnetv2' in args.model_name or 'convnext' in args.model_name:
            # v2 and v2_101x3
            keywords.append('stem')
        elif 'resnet' in args.model_name:
            # (v1) _101
            raise NotImplementedError
            keywords.append('conv1')

    for name, param in model.named_parameters():
        if any(kw in name for kw in keywords):
            param.requires_grad = True
            print(name)
        else:
            param.requires_grad = False

    print('Total parameters (M): ', sum([p.numel() for p in model.parameters()]) / (1e6))
    print('Trainable parameters (M): ', sum([p.numel() for p in model.parameters() if p.requires_grad]) / (1e6))


def get_first_conv_kernel_stride(args):
    if args.model_name in VITS:
        name = args.model_name.split('_')[-1]
        pattern = r'[a-zA-Z](\d+)'
        patch_size = int(re.search(pattern, name).group(1))
        stride = patch_size
    elif 'vgg' in args.model_name:
        patch_size = 3
        stride = 1
    elif 'van' in args.model_name:
        # https://github.com/Visual-Attention-Network/VAN-Classification/blob/main/models/van.py#L129
        patch_size = 7
        stride = 4
    elif any(name in args.model_name for name in ['swin', 'convnext']):
        patch_size = 4
        stride = 4
    elif 'resnet' in args.model_name:
        patch_size = 7
        stride = 2
    elif 'beit' in args.model_name:
        patch_size = 16
        stride = 16
    else:
        raise NotImplementedError

    args.patch_kernel = patch_size
    args.patch_stride = stride
            
    return None
        

def load_model_compatibility_mode(args, model):
    state_dict = torch.load(
        args.ckpt_path, map_location=torch.device('cpu'))['model']
    expected_missing_keys = []

    # retrocompatibility with prev experiments
    if 'model.head.head.weight' in state_dict.keys():
        state_dict['head.head.weight'] = state_dict.pop('model.head.head.weight')
        state_dict['head.head.bias'] = state_dict.pop('model.head.head.bias')
    elif 'head.weight' in state_dict.keys():
        state_dict['head.head.weight'] = state_dict.pop('head.weight')
        state_dict['head.head.bias'] = state_dict.pop('head.bias')

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
    return 0


def get_backbone(args):
    if args.model_name in VITS:
        args.classifier = 'pool' if args.selector == 'cal' else args.classifier
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
        model = ViT(cfg, pretrained=args.pretrained)
        # cfg = cfg

    elif 'beitv2' in args.model_name:
        args.classifier = 'cls' if args.classifier is None else args.classifier
        model = beit_dict[args.model_name](
            pretrained=args.pretrained, img_size=args.image_size, num_classes=0,
            drop_path_rate=args.sd, global_pool='')
    elif 'van' in args.model_name:
        model = van_dict[args.model_name](
            pretrained=args.pretrained,img_size=args.image_size, drop_path_rate=args.sd)
    elif 'vgg' in args.model_name:
        model = timm.create_model(args.model_name, pretrained=args.pretrained,
                                        num_classes=0, global_pool='')
    elif any(model in args.model_name for model in ['resnet', 'convnext']):
        model = timm.create_model(
            args.model_name, pretrained=args.pretrained, num_classes=0,
            drop_path_rate=args.sd, global_pool='')
    else:
        model = timm.create_model(
            args.model_name, pretrained=args.pretrained, num_classes=0,
            img_size=args.image_size, drop_path_rate=args.sd, global_pool='')

    return model


class ClassifierModel(nn.Module):
    def __init__(self, args):
        super(ClassifierModel, self).__init__()

        self.model_name = args.model_name
        model = get_backbone(args)

        s, d, bsd = self.get_out_features(args.image_size, model)

        if args.selector == 'cal':
            self.model = CAL(model, s, d, args.num_classes, bsd, args.device)
            assert 'beit' not in args.model_name, 'beit not compatible with cal'
        else:
            self.model = get_backbone(args)
            self.head = Head(args.classifier, d, args.num_classes, bsd)

        self.cfg = SimpleNamespace(**{'seq_len': s, 'hidden_size': d, 'num_channels': 3})

        if args.ppt:
            get_first_conv_kernel_stride(args)
            patch_size = args.patch_stride if args.prompt_stride else args.patch_kernel
            self.prompt = PatchPromptTuning(self.cfg.num_channels, patch_size, args.prompt_len)

    @torch.no_grad()
    def get_out_features(self, image_size, model):
        x = torch.rand(2, 3, image_size, image_size)
        x = model(x)

        if len(x.shape) == 3:
            b, s, d = x.shape
            bsd = True
        elif len(x.shape) == 4:
            b, d, h, w = x.shape
            s = h * w
            bsd = False

        return s, d, bsd

    def forward(self, images, targets=None, ret_inter=False):
        if hasattr(self, 'prompt'):
            images = self.prompt(images)

        if hasattr(self, 'head'):
            if (self.model_name in VITS or 'beit' in self.model_name) and ret_inter:
                features, scores = self.model(images, ret_inter)
            else:
                features = self.model(images)

            out = self.head(features)

            if ret_inter:
                return out, scores

        else:
            out = self.model(images, targets)

        return out
