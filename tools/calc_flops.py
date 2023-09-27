# https://github.com/Lyken17/pytorch-OpCounter
# https://github.com/sovrasov/flops-counter.pytorch
import torch
from thop import profile
from ptflops import get_model_complexity_info

from fgir_backbones.model_utils.build_model import build_model
from fgir_backbones.other_utils.build_args import parse_train_args
from fgir_backbones import ViTConfig
from fgir_backbones.model_utils.modules_others.bit import resnetv2_101x3_bitm_in21k


MODELS = ('vgg19_bn', 'resnet101', 'resnetv2_101x3_bitm_in21k', 'vit_b16',
          'swin_base_patch4_window7_224_in22k', 'resnetv2_101', 'convnext_base_in22k',
          'van_b3', 'beitv2_base_patch16_224_in22k')


def count_params(model):
    return sum([p.numel() for p in model.parameters()])


class FGFLOPS(object):
    """Computes the inference flops for transformers."""

    def __init__(self, image_size=224, patch_size=16, hidden_size=768, 
                 num_hidden_layers=12, num_classes=1000, channels_in=3, **kwargs):
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.channels_in = channels_in
        self.num_hidden_layers = num_hidden_layers

        self.seq_len = self.calc_seq_len()

        self.num_attention_heads = self.hidden_size // 64

        self.num_hidden_layers = 12

    def calc_seq_len(self):
        stride_size = 16
        seq_len = (((self.image_size - self.patch_size) / stride_size) + 1) ** 2
        seq_len += 1
        return seq_len

    def get_flops(self):
        patch_flops = 2 * (self.seq_len - 1) * (self.patch_size ** 2) * self.channels_in * self.hidden_size

        msa_flops = (4 * self.seq_len * (self.hidden_size ** 2)) + (2 * (self.seq_len ** 2) * self.hidden_size)
        pwffn_flops = 8 * self.seq_len * (self.hidden_size ** 2)
        layerwise_flops = msa_flops + pwffn_flops

        out_flops = self.hidden_size * self.num_classes

        flops = patch_flops + (self.num_hidden_layers * layerwise_flops) + out_flops

        return flops


def main():

    args = parse_train_args()

    args.num_classes = 1000

    x = torch.rand(1, 3, 224, 224).to(args.device)

    for name in MODELS:
        args.model_name = name
        if name == 'resnetv2_101x3_bitm_in21k':
            model = resnetv2_101x3_bitm_in21k().to(args.device)
        else:
            model = build_model(args)

        params = count_params(model) / 1e6

        macs, _ = profile(model, inputs=(x, ))
        macs = macs / 1e9

        macs2, _ = get_model_complexity_info(
            model, (3, 224, 224), as_strings=False,
            print_per_layer_stat=args.debugging, verbose=args.debugging)
        macs2 = macs2 / 1e9

        print(name, params, macs, macs2)

        if 'vit' in name:
            cfg = ViTConfig(model_name=args.model_name)
            flops = FGFLOPS(
                image_size=args.image_size, patch_size=cfg.patch_size[0],
                hidden_size=cfg.hidden_size, layers=cfg.num_hidden_layers, num_classes=1000).get_flops()
            print('{}: {:.2f} GFLOPs'.format(name, (flops / (1e9))))

if __name__ == "__main__":
    main()
