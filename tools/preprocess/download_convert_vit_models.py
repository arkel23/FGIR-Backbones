import argparse

from ifahead import ViT, ViTConfig


parser = argparse.ArgumentParser()
parser.add_argument('--models_list', type=str, nargs='+', default=['vit_b16'])
args = parser.parse_args()

# in21k pretrained
# models_list = ['vit_b16', 'vit_b32', 'vit_l16', 'vit_h14']

for model_name in args.models_list:
    cfg = ViTConfig(model_name=model_name)
    model = ViT(cfg, pretrained=True)
    print(cfg)
    # print(model)
