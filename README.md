# Backbone Evaluation for Fine-Grained Image Recognition

Official PyTorch code for the paper: [A Large-Scale Study on the Accuracy vs Cost Trade-offs of Training and Evaluation Settings in Fine-Grained Image Recognition](http://arxiv.org/abs/2605.18700), published at the Fine-Grained Visual Categorization (FGVC13) Workshop @ CVPR 2026.

This project provides a comprehensive analysis and benchmarking framework for evaluating different backbone architectures on fine-grained image recognition (FGIR) tasks. The repository focuses on comparing various training strategies (Frozen, Fine-tuned, CAL, CALMix) across multiple neural network architectures and datasets, with detailed metrics on both accuracy and computational costs.

The primary experiments evaluate **9 original backbones** across **17 datasets**:
**Models:**
- `vit_b16` - ViT B-16
- `vgg19_bn` - VGG-19
- `van_b3` - VAN B-3
- `swin_base_patch4_window7_224_in22k` - Swin B (IN21k)
- `resnetv2_101x3_bitm_in21k` - ResNet-101x3 (BiT-M)
- `resnetv2_101` - ResNetV2-101
- `resnet101` - ResNet-101
- `convnext_base_in22k` - ConvNeXt B (IN21k)
- `beitv2_base_patch16_224_in22k` - BEiT B-16

**Datasets:** Aircraft, Cars, Cotton, CUB, DAFB, Dogs, Flowers, Food, iNat17, Moe, NABirds, Pets, SoyAgeing, SoyGene, SoyGlobal, SoyLocal, VegFru

### Phase 2: Extended Evaluation (20 models × 4 datasets)
Additional experiments expand the model coverage to **20 backbones** on only **4 datasets**:

**Models:**
- `convnext_large_in22k` - ConvNeXt L (IN21k)
- `convnext_base` - ConvNeXt B (IN1k)
- `deit3_large_patch16_224_in21ft1k` - DeiT3 L-16 (IN21k)
- `deit3_base_patch16_224_in21ft1k` - DeiT3 B-16 (IN21k)
- `deit3_base_patch16_224` - DeiT3 B-16 (IN1k)
- `swin_large_patch4_window7_224_in22k` - Swin L (IN21k)
- `swin_base_patch4_window7_224` - Swin B (IN1k)
- `resnet18` - ResNet-18
- `tv_resnet101` - ResNet-101
- `tv_resnet34` - ResNet-34
- `tv_resnet50` - ResNet-50

**Datasets:** Aircraft, CUB, SoyGene, SoyLocal

Samples of Dataset used:
![](./assets/datasets.png)

![](./assets/box_acc_cost_3090_224_reduced.png)
Our CALMix variant improves accuracy further from CAL, but with more train time and the same problem of reduced inference throughput.

Our CAL-NC and CALMix-NC removes cropping during inference, restoring inference throughput comparable to Frozen or Fine-Tuned settings.

![](./assets/table_models_all.png)

Extensive benchmark across models and datasets. Swin B (IN21k) and ConvNeXt B (IN21k) achieve best results

## Setup

```
pip install -e . 
```

## Preparation

All of these require to first `chmod +x script_name` the corresponding scripts.

To download pretrained checkpoints for CUB, DAFB, iNat17, NABirds (and vanilla In-21k ckpts):
```
./scripts/download_ckpts.sh
python tools/preprocess/download_convert_vit_models.py
```

To download datasets:
```
./scripts/download.sh
```

To prepare the train and validation splits from the train_val set for each dataset (otherwise can skip this step and just copy the ones we included in the `data` directory to each respective dataset directory in order to ensure the splits are the same as ours):
```
./prepare_datasets.sh
```

To download and prepare NCFM dataset (requires Kaggle API token):
```
./ncfm_prepare_dataset.sh
```

Dataset stats:
```
./scripts/calc_hw.sh
```

## Train

To train a `GLSim-ViT B-16` with CLS classifier on CUB using image size 224:
```
python tools/train.py --cfg configs/cub_ft_is224_medaugs.yaml --lr 0.01 --model_name vit_b16 --cfg_method configs/methods/glsim.yaml
```

Similarly, for image size 448:
```
python tools/train.py --cfg configs/cub_ft_is224_medaugs.yaml --lr 0.01 --model_name vit_b16 --cfg_method configs/methods/glsim.yaml --cfg_is configs/settings/ft_is448.yaml
```

## Evaluation

To evaluate a particular checkpoint on the test set (logs results to W&B):

```
python tools/train.py --ckpt_path ckpts/cub_glsim_224.pth --test_only
```

To enforce batch size 1 (emulates streaming / on-demand classification behavior):
```
python tools/train.py --ckpt_path ckpts/cub_glsim_224.pth --batch_size 1
```

To visulize misclassification for a particular network on the test set:

```
python tools/train.py --ckpt_path ckpts/cub_glsim_224.pth --vis_errors
```

To save these results (results are saved in the same folder as train folder) (*note: takes some time):

```
python tools/train.py --ckpt_path ckpts/cub_glsim_224.pth --vis_errors_save
```

## Inference

Inference on a single image (saves results of original and crop side by side on `results_inference/`):
```
python tools/inference.py --ckpt_path ckpts/dafb_glsim.pth --images_path samples/others/dafb_rena_170785.jpg
```

To visualize the global-local similarity (and other fine-grained discriminative
feature selection mechanisms such as attention rollout as shown in the
following figure):

![](./assets/dfsm.png)

```
python tools/inference.py --ckpt_path ckpts/dafb_glsim.pth --images_path samples/others/dafb_rena_170785.jpg --vis_mask_sq --vis_mask glsim_norm
python tools/inference.py --ckpt_path ckpts/dafb_glsim.pth --images_path samples/others/dafb_rena_170785.jpg --vis_mask rollout
```

For doing inference on a whole folder (and its subdirectories):
```
python tools/inference.py --ckpt_path ckpts/cub_glsim_224.pth --images_path samples/
```

## Usage as a module
```
import torch
from glsim.model_utils import ViTGLSim, ViTConfig

model_name = 'vit_b16'
cfg = ViTConfig(model_name, debugging=True, classifier='cls', dynamic_anchor=True,
    reducer='cls', aggregator=True, aggregator_norm=True, aggregator_num_hidden_layers=1)
model = ViTGLSim(cfg)

x = torch.rand(2, cfg.num_channels, cfg.image_size, cfg.image_size)
out = model(x)
```
