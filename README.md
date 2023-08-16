# Backbone Evaluation for Fine-Grained Image Recognition

Our method obtains favorable results in terms of accuracy in a variety of 
fine-grained tasks(aircraft, cars, variety of plants, birds, other animals
asides from birds, anime characters, and food).

![](./assets/table_sota_224.png)

Samples of our crops are shown below:

![](./assets/crops.png)

Despite it failing in certain scenarios, we remark that our method achieves 
these results at a much lower computational cost  compared to the alternatives.

![](./assets/table_cost.png)

Try it on [Colab!](https://colab.research.google.com/drive/1Jt9bLqHyyqTGARQjBJ2-Ge0IYXIYk7yE?usp=sharing)

Logs for all runs (including LR search) on [Wandb!](https://wandb.ai/edwin_ed520/GLSim/)

The code for our model (and the ViT backbone) is in `glsim/model_utils/glsim.py`.

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
