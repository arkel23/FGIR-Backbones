from math import sqrt
import torch
from einops import rearrange
from pytorch_grad_cam import GradCAM


VIT_MODELS = ['deit', 'beit', 'swin', 'van', 'vit']


def reshape_transform(tensor):
    h = int(sqrt(tensor.shape[1]))
    if h ** 2 != tensor.shape[1]:
        tensor = tensor[:, 1:, :]

    result = rearrange(tensor, 'b (h w) d -> b d h w', h=h)
    return result


def calc_gradcam(args, model, img):
    # for name, param in model.named_parameters():
    #   print(name)
    target_layer_dic ={
        'beitv2_base_patch16_224_in22k': 'model.model.blocks[-1].norm1',
        #'convnext_base_in22k': 'model.model.stages[-1].blocks[-1].mlp.fc2',
        'convnext_base_in22k': 'model.model.head.norm',
        'resnet101': 'model.model.layer4[-1]',
        'resnetv2_101': 'model.model.norm',
        'resnetv2_101x3_bitm_in21k': 'model.model.norm',
        # 'swin_base_patch4_window7_224_in22k': 'model.model.norm',
        'swin_base_patch4_window7_224_in22k': 'model.model.layers[-1].blocks[-1].norm1',
        'van_b3': 'model.model.norm4',
        'vgg19_bn': 'model.model.features[-1]',
        'vit_b16': 'model.model.encoder.blocks[-1].norm1',
    }

    target_layer = eval(target_layer_dic[args.model_name])

    for _, param in model.named_parameters():
        param.requires_grad = True

    if any(mn in args.model_name for mn in VIT_MODELS):
        compute_cam = GradCAM(model=model, target_layers=[target_layer],
                              use_cuda=True, reshape_transform=reshape_transform)
    else:
        compute_cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=True)

    cam = compute_cam(input_tensor=img)

    cam = torch.from_numpy(rearrange(cam, '1 h w -> (h w)'))
    return cam
