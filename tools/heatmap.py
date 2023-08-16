import math

import cv2
from PIL import Image
from einops import rearrange, reduce

import torch
from torch.nn import functional as F

from gradcam import calc_gradcam


def save_pil_images(images, fp):
    images.save(fp)
    print(f'Saved file to : {fp}')
    return 0


def apply_mask(img, mask, top_k=8, th=False, sq=True, color=True):
    '''
    img are pytorch tensors of size C x S1 x S2, range 0-255 (unnormalized)
    mask are pytorch tensors of size (S1 / P * S2 / P) of floating values (range prob -1 ~ 1)
    heatmap combination requires using opencv (and therefore numpy arrays)
    '''

    if th:
        # compute threshold for binarizing
        highest, _ = mask.topk(top_k, dim=-1, largest=True)
        th_value = highest[-1].item()
        mask = torch.where(mask >= th_value, 1, 0)
    elif sq:
        mask = (mask ** 4)

    # 1d sequence to 2d feature map and convert to numpy array
    h = int(math.sqrt(mask.shape[0]))
    mask = rearrange(mask, '(h w) -> h w', h=h)
    mask = mask.cpu().numpy()

    # print('Mean and std of mask: ', np.mean(mask), np.std(mask))

    if color:
        mask = cv2.normalize(
            mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        mask = mask.astype('uint8')

    mask = cv2.resize(mask, (img.shape[-1], img.shape[-1]))

    img = rearrange(img.cpu().numpy(), '1 c h w -> h w c')
    img = cv2.cvtColor(img * 255, cv2.COLOR_RGB2BGR).astype('uint8')

    if color:
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    else:
        mask = rearrange(mask, 'h w -> h w 1')        

    if color:
        result = cv2.addWeighted(mask, 0.5, img, 0.5, 0)
    else:
        result = (mask * img)
        result = cv2.normalize(
            result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result = result.astype('uint8')

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    result = Image.fromarray(result)

    return result


def inverse_normalize(tensor, norm_custom=False):
    if norm_custom:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def attention_rollout(scores_soft):
    # https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
    att_mat = torch.stack(scores_soft).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1), device=att_mat.device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size(), device=att_mat.device)
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    # Attention from the output token to the input space.
    h_2d = int(math.sqrt(joint_attentions[-1].shape[-1]))
    if h_2d == joint_attentions[-1].shape[-1]:
        mask = reduce(joint_attentions[-1], 's1 s2 -> s1', 'mean')
    else:
        mask = joint_attentions[-1][0, 1:]
    # grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    # mask = v[0, 1:].reshape(grid_size, grid_size)

    '''
    att_max, _ = torch.max(mask, dim=-1)
    att_min, _ = torch.min(mask, dim=-1)
    att = (mask - att_min) / (att_max - att_min)
    att = rearrange(att, '(h w) -> 1 h w', h=int(math.sqrt(att.shape[0])))
    pool = torch.nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
    att = pool(att)
    print(torch.round(att, decimals=2))
    '''

    return mask


def calc_attention(attention):
    attention = attention.squeeze()
    attention = torch.mean(attention, dim=1)

    h_2d = int(math.sqrt(attention.shape[-1]))
    if h_2d == attention.shape[-1]:
        attention = reduce(attention, 's1 s2 -> s2', 'mean')
    else:
        attention = attention[0, 1:]

    '''
    att_max, _ = torch.max(attention, dim=-1)
    att_min, _ = torch.min(attention, dim=-1)
    att = (attention - att_min) / (att_max - att_min)
    att = rearrange(att, '(h w) -> 1 h w', h=int(math.sqrt(att.shape[0])))
    pool = torch.nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
    att = pool(att)
    print(torch.round(att, decimals=2))
    '''

    return attention


def calc_distance_glsim(tokens):
    h_2d = int(math.sqrt(tokens.shape[1]))
    if h_2d == tokens.shape[1]:
        g = tokens[0, :1, :]
        l = tokens[0, 1:, :]
    else:
        g = reduce(tokens, '1 s d -> 1 d', 'mean')
        l = tokens[0, :, :]

    distances = F.cosine_similarity(g, l, dim=-1)

    return distances


def calc_distance_psm(scores_soft, head=0):
    att_mat = torch.stack(scores_soft).squeeze(1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(att_mat.size(), device=att_mat.device)
    joint_attentions[0] = att_mat[0]

    for n in range(1, att_mat.size(0)):
        joint_attentions[n] = torch.matmul(att_mat[n], joint_attentions[n-1])

    # Attention from the output CLS token to the input space.
    mask = joint_attentions[-1][head, 0, 1:]

    return mask


def calc_distance_maws(scores_soft, scores):
    a = reduce(scores_soft.squeeze(), 'h s1 s2 -> s1 s2', 'mean')[0, 1:]
    b = F.softmax(scores.squeeze(), dim=-2)[:, 1:, 0]
    b = reduce(b, 'h s1 -> s1', 'mean')
    ma = a * b
    return ma


def calc_asim(x_norm, scores_soft):
    distances = calc_distance_glsim(x_norm)

    attn = attention_rollout(scores_soft)

    asim = ((distances - torch.min(distances)) / torch.max(distances)) * ((attn - torch.min(attn)) / torch.max(attn))

    return asim


def make_heatmaps(args, fp, img, model, inter, scores_soft, scores, x_norm,
                  vis_mask_all=False, save=True):

    if vis_mask_all:

        for mechanism in args.vis_mask_list:
            args.vis_mask = mechanism
            make_heatmaps(args, fp, img, model, inter, scores_soft, scores, x_norm)

        return 0

    img_unnorm = inverse_normalize(img.detach().clone(), args.custom_mean_std)

    if args.vis_mask.split('_')[-1].isnumeric() and len(args.vis_mask.split('_'))  == 3:
        select_start = int(args.vis_mask.split('_')[-2])
        select_end = int(args.vis_mask.split('_')[-1])
    elif args.vis_mask.split('_')[-1].isnumeric():
        select = int(args.vis_mask.split('_')[-1])
    elif len(args.vis_mask.split('_')) == 1:
        if 'maws' in args.vis_mask:
            select = 10
        elif 'psm' in args.vis_mask:
            select = 0
        elif 'rollout' in args.vis_mask:
            select_start = 0
            select_end = 12
        else:
            select = 11

    if args.vis_mask == 'gradcam':
        mask = calc_gradcam(args, model, img)
    elif args.vis_mask == 'asim':
        mask = calc_asim(x_norm,  scores_soft)
    elif args.vis_mask == 'glsim_norm':
        mask = calc_distance_glsim(x_norm)
    elif 'glsim' in args.vis_mask:
        mask = calc_distance_glsim(inter[select])
    elif 'attention' in args.vis_mask:
        mask = calc_attention(scores_soft[select])        
    elif 'rollout' in args.vis_mask:
        mask = attention_rollout(scores_soft[select_start:select_end])
    elif 'psm' in args.vis_mask:
        mask = calc_distance_psm(scores_soft, select)
    elif 'maws' in args.vis_mask:
        mask = calc_distance_maws(scores_soft[select], scores[select])

    img_masked = apply_mask(img_unnorm, mask, top_k=args.dynamic_top, th=args.vis_th_topk,
                            sq=args.vis_mask_sq, color=args.vis_mask_color)

    if not save:
        return img_masked

    fp = fp.replace('.png', f'_{args.vis_mask}.png')

    save_pil_images(img_masked, fp)

    return img_masked
