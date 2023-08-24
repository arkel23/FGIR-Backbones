import math
import torch
from torch import nn

from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from kornia.contrib import extract_tensor_patches, combine_tensor_patches

from .drop_path import DropPath
from .vit import LearnedPositionalEmbedding1D
from .transformer import PositionWiseFeedForward


try:
    from torch.nn.functional import scaled_dot_product_attention as eff_attention
    EFF_ATTENTION_AVAILABLE = True
except ImportError:
    print('PyTorch2.0 not available')
    EFF_ATTENTION_AVAILABLE = False


class Attention(nn.Module):
    # Inspired by https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
    def __init__(self, dim_in, dim_context=None, heads=8, dim_head=64, drop_prob=0., eff_attention=True):
        super().__init__()
        dim_context = dim_in if dim_context is None else dim_context

        dim_inner = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.drop_prob = drop_prob

        self.proj_q = nn.Linear(dim_in, dim_inner, bias=True)
        self.proj_kv = nn.Linear(dim_context, dim_inner * 2, bias=True)

        if EFF_ATTENTION_AVAILABLE and eff_attention:
            self.eff_attention = True

        self.soft = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(drop_prob)

        self.proj_out = nn.Linear(dim_inner, dim_in)

    def forward(self, x, context=None):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        x: B, S_in, D_in ; context: B, S_c, D_c
        """

        # (B, S_in, D_in) -proj-> (B, S_in, D_inner) -rearrange/split-> (B, H, S_in, W)
        # (B, S_c, D_c) -proj-> (B, S_c, D_inner) -rearrange/split-> (B, H, S_c, W)
        q = self.proj_q(x)
        if context is None:
            k, v = self.proj_kv(x).chunk(2, dim=-1)
        else:
            k, v = self.proj_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b s (h w) -> b h s w', h=self.heads), [q, k, v])

        if hasattr(self, 'eff_attention'):
            out = eff_attention(q, k, v, dropout_p=self.drop_prob)
        else:
            # (B, H, S_in, W) @ (B, H, W, S_c) -> (B, H, S_in, S_c)
            sim = torch.matmul(q, rearrange(k, 'b h s w -> b h w s')) * self.scale

            # rescaled and normalized similarity (all you need)
            attn = self.drop(self.soft(sim))

            # (B, H, S_in, S_c) @ (B, H, S_c, W) -> (B, H, S_in, W)
            out = torch.matmul(attn, v)

        out = rearrange(out, 'b h s w -> b s (h w)')
        out = self.proj_out(out)

        return out


class BlockXFormer(nn.Module):
    """SPF Block"""
    def __init__(self, dim_hidden, ff_dim=None, heads=None, dim_head=None,
                 drop_prob=0.1, attn_drop_prob=0, layer_norm_eps=1e-12, sd=0, eff_attn=True):
        super().__init__()

        ff_dim = dim_hidden * 4 if ff_dim is None else ff_dim
        dim_head = 64 if dim_head is None else dim_head
        heads = int(dim_hidden // dim_head) if heads is None else heads

        if sd > 0:
            self.drop = DropPath(sd)
        else:
            self.drop = nn.Dropout(drop_prob)

        self.xattn_norm = nn.LayerNorm(dim_hidden, eps=layer_norm_eps)
        self.xattn = Attention(dim_hidden, dim_hidden, heads, dim_head, attn_drop_prob, eff_attn)

        self.pwff = nn.Sequential(
            self.drop,
            nn.LayerNorm(dim_hidden, eps=layer_norm_eps),
            PositionWiseFeedForward(dim_hidden, ff_dim)
        )

    def forward(self, x, context):
        if hasattr(self, 'attn'):
            x = x + self.drop(self.xattn(self.xattn_norm(x), context))
            x = x + self.pwff(x)
        return x

class PatchPromptTuning(nn.Module):
    """Learnable prompt and database of styles
    Sequence of SPF Blocks"""
    def __init__(self, prompt_len=1, num_channels=3, image_size=224, patch_size=16, 
                 num_layers=1, dim_hidden=128, pos_embeddings='learned',
                 ff_dim=None, heads=None, dim_head=None, drop_prob=0.1, 
                 attn_drop_prob=0, layer_norm_eps=1e-12, sd=0, eff_attn=True):
        super().__init__()

        self.patch_size = patch_size
        self.prompt_len = prompt_len

        dim_vision = int(num_channels * (patch_size ** 2))
        num_patches = int((image_size // patch_size) ** 2)

        self.prompt = nn.Parameter(torch.zeros(1, prompt_len, dim_hidden))

        self.proj_vision = nn.Sequential(
            Rearrange('b c (ph p1) (pw p2) -> b (ph pw) (c p1 p2)', p1=patch_size, p2=patch_size),
            nn.Linear(dim_vision, dim_hidden)
        )

        if pos_embeddings == 'learned':
            self.pos_embeddings_prompt = LearnedPositionalEmbedding1D(prompt_len, dim_hidden)
            self.pos_embeddings_context = LearnedPositionalEmbedding1D(num_patches, dim_hidden)

        self.blocks = nn.ModuleList([
            BlockXFormer(
                dim_hidden, ff_dim, heads, dim_head, drop_prob,
                attn_drop_prob, layer_norm_eps, sd, eff_attn)
        for _ in range(num_layers)])

        self.proj_out = nn.Sequential(
            nn.Linear(dim_hidden, num_channels),
            nn.LayerNorm(num_channels, eps=layer_norm_eps),
        )

        # Initialize weights
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        if hasattr(self, 'pos_embedding_prompt'):
            nn.init.normal_(self.pos_embedding_prompt.pos_embedding, std=0.02)
            nn.init.normal_(self.pos_embedding_context.pos_embedding, std=0.02)

    def forward(self, imgs):
        """Input has shape `(batch_size, c, h, w)`"""
        b, c, h, w = imgs.shape

        prompt = repeat(self.prompt, '1 s d -> b s d', b=b)

        context = self.proj_vision(imgs)

        if hasattr(self, 'pos_embeddings_prompt'):
            prompt = self.pos_embeddings_prompt(prompt)
            context = self.pos_embeddings_context(context)

        for block in self.blocks:
            prompt = block(prompt, context)

        prompt = self.proj_out(prompt)

        imgs = extract_tensor_patches(imgs, window_size=(self.patch_size, self.patch_size),
                                   stride=(self.patch_size, self.patch_size))
        num_patches = imgs.shape[1]
        ph = int(math.sqrt(num_patches))

        imgs = rearrange(imgs, 'b p c h w -> b p c (h w)')

        imgs = torch.cat([
            repeat(prompt, 'b s c -> b p c s', p=num_patches),
            imgs[:, :, :, self.prompt_len:]
        ], dim=-1)

        imgs = rearrange(imgs, 'b p c (h w) -> b p c h w', h=self.patch_size)

        imgs = combine_tensor_patches(imgs, original_size=(c, h, w), 
                                   window_size=(ph, ph), stride=(ph, ph))

        return imgs

    @torch.no_grad()
    def resize_pos_embedding_context(self, len_new):
        """Rescale the grid of position embeddings in a sensible manner"""
        import numpy as np
        from scipy.ndimage import zoom

        if not hasattr(self, 'pos_embeddings_context'):
            print('No embeddings to resize')
            return 0
        elif self.pos_embeddings_context.pos_embedding.shape[1] == len_new:
            print('Positional embeddings are already at correct size')
            return 0

        posemb = self.pos_embeddings_context.pos_embedding.detach().cpu().numpy()

        # Get old and new grid sizes
        h_old = int(np.sqrt(posemb.shape[1]))
        h_new = int(np.sqrt(len_new))
        posemb_grid = rearrange(posemb, '1 (h_old w_old) d -> h_old w_old d', h_old=h_old)

        # Rescale grid
        zoom_factor = (h_new / h_old, h_new / h_old, 1)
        posemb_new = zoom(posemb_grid, zoom_factor, order=1)
        posemb_new = rearrange(posemb_new, 'h_new w_new d -> 1 (h_new w_new) d')
        posemb_new = torch.nn.Parameter(torch.from_numpy(posemb_new))

        self.pos_embeddings_context.pos_embedding = posemb_new

        print(f'Resized pos embedding for context from: {posemb.shape} to {posemb_new.shape}')
        return 0
