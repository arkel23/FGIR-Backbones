import math
import torch
from torch import nn

from einops import repeat, rearrange
from kornia.contrib import extract_tensor_patches, combine_tensor_patches


class PatchPromptTuning(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, dim, patch_size=16, prompt_len=1):
        super().__init__()
        self.patch_size = patch_size
        self.prompt_len = prompt_len

        self.prompt = nn.Parameter(torch.zeros(1, dim, prompt_len))
        val = math.sqrt(6. / float(3 * (patch_size ** 2) + dim))
        nn.init.uniform_(self.prompt.data, -val, val)

    def forward(self, x):
        """Input has shape `(batch_size, c, h, w)`"""
        b, c, h, w = x.shape

        x = extract_tensor_patches(x, window_size=(self.patch_size, self.patch_size),
                                   stride=(self.patch_size, self.patch_size))
        num_patches = x.shape[1]
        ph = int(math.sqrt(num_patches))

        x = rearrange(x, 'b p c h w -> b p c (h w)')

        x = torch.cat([
            repeat(self.prompt, '1 c s -> b p c s', b=b, p=num_patches),
            x[:, :, :, self.prompt_len:]
        ], dim=-1)

        x = rearrange(x, 'b p c (h w) -> b p c h w', h=self.patch_size)

        x = combine_tensor_patches(x, original_size=(c, h, w), 
                                   window_size=(ph, ph), stride=(ph, ph))

        return x
