# MIT License
#
# Copyright (c) 2024 Intelligent Robot Motion Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
ViT image encoder implementation from IBRL, https://github.com/hengyuan-hu/ibrl

Modified to support obs dicts with multiple image keys.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Union
import einops
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import math


@dataclass
class VitEncoderConfig:
    patch_size: int = 8
    depth: int = 1
    embed_dim: int = 128
    num_heads: int = 4
    act_layer = nn.GELU
    stride: int = -1
    embed_style: str = "embed2"
    embed_norm: int = 0


class VitEncoder(nn.Module):
    """
    Accepts either:
      - obs: Tensor [B,C,H,W] or [C,H,W]
      - obs: Dict[str, Tensor] with multiple image entries

    For dict input, it encodes each image key and concatenates tokens along seq dim:
      output: [B, K*num_patches, embed_dim]
    If flatten=True:
      output: [B, K*num_patches*embed_dim]
    """

    def __init__(
        self,
        obs_shape: List[int],
        cfg: VitEncoderConfig,
        num_channel=3,
        img_h=96,
        img_w=96,
        image_keys: Optional[Sequence[str]] = None,
        combine_mode: str = "token_concat",  # "token_concat" | "token_mean"
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.cfg = cfg

        self.vit = MinVit(
            embed_style=cfg.embed_style,
            embed_dim=cfg.embed_dim,
            embed_norm=cfg.embed_norm,
            num_head=cfg.num_heads,
            depth=cfg.depth,
            num_channel=num_channel,
            img_h=img_h,
            img_w=img_w,
        )
        self.img_h = img_h
        self.img_w = img_w

        self.image_keys = list(image_keys) if image_keys is not None else None
        self.combine_mode = combine_mode

        self.num_patch = self.vit.num_patches
        self.patch_repr_dim = self.cfg.embed_dim

        # repr_dim depends on key count if known + concat mode
        if self.image_keys is not None and self.combine_mode == "token_concat":
            self.repr_dim = self.cfg.embed_dim * self.vit.num_patches * len(self.image_keys)
        else:
            self.repr_dim = self.cfg.embed_dim * self.vit.num_patches

    def _to_bchw(self, x: torch.Tensor) -> torch.Tensor:
        # [C,H,W] -> [1,C,H,W]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"Expected image tensor with 3 or 4 dims, got shape {tuple(x.shape)}")

        # channels-last [B,H,W,C] -> [B,C,H,W]
        if x.shape[-1] in (1, 3, 4) and x.shape[1] not in (1, 3, 4):
            x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_bchw(x)

        # resize
        if x.shape[-2:] != (self.img_h, self.img_w):
            x = F.interpolate(x, size=(self.img_h, self.img_w), mode="bilinear", align_corners=False)

        # normalize like original
        if x.dtype.is_floating_point:
            if x.max().detach() <= 1.5:
                x = x - 0.5
            else:
                x = x / 255.0 - 0.5
        else:
            x = x.float() / 255.0 - 0.5

        return x

    def _auto_image_keys(self, obs_dict: Dict[str, torch.Tensor]) -> List[str]:
        keys: List[str] = []
        for k, v in obs_dict.items():
            if not torch.is_tensor(v):
                continue
            if v.dim() in (3, 4):
                keys.append(k)
        if len(keys) == 0:
            raise ValueError("No image-like tensors found in obs dict.")
        return keys

    def forward(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]], flatten=False) -> torch.Tensor:
        # single tensor (original behavior)
        if torch.is_tensor(obs):
            x = self._preprocess(obs)
            feats: torch.Tensor = self.vit.forward(x)  # [B,P,D]
            return feats.flatten(1, 2) if flatten else feats

        # dict of images
        if not isinstance(obs, dict):
            raise TypeError(f"obs must be a Tensor or a dict of Tensors, got {type(obs)}")

        keys = self.image_keys if self.image_keys is not None else self._auto_image_keys(obs)

        feats_list: List[torch.Tensor] = []
        for k in keys:
            if k not in obs:
                raise KeyError(f"Missing image key '{k}' in obs dict. Available keys: {list(obs.keys())}")
            xk = self._preprocess(obs[k])
            feats_k = self.vit.forward(xk)  # [B,P,D]
            feats_list.append(feats_k)

        if self.combine_mode == "token_concat":
            feats = torch.cat(feats_list, dim=1)  # [B,K*P,D]
        elif self.combine_mode == "token_mean":
            feats = torch.stack(feats_list, dim=0).mean(dim=0)  # [B,P,D]
        else:
            raise ValueError(f"Unknown combine_mode='{self.combine_mode}'")

        return feats.flatten(1, 2) if flatten else feats


class PatchEmbed1(nn.Module):
    def __init__(self, embed_dim, num_channel=3, img_h=96, img_w=96):
        super().__init__()
        self.conv = nn.Conv2d(num_channel, embed_dim, kernel_size=8, stride=8)

        self.num_patch = math.ceil(img_h / 8) * math.ceil(img_w / 8)
        self.patch_dim = embed_dim

    def forward(self, x: torch.Tensor):
        y = self.conv(x)
        y = einops.rearrange(y, "b c h w -> b (h  w) c")
        return y


class PatchEmbed2(nn.Module):
    def __init__(self, embed_dim, use_norm, num_channel=3, img_h=96, img_w=96):
        super().__init__()
        layers = [
            nn.Conv2d(num_channel, embed_dim, kernel_size=8, stride=4),
            nn.GroupNorm(embed_dim, embed_dim) if use_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2),
        ]
        self.embed = nn.Sequential(*layers)

        H1 = math.ceil((img_h - 8) / 4) + 1
        W1 = math.ceil((img_w - 8) / 4) + 1
        if img_w == 84 and img_h == 84:  # Hongsuk. Oct 26, 2025.
            H2 = math.ceil((H1 - 3) / 2)
            W2 = math.ceil((W1 - 3) / 2)
        else:
            H2 = math.ceil((H1 - 3) / 2) + 1
            W2 = math.ceil((W1 - 3) / 2) + 1

        self.num_patch = H2 * W2
        self.patch_dim = embed_dim

    def forward(self, x: torch.Tensor):
        y = self.embed(x)
        y = einops.rearrange(y, "b c h w -> b (h  w) c")
        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_head):
        super().__init__()
        assert embed_dim % num_head == 0

        self.num_head = num_head
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask):
        """
        x: [batch, seq, embed_dim]
        """
        qkv = self.qkv_proj(x)
        q, k, v = einops.rearrange(
            qkv, "b t (k h d) -> b k h t d", k=3, h=self.num_head
        ).unbind(1)
        # force flash/mem-eff attention, it will raise error if flash cannot be applied
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            attn_v = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, attn_mask=attn_mask
            )
        attn_v = einops.rearrange(attn_v, "b h t d -> b t (h d)")
        return self.out_proj(attn_v)


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_head, dropout):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_head)

        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.dropout(self.mha(self.layer_norm1(x), attn_mask))
        x = x + self.dropout(self._ff_block(self.layer_norm2(x)))
        return x

    def _ff_block(self, x):
        x = self.linear2(nn.functional.gelu(self.linear1(x)))
        return x


class MinVit(nn.Module):

    def __init__(
        self,
        embed_style,
        embed_dim,
        embed_norm,
        num_head,
        depth,
        num_channel=3,
        img_h=96,
        img_w=96,
    ):
        super().__init__()

        if embed_style == "embed1":
            self.patch_embed = PatchEmbed1(
                embed_dim,
                num_channel=num_channel,
                img_h=img_h,
                img_w=img_w,
            )
        elif embed_style == "embed2":
            self.patch_embed = PatchEmbed2(
                embed_dim,
                use_norm=embed_norm,
                num_channel=num_channel,
                img_h=img_h,
                img_w=img_w,
            )
        else:
            assert False

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patch, embed_dim)
        )
        layers = [
            TransformerLayer(embed_dim, num_head, dropout=0) for _ in range(depth)
        ]

        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.num_patches = self.patch_embed.num_patch

        # weight init
        trunc_normal_(self.pos_embed, std=0.02)
        named_apply(init_weights_vit_timm, self)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.net(x)
        return self.norm(x)


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def named_apply(
    fn, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


if __name__ == "__main__":
    # --- Single tensor (original style) ---
    obs_shape = [6, 128, 128]
    enc = VitEncoder(
        obs_shape,
        VitEncoderConfig(),
        num_channel=obs_shape[0],
        img_h=obs_shape[1],
        img_w=obs_shape[2],
    )

    print(enc)
    x = torch.rand(1, *obs_shape) * 255
    print("output size:", enc(x, flatten=False).size())
    print("repr dim:", enc.repr_dim, ", real dim:", enc(x, flatten=True).size())

    # --- Dict obs with multiple image keys (same style) ---
    img_shape = [3, 128, 128]
    enc_multi = VitEncoder(
        img_shape,
        VitEncoderConfig(),
        num_channel=img_shape[0],
        img_h=img_shape[1],
        img_w=img_shape[2],
        image_keys=["front_rgb", "wrist_rgb"],
        combine_mode="token_concat",
    )

    print(enc_multi)
    obs = {
        "front_rgb": torch.rand(1, *img_shape) * 255,
        "wrist_rgb": torch.rand(1, *img_shape) * 255,
    }
    print("output size:", enc_multi(obs, flatten=False).size())
    print("repr dim:", enc_multi.repr_dim, ", real dim:", enc_multi(obs, flatten=True).size())