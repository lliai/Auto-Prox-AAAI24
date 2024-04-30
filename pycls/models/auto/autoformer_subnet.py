import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import MODEL
from pycls.core.config import cfg
from timm.models.layers import DropPath,  trunc_normal_
import numpy as np
from .base import BaseTransformerModel
from .common import PatchEmbedding, TransformerLayer, layernorm




def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))



@MODEL.register()
class AutoFormerSub(BaseTransformerModel):

    def __init__(self, arch_config = None, num_classes = None):
        super(AutoFormerSub, self).__init__()
        # the configs of super arch

        if arch_config:
            self.num_heads = arch_config['num_heads']
            self.mlp_ratio = arch_config['mlp_ratio']
            self.hidden_dim = arch_config['hidden_dim']
            self.depth = arch_config['depth']

        else:
            self.num_heads = cfg.AUTOFORMER_SUBNET.NUM_HEADS
            self.mlp_ratio = cfg.AUTOFORMER_SUBNET.MLP_RATIO
            self.hidden_dim = cfg.AUTOFORMER_SUBNET.HIDDEN_DIM
            self.depth = cfg.AUTOFORMER_SUBNET.DEPTH

        if num_classes:
            self.num_classes = num_classes
        else:
            self.num_classes = cfg.MODEL.NUM_CLASSES

        # print('hidden dim is:'. self.hidden_dim)
        self.feature_dims = [self.hidden_dim] * self.depth

        self.patch_embed = PatchEmbedding(img_size=self.img_size, patch_size=self.patch_size, in_channels=self.in_channels, out_channels=self.hidden_dim)
        self.num_patches = self.patch_embed.num_patches
        self.num_tokens = 1 + cfg.DISTILLATION.ENABLE_LOGIT



        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]  # stochastic depth decay rule

        for i in range(self.depth):
            self.blocks.append(TransformerLayer(in_channels=self.hidden_dim, num_heads=self.num_heads[i], qkv_bias=True,
                                                       mlp_ratio=self.mlp_ratio[i], drop_rate=self.drop_rate,
                                                       attn_drop_rate=self.attn_drop_rate, drop_path_rate=dpr[i],
                                                       ))

        self.initialize_hooks(self.blocks)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, self.hidden_dim))
        trunc_normal_(self.pos_embed, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.cls_token, std=.02)

        # self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = layernorm(self.hidden_dim)


        # classifier head
        self.head = nn.Linear(self.hidden_dim, self.num_classes)

        self.apply(self._init_weights)


        self.distill_logits = None

        self.distill_token = None
        self.distill_head = None
        if cfg.DISTILLATION.ENABLE_LOGIT:
            self.distill_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            self.distill_head = nn.Linear(self.hidden_dim, self.num_classes)
            nn.init.zeros_(self.distill_head.weight)
            nn.init.constant_(self.distill_head.bias, 0)
            trunc_normal_(self.distill_token, std=.02)

    def _feature_hook(self, module, inputs, outputs):
        feat_size = int(self.patch_embed.num_patches ** 0.5)
        x = outputs[:, self.num_tokens:].view(outputs.size(0), feat_size, feat_size, self.hidden_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        self.features.append(x)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.num_tokens == 1:
            x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], dim=1)
        else:
            x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), self.distill_token.repeat(x.size(0), 1, 1), x], dim=1)

        x = x + self.pos_embed
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return torch.mean(x[:, 1:] , dim=1)



    def forward(self, x):
        x = self.forward_features(x)
        logits = self.head(x)
        if self.num_tokens == 1:
            return logits

        self.distill_logits = None
        self.distill_logits = self.distill_head(x)

        if self.training:
            return logits
        else:
            return (logits + self.distill_logits) / 2




