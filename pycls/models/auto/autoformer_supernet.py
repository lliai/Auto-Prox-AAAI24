import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import MODEL
from pycls.core.config import cfg
from .module.Linear_super import LinearSuper
from .module.layernorm_super import LayerNormSuper
from .module.multihead_super import AttentionSuper
from .module.embedding_super import PatchembedSuper
from timm.models.layers import DropPath,  trunc_normal_
import numpy as np
from .base import BaseTransformerModel




def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))



@MODEL.register()
class AutoFormer(BaseTransformerModel):

    def __init__(self):
        super(AutoFormer, self).__init__()
        # the configs of super arch

        self.patch_embed_super = PatchembedSuper(img_size=self.img_size, patch_size=self.patch_size,
                                                 in_chans=self.in_channels, embed_dim=self.hidden_dim)
        self.num_tokens = 1 + cfg.DISTILLATION.ENABLE_LOGIT

        # configs for the sampled subTransformer
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_layer_num = None
        self.sample_num_heads = None
        self.sample_dropout = None
        self.sample_output_dim = None

        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]  # stochastic depth decay rule

        for i in range(self.depth):
            self.blocks.append(TransformerEncoderLayer(dim=self.hidden_dim, num_heads=self.num_heads,
                                                       mlp_ratio=self.mlp_ratio, dropout=self.drop_rate,
                                                       attn_drop=self.attn_drop_rate, drop_path=dpr[i]
                                                       ))

        # parameters for vision transformer
        num_patches = self.patch_embed_super.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.hidden_dim))
        trunc_normal_(self.pos_embed, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.cls_token, std=.02)

        # self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = LayerNormSuper(super_embed_dim=self.hidden_dim)


        # classifier head
        self.head = LinearSuper(self.hidden_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self.initialize_hooks(self.blocks)

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
        feat_size = int(self.patch_embed_super.num_patches ** 0.5)
        # x = outputs[:, self.num_tokens:].view(outputs.size(0), feat_size, feat_size, self.hidden_dim)
        x = outputs[:, self.num_tokens:].view(outputs.size(0), feat_size, feat_size, self.sample_embed_dim)
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



    def set_sample_config(self, config: dict):

        self.sample_embed_dim = config['hidden_dim']
        self.sample_mlp_ratio = config['mlp_ratio']
        self.sample_layer_num = config['depth']
        self.sample_num_heads = config['num_heads']
        self.sample_dropout = calc_dropout(self.drop_rate, self.sample_embed_dim, self.hidden_dim)
        self.patch_embed_super.set_sample_config(self.sample_embed_dim)
        self.sample_output_dim = [self.sample_embed_dim]*(self.sample_layer_num)
        for i, blocks in enumerate(self.blocks):
            # not exceed sample layer number
            if i < self.sample_layer_num:
                sample_dropout = calc_dropout(self.drop_rate, self.sample_embed_dim, self.hidden_dim)
                sample_attn_dropout = calc_dropout(self.attn_drop_rate, self.sample_embed_dim, self.hidden_dim)
                blocks.set_sample_config(is_identity_layer=False,
                                        sample_embed_dim=self.sample_embed_dim,
                                        sample_mlp_ratio=self.sample_mlp_ratio[i],
                                        sample_num_heads=self.sample_num_heads[i],
                                        sample_dropout=sample_dropout,
                                        sample_out_dim=self.sample_output_dim[i],
                                        sample_attn_dropout=sample_attn_dropout)
            # exceeds sample layer number
            else:
                blocks.set_sample_config(is_identity_layer=True)
        self.norm.set_sample_config(self.sample_embed_dim)
        self.head.set_sample_config(self.sample_embed_dim, self.num_classes)

    def get_sampled_params_numel(self, config):
        self.set_sample_config(config)
        numels = []
        for name, module in self.named_modules():
            if hasattr(module, 'calc_sampled_param_num'):
                if name.split('.')[0] == 'blocks' and int(name.split('.')[1]) >= config['depth']:
                    continue
                numels.append(module.calc_sampled_param_num())

        return sum(numels) + self.sample_embed_dim* (2 +self.patch_embed_super.num_patches)





    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed_super(x)
        cls_tokens = self.cls_token[..., :self.sample_embed_dim].expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed[..., :self.sample_embed_dim]

        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        # start_time = time.time()
        for blk in self.blocks:
            x = blk(x)
        # print(time.time()-start_time)

        x = self.norm(x)

        # if self.gp:
        return torch.mean(x[:, 1:] , dim=1)

        # return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        logits = self.head(x)
        if self.num_tokens == 1:
            return logits

        self.distill_logits = None
        self.distill_logits = self.distill_head(x[:, 1])

        if self.training:
            return logits
        else:
            return (logits + self.distill_logits) / 2



class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments which
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, dropout=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, pre_norm=True, scale=False,
                 relative_position=False, change_qkv=True, max_relative_position=14):
        super().__init__()

        # the configs of super arch of the encoder, three dimension [embed_dim, mlp_ratio, and num_heads]
        self.super_embed_dim = dim
        self.super_mlp_ratio = mlp_ratio
        self.super_ffn_embed_dim_this_layer = int(mlp_ratio * dim)
        self.super_num_heads = num_heads
        self.normalize_before = pre_norm
        self.super_dropout = attn_drop
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale = scale
        self.relative_position = relative_position
        # self.super_activation_dropout = getattr(args, 'activation_dropout', 0)

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_num_heads_this_layer = None
        self.sample_scale = None
        self.sample_dropout = None
        self.sample_attn_dropout = None

        self.is_identity_layer = None
        self.attn = AttentionSuper(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=dropout, scale=self.scale, relative_position=self.relative_position, change_qkv=change_qkv,
            max_relative_position=max_relative_position
        )

        self.attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.ffn_layer_norm = LayerNormSuper(self.super_embed_dim)
        # self.dropout = dropout
        self.activation_fn = gelu
        # self.normalize_before = args.encoder_normalize_before

        self.fc1 = LinearSuper(super_in_dim=self.super_embed_dim, super_out_dim=self.super_ffn_embed_dim_this_layer)
        self.fc2 = LinearSuper(super_in_dim=self.super_ffn_embed_dim_this_layer, super_out_dim=self.super_embed_dim)


    def set_sample_config(self, is_identity_layer, sample_embed_dim=None, sample_mlp_ratio=None, sample_num_heads=None, sample_dropout=None, sample_attn_dropout=None, sample_out_dim=None):

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_out_dim = sample_out_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        self.sample_ffn_embed_dim_this_layer = int(sample_embed_dim*sample_mlp_ratio)
        self.sample_num_heads_this_layer = sample_num_heads

        self.sample_dropout = sample_dropout
        self.sample_attn_dropout = sample_attn_dropout
        self.attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

        self.attn.set_sample_config(sample_q_embed_dim=self.sample_num_heads_this_layer*64, sample_num_heads=self.sample_num_heads_this_layer, sample_in_embed_dim=self.sample_embed_dim)

        self.fc1.set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        self.fc2.set_sample_config(sample_in_dim=self.sample_ffn_embed_dim_this_layer, sample_out_dim=self.sample_out_dim)

        self.ffn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)


    def forward(self, x):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, patch_num , sample_embed_dim)`

        Returns:
            encoded output of shape `(batch, patch_num, sample_embed_dim)`
        """
        if self.is_identity_layer:
            return x

        # compute attn
        # start_time = time.time()

        residual = x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
        x = self.attn(x)
        x = F.dropout(x, p=self.sample_attn_dropout, training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)
        # print("attn :", time.time() - start_time)
        # compute the ffn
        # start_time = time.time()
        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        if self.scale:
            x = x * (self.super_mlp_ratio / self.sample_mlp_ratio)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)
        # print("ffn :", time.time() - start_time)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x



def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim





