import math
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from ..p_utils import get_layer_metric_array
from . import measure

class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2, dim=1):
        super(Normalize, self).__init__()
        self.power = power
        self.dim = dim

    def forward(self, x):
        norm = x.pow(self.power).sum(self.dim,
                                     keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Embed(nn.Module):
    def __init__(self, dim_in=256, dim_out=128):
        super(Embed, self).__init__()
        self.conv2d = nn.Conv2d(dim_in,
                                dim_out,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)
        self.l2norm = nn.BatchNorm2d(dim_out)  # Normalize(2)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.l2norm(x)
        return x

# ICKD
class ICKDLoss(nn.Module):
    """Inter-Channel Correlation"""
    def __init__(self, s_dim=64, t_dim=None):
        super(ICKDLoss, self).__init__()
        # if feat_dim is None:
        #     feat_dim = s_dim
        self.embed_s = Embed(s_dim, t_dim)
        # self.embed_t = Embed(s_dim, feat_dim)

    def forward(self, g_s, g_t):
        # loss = [self.batch_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]
        loss = self.batch_loss(g_s, g_t)
        return loss

    def batch_loss(self, f_s, f_t):

        f_s = self.embed_s(f_s)

        bsz, ch = f_s.size(0), f_s.size(1)

        f_s = f_s.view(bsz, ch, -1)
        f_t = f_t.view(bsz, ch, -1)

        emd_s = torch.bmm(f_s, f_s.permute(0, 2, 1))
        emd_s = torch.nn.functional.normalize(emd_s, dim=2)

        emd_t = torch.bmm(f_t, f_t.permute(0, 2, 1))
        emd_t = torch.nn.functional.normalize(emd_t, dim=2)

        G_diff = emd_s - emd_t
        loss = (G_diff * G_diff).view(bsz, -1).sum() / (ch * bsz)
        return loss

# SP
class Similarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s, g_t):
        # return [self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]
        return self.similarity_loss(g_s, g_t)

    def similarity_loss(self, f_s, f_t):

        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss




def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is None:
                    continue
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue
    return net




@measure('diswot', bn=True)
def compute_diswot_procedure(net,
    inputs,
    targets,
    loss_fn=None,
    split_data=1,
    repeat=1,
    mixup_gamma=1e-2,
    fp16=False,
):
    gt_list = []
    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_sp = Similarity()
    # criterion_sp= similarity_transform()


    tnet = net.teacher_model 
    snet = net.student_model

    network_weight_gaussian_init(snet)
    network_weight_gaussian_init(tnet)


    # tfeature, tlogits = tnet.forward_with_features(inputs)
    # sfeature, slogits = snet.forward_with_features(inputs)

    #----------------------------------------------------------------------

    # print('input size:', inputs.size())
    slogits = net.student_model(inputs)
    feats_s = net.student_model.features[-1]



    inputs = F.interpolate(inputs, size=(net.teacher_img_size, net.teacher_img_size), mode='bilinear',
                           align_corners=False)
    tlogits = net.teacher_model(inputs)
    feats_t = net.teacher_model.features[-1]




    dsize = (max(feats_t.size(-2), feats_s.size(-2)), max(feats_t.size(-1), feats_s.size(-1)))
    tfeature = F.interpolate(feats_t, dsize, mode='bilinear', align_corners=False)
    sfeature = F.interpolate(feats_s, dsize, mode='bilinear', align_corners=False)


    criterion_ickd = ICKDLoss(s_dim=tfeature.size(1), t_dim=sfeature.size(1)).cuda()


    criterion_ce(tlogits, targets).backward()
    criterion_ce(slogits, targets).backward()

    tcompressed = tnet.head.fc.weight.grad.unsqueeze(-1).unsqueeze(-1)
    scompressed = snet.head.weight.grad.unsqueeze(-1).unsqueeze(-1)



    # score_sp = -1 * criterion_sp(tfeature,
    #                                 sfeature)[0].detach().cpu().numpy()
    #
    # score_ickd = -1 * criterion_ickd([tcompressed],
    #                                     [scompressed])[0].detach().cpu().numpy()
    score_sp = -1 * criterion_sp(tfeature, sfeature)[0].detach()

    score_ickd = -1 * criterion_ickd(tcompressed, scompressed).detach()

    print('score_sp is:', score_sp)
    print('score_ickd is:', score_ickd)

    result = score_ickd.cpu().numpy() + score_sp.cpu().numpy()
    return result 

