import math

import numpy as np
import torch
import torch.nn.functional as F
from ..p_utils import get_layer_metric_array
from . import measure





# advised by wzm
def attention_transform(feat):
    return F.normalize(feat.pow(2).mean(1).view(feat.size(0), -1))



def pdist(fm, squared=False, eps=1e-12):
    if len(fm.size())==4:
        fm = fm.view(fm.shape[0], -1)
    feat_square = fm.pow(2).sum(dim=1)
    feat_prod = torch.mm(fm, fm.t())
    feat_dist = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) -
                 2 * feat_prod).clamp(min=eps)
    if not squared:
        feat_dist = feat_dist.sqrt()
    feat_dist = feat_dist.clone()
    feat_dist[range(len(fm)), range(len(fm))] = 0

    return feat_dist


def similarity_transform(feat):
    feat = feat.view(feat.size(0), -1)
    gram = feat @ feat.t()
    return F.normalize(gram)


def nst(f_s, f_t):

    f_s = f_s.view(f_s.shape[0], f_s.shape[1], -1)
    f_s = F.normalize(f_s, dim=2)
    f_t = f_t.view(f_t.shape[0], f_t.shape[1], -1)
    f_t = F.normalize(f_t, dim=2)

    def poly_kernel(a, b):
        a = a.unsqueeze(1)
        b = b.unsqueeze(2)
        res = (a * b).sum(-1).pow(2)
        return res

    return poly_kernel(
        f_s, f_s).mean() - 2 * poly_kernel(f_s, f_t).mean()



def rkd_angle(fm):
    # N x C --> N x N x C
    if len(fm.size())==4:
        fm = fm.view(fm.shape[0], -1)
    feat_t_vd = (fm.unsqueeze(0) - fm.unsqueeze(1))
    norm_feat_t_vd = F.normalize(feat_t_vd, p=2, dim=2)
    feat_t_angle = torch.bmm(norm_feat_t_vd,
                             norm_feat_t_vd.transpose(1, 2)).view(-1)
    return feat_t_angle


def cc(fm):
    P_order = 2
    gamma = 0.4
    if len(fm.size())==4:
        fm = fm.view(fm.shape[0], -1)
    fm = F.normalize(fm, p=2, dim=-1)
    sim_mat = torch.matmul(fm, fm.t())
    corr_mat = torch.zeros_like(sim_mat)
    for p in range(P_order + 1):
        corr_mat += math.exp(-2 * gamma) * (2 * gamma) ** p / \
            math.factorial(p) * torch.pow(sim_mat, p)
    return corr_mat




def ickd(fm):
    bsz, ch = fm.shape[0], fm.shape[1]
    fm = fm.view(bsz, ch, -1)
    emd_s = torch.bmm(fm, fm.permute(0, 2, 1))
    emd_s = torch.nn.functional.normalize(emd_s, dim=2)

    G_diff = emd_s
    loss = (G_diff * G_diff).view(bsz, -1).sum() / (ch * bsz)
    return loss





def inter_distill_loss(feat_t, feat_s, transform_type):
    assert transform_type in _TRANS_FUNC, f"Transformation function {transform_type} is not supported."
    trans_func = _TRANS_FUNC[transform_type]
    feat_t = trans_func(feat_t)
    feat_s = trans_func(feat_s)
    return (feat_t - feat_s).pow(2).mean()



def get_l2_norm_array(net, mode):
    return get_layer_metric_array(net, lambda l: l.weight.norm(), mode=mode)





_TRANS_FUNC = {"attention": attention_transform, "similarity": similarity_transform, "linear": lambda x: x, 'nst': nst, 'ickd': ickd, 'pdist': pdist, 'rkd_angle': rkd_angle, 'cc':cc, 'ickd': ickd}


@measure('tvt', bn=True)
def compute_nst_score(
    net,
    inputs,
    targets,
    loss_fn=None,
    split_data=1,
    repeat=1,
    mixup_gamma=1e-2,
    fp16=False,
):
    nas_score_list = []

    with torch.no_grad():
        # output, logits = net.forward_with_features(inputs)
        net(inputs)
        # logits_t = net.teacher_model(inputs)
        # logits_s = net.student_model.distill_logits
        logits_s = net.student_model.forward_features(inputs)
        # feats_s = net.student_model.forward_features(inputs)
        feats_s = net.student_model.features[-1]
        # feats_s_group = net.student_model.features


        inputs = F.interpolate(inputs, size=(net.teacher_img_size, net.teacher_img_size), mode='bilinear',
                          align_corners=False)
        logits_t = net.teacher_model(inputs)
        feats_t = net.teacher_model.features[-1]
        # feats_t_group = net.teacher_model.features

        dsize = (max(feats_t.size(-2), feats_s.size(-2)), max(feats_t.size(-1), feats_s.size(-1)))
        feats_t = F.interpolate(feats_t, dsize, mode='bilinear', align_corners=False)
        feats_s = F.interpolate(feats_s, dsize, mode='bilinear', align_corners=False)


        # feats_t = net.teacher_model.features[-1]
        outputs = []

        # print('student logit size:', logits_s.size())
        # print('teacher logit size:', logits_t.size())
        #
        # print('student feature size:', feats_s.size())
        # print('teacher feature size:', feats_t.size())

        # for i, (idx_t, idx_s) in enumerate(zip(net.teacher_idx, net.student_idx)):
        #     feat_t_temp = feats_t_group[idx_t]
        #     feat_s_temp = feats_s_group[idx_s]
        #     dsize = (max(feat_t_temp.size(-2), feat_s_temp.size(-2)), max(feat_t_temp.size(-1), feat_s_temp.size(-1)))
        #     feat_t_temp = F.interpolate(feat_t_temp, dsize, mode='bilinear', align_corners=False)
        #     feat_s_temp = F.interpolate(feat_s_temp, dsize, mode='bilinear', align_corners=False)
        #     print('student feature size:', feat_s_temp.size())
        #     print('teacher feature size:', feat_t_temp.size())
        #     outputs.append(nst(feat_s_temp, feat_t_temp))



        t_score = inter_distill_loss(feats_s, feats_t, transform_type="attention")
        s_score = sum(get_l2_norm_array(net.student_model, 'param'))

        # score = ickd(feats_s, feats_t)
        # outputs.append()
        # for f in outputs:
        #     print('size is:', f.size())
        # # for idx in [0, 1, 2]:
        #     outputs.append(net.features[idx])

        # nas_score_list = [
        #     torch.sum(f.detach().numpy()) for f in outputs
        # ]
        # # nas_score_list = [
        # #     torch.sum(single_kd(f)).detach().numpy() for f in output[1:-1]
        # # ]
        #
        # avg_nas_score = float(sum(outputs)/len(outputs))

    return -t_score, s_score
    # return -avg_nas_score