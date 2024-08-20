import torch
from robot_specs import CCRT


def compute_s0_sL_act(s_act):
    s0 = torch.zeros_like(s_act[:, 0]).reshape(-1, 1)
    s0_act = torch.cat((s0, s_act[:, 1:]), dim=1)
    sL = torch.ones_like(s0) * CCRT.L.detach().clone()
    sL_act = torch.cat((sL, s_act[:, 1:]), dim=1)

    return s0_act, sL_act


def compute_arc_length(p):
    p0Lm1 = p[:, 0:-1]
    p1L = p[:, 1:]
    Dp = p1L - p0Lm1
    Dp_dists = torch.norm(Dp, dim=2)
    sum_Dp = torch.sum(Dp_dists, dim=1)

    return sum_Dp


def add_individual_losses_in_epoch(losses_dict, individual_losses_per_epoch):

    for l in losses_dict:
        if l not in individual_losses_per_epoch:
            individual_losses_per_epoch[l] = float(losses_dict[l])
        else:
            individual_losses_per_epoch[l] += float(losses_dict[l])

    return individual_losses_per_epoch
