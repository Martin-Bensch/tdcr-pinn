import torch
from robot_specs import CCRT
import numpy as np
from pinn_nn import DEVICE


def hat_mul(uR):
    """
    Compute the skew symmetric representation of the given tensor
    Args:
        uR: tensor (N,3,1)

    Returns:
        uR_skew: tensor (-1, 3, 3)
    """
    # Create generators
    ez = torch.FloatTensor([0,-1,0,1,0,0,0,0,0]).reshape((1,1,3,3))
    ey = torch.FloatTensor([0,0,1,0,0,0,-1,0,0]).reshape((1,1,3,3))
    ex = torch.FloatTensor([0,0,0,0,0,-1,0,1,0]).reshape((1,1,3,3))
    exyz = torch.cat([ex, ey, ez], dim=1)
    Exyz = torch.cat([exyz] * uR.shape[0], dim=0)

    uRN311 = uR.reshape((-1, 3, 1, 1))

    uRhat_ = uRN311 * Exyz.to(DEVICE)

    uRhat = torch.sum(uRhat_, dim=1)
    return uRhat


def compute_mn(R, u, v):
    vstar = torch.tensor([0, 0, 1]).reshape(1, 3, 1)
    n = R @ CCRT.Kse.detach().clone() @ (v - vstar)
    m = R @ CCRT.Kbt.detach().clone() @ u

    return m, n


def compute_r1():
    _r1 = CCRT.r1.detach().clone()
    r11 = torch.Tensor([0, _r1, 0]).reshape((1, 3, 1))
    r12 = torch.Tensor([_r1 * np.cos(-torch.pi / 6),
                        _r1 * np.sin(-torch.pi / 6),
                        0]).reshape((1, 3, 1))
    r12.requires_grad = False
    r13 = torch.Tensor([_r1 * np.cos(7 * torch.pi / 6),
                        _r1 * np.sin(7 * torch.pi / 6),
                        0]).reshape((1, 3, 1))
    r13.requires_grad = False
    r1 = [r11, r12, r13]

    return r1


def compute_p_bi_d(u, v, r):
    pi_d_lst = []
    for ri in r:
        pi_d = torch.cross(u, ri, 1) + v
        pi_d_lst.append(pi_d)

    return pi_d_lst


def compute_p_bi_dd(u_hat, u_d, v_d, r, p_b_d,):
    pi_d_lst = []
    for ri, pi_d in zip(r, p_b_d):
        pi_dd =  u_hat @ pi_d + torch.cross(u_d, ri, 1) + v_d
        pi_d_lst.append(pi_dd)
    return pi_d_lst


def compute_Ai_sum(tau, p_b_d):
    Ai_sum = torch.zeros((p_b_d[0].shape[0], 3, 3))
    Ai = []
    for idx, pi_d in enumerate(p_b_d):
        p_bi_d_norm = torch.norm(pi_d, dim=1).pow(3).reshape(-1,1,1)
        pi_d_hat = hat_mul(pi_d)
        t = tau[:, idx].reshape(-1,1,1)
        ai = -t * pi_d_hat @ pi_d_hat / p_bi_d_norm
        Ai.append(ai)
        Ai_sum = Ai_sum + ai

    return Ai_sum, Ai


def compute_Bi_sum(Ai_, r_):
    Bi_sum = torch.zeros_like(Ai_[0])
    Bi = []
    for Ai, ri in zip(Ai_, r_):
        ri_hat = hat_mul(ri)
        Bi.append(ri_hat @ Ai)
        Bi_sum = Bi_sum + Bi[-1]

    return Bi_sum, Bi


def compute_G(Ai_, ri_):
    G = torch.zeros_like(Ai_[0])
    for Ai, ri in zip(Ai_, ri_):
        ri_hat = hat_mul(ri)
        G = G - Ai @ ri_hat
    return G

def compute_H(Bi_, ri_):
    H = torch.zeros_like(Bi_[0])
    for Bi, ri in zip(Bi_, ri_):
        ri_hat = hat_mul(ri)
        H = H - Bi @ ri_hat
    return H


def compute_ai_sum(Ai_, u_hat, p_b_d):
    ai_ = []
    a = torch.zeros_like(p_b_d[0])
    for Ai, pi_d in zip(Ai_, p_b_d):
        ai = Ai @ u_hat @ pi_d
        a = a + ai
        ai_.append(ai)
    return a, ai_


def compute_bi_sum(r_, ai_):
    bi_ = []
    b = torch.zeros_like(ai_[0])
    for ri, ai in zip(r_, ai_):
        bi = torch.cross(ri, ai, 1)
        b = b + bi
        bi_.append(bi)
    return b, bi_


def rodrigues_uR(uR):
    ## Rodrigues formula
    # Magnitude of uR
    theta = torch.norm(uR, dim=1).reshape(-1, 1, 1).to(DEVICE)
    uR_norm = uR / theta
    # u hat representation
    uR_hat = hat_mul(uR_norm)
    uR_hat_sq = uR_hat[:] @ uR_hat[:]

    eyeN = torch.cat([torch.eye(3)] * uR.shape[0]).reshape(-1, 3, 3).to(DEVICE)
    sintheta = torch.sin(theta).reshape(-1, 1, 1).to(DEVICE)
    costheta = 1 - torch.cos(theta).reshape(-1, 1, 1).to(DEVICE)
    R = eyeN + sintheta * uR_hat + costheta * uR_hat_sq

    return R


def compute_fL_lL_vconst(sL_act, vL, rL, RL):
    fLb = torch.zeros_like(vL)
    mLb = torch.zeros_like(fLb)
    for idx, riL in enumerate(rL):
        tau = sL_act[:, idx+1].reshape(-1, 1, 1)
        fi_b = -tau * vL
        mi_b = torch.cross(riL, fi_b, 1)
        fLb = fLb + fi_b
        mLb = mLb + mi_b

    fL =  RL @ fLb
    mL = RL @ mLb

    return fL, mL
