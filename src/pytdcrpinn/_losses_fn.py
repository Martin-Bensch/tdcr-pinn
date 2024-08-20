import torch
from robot_specs import CCRT
from _physics_fn import hat_mul, rodrigues_uR


def compute_pd_loss(p_d, R, v):
    dp_d = p_d - R @ v
    dp_d_sq = dp_d.pow(2)
    dp_mean_batch = torch.mean(dp_d_sq, 0, keepdim=True)
    dp_loss = torch.mean(dp_mean_batch)
    return dp_loss


def compute_Rd_loss(R_d, R, u_hat):

    dR_d = R_d - R @ u_hat
    dR_d_sq = dR_d.pow(2)
    dR_mean_batch = torch.mean(dR_d_sq, 0, keepdim=True)
    dR_loss = torch.sum(dR_mean_batch) / 9
    return dR_loss

def compute_cd(u, v, a, b):

    u_hat = hat_mul(u)
    v_hat = hat_mul(v)
    vstar = torch.tensor([0, 0, 1]).reshape(1, 3, 1)
    uKbtU = u_hat @ CCRT.Kbt.detach().clone() @ u
    vKsev = v_hat @ CCRT.Kse.detach().clone() @ (v - vstar)
    c = (-uKbtU - vKsev - b)
    d = (-u_hat @ CCRT.Kse.detach().clone() @ (v - vstar) - a)

    return c, d


def uv_d_loss(u_d, v_d, d, c, A, G, B, H):

    state_mat = torch.zeros((A.shape[0], 6, 6))
    state_mat[:, :3, :3] = CCRT.Kse.detach().clone() + A
    state_mat[:, :3, 3:] = G
    state_mat[:, 3:, :3] = B
    state_mat[:, 3:, 3:] = CCRT.Kbt.detach().clone() + H
    state_mat_inv = torch.inverse(state_mat)
    dvdot = v_d - state_mat_inv[:,:3,:3] @ d - state_mat_inv[:,:3, 3:] @ c
    dudot = u_d - state_mat_inv[:, 3:, :3] @ d - state_mat_inv[:, 3:, 3:] @ c

    dvdot_sq = dvdot.pow(2)
    dvdot_mean_batch = torch.mean(dvdot_sq, 0, keepdim=True)
    dvdot_loss = torch.mean(dvdot_mean_batch)

    dudot_sq = dudot.pow(2)

    dudot_mean_batch = torch.mean(dudot_sq, 0, keepdim=True)
    dudot_loss = torch.mean(dudot_mean_batch)

    return dudot_loss, dvdot_loss


def compute_b0_loss(uR0, p0):

    duR0_d_sq = (uR0).pow(2)
    duR0_mean_batch = torch.mean(duR0_d_sq, 0, keepdim=True)
    duR0_loss = torch.mean(duR0_mean_batch)

    # p0 loss
    dp0_d_sq = (p0).pow(2)
    dp0_mean_batch = torch.mean(dp0_d_sq, 0, keepdim=True)
    dp0_loss = torch.mean(dp0_mean_batch)
    return duR0_loss, dp0_loss


def compute_bL1_loss(mL, lL, nL,fL):

    dmL_lL = mL - lL
    dml_d_sq = dmL_lL.pow(2)
    dmL_lL_mean_batch = torch.mean(dml_d_sq, 0, keepdim=True)
    dmL_lL_loss = torch.mean(dmL_lL_mean_batch)

    dnL_fL = (nL - fL - CCRT.f_ext.detach().clone())
    dnl_d_sq = dnL_fL.pow(2)
    dnL_fL_mean_batch = torch.mean(dnl_d_sq, 0, keepdim=True)
    dnL_fL_loss = torch.mean(dnL_fL_mean_batch)

    return dmL_lL_loss, dnL_fL_loss


def compute_uv_loss(v_ref, u_ref, v, u):

    dv_sq = (v_ref - v).pow(2)
    v_loss = torch.mean(torch.mean(dv_sq, dim=0, keepdim=True))
    du_sq = (u_ref - u).pow(2)
    u_loss = torch.mean(torch.mean(du_sq, dim=0, keepdim=True))

    return v_loss, u_loss

def split_state_vector(pRvus):

    p = pRvus[:, :3].reshape((-1, 3, 1))
    R = pRvus[:, 3:12].reshape((-1, 3,3))
    v = pRvus[:, 12:15].reshape((-1,3,1))
    u = pRvus[:, 15:18].reshape(-1,3,1)
    s = pRvus[:, 18].reshape((-1,1,1))

    return p,R,v,u,s


def compute_SO3_loss_liealg(Rref, Rnn):

    dR = Rref @ torch.transpose(Rnn, 1, 2)
    dRtrace = (dR.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1) / 2
    dRtrace_clamp = torch.clamp(dRtrace, -0.99999, 0.99999)
    theta_abs = torch.abs(torch.acos(dRtrace_clamp))

    R_loss = torch.mean(theta_abs)

    return R_loss