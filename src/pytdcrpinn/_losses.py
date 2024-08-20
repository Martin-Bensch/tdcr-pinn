import gc
import torch
import _helper_fn as hf
import _losses_fn as lf
import _physics_fn as pf
import pinn_nn as pnn
from tdcrpinn.icra2024.robot_specs import CCRT

def compute_loss(
        nn: pnn.NNApproximator, s_act: torch.Tensor = None,
        i0=None,
        bv_sL=None,
        bv_data=None,
        bv_tau_max=None,
        bv_tau_min=None,
        ) -> torch.float:

    s_act = s_act.detach().clone()

    pd_loss, Rd_loss, udot_loss, vdot_loss = compute_p_loss(nn=nn,
                                                           s_act_=s_act,
                                                            )

    uR0_loss, p0_loss = compute_bv_s0_collocation_points_loss(nn=nn,
                                                              s_act=s_act)
    mL_loss, nL_loss = compute_b_loss_sL_collocation_points(
        nn=nn, s_act=s_act)

    # Introduces Fext into the physics collocation loss
    u0_loss = compute_b_loss_sL_collocation_points_u0(nn=nn, s_act=s_act)

    # Boundary values at s=L, from reference model
    bRLtau_loss, v_L_loss, u_L_loss, bpLtau_loss = compute_bv_sL_loss(
        nn=nn, boundary_sL=bv_sL)

    # Boundary values for tau_max, from reference model
    bptau_max_loss, bRtau_max_loss, \
    v_taumax_loss, u_taumax_loss = compute_b_tau_max_loss(nn=nn,
                                                boundary_tau_max=bv_tau_max)

    # Boundary values for tau_min, from reference model
    bptau_min_loss, bRtau_min_loss, \
    v_taumin_loss, u_taumin_loss = compute_b_tau_min_loss(nn=nn,
                                                boundary_tau_min = bv_tau_min)

    # Intial values at s = 0, from reference model
    i0_p_loss, i0_uR_loss, i0_v_loss, i0_u_loss = compute_b_s0_loss(nn=nn,
                                                                i0_values=i0)
    # Data loss s, tau, from boundary calculation s=L
    sLdata_p_loss, sLdata_R_loss, \
    sLdata_v_loss, sLdata_u_loss = compute_b_sLdata_loss(nn=nn,
                                                     sLdata_values=bv_data)

    arc_length_loss = compute_arc_length_loss(nn=nn, s_act=s_act)

    # Manually weigh losses
    losses_dict = {
                    "pd_loss": 1e4 * pd_loss,
                    "Rd_loss": 1e3 * Rd_loss,
                    "udot_loss": udot_loss,
                    "vdot_loss": 1e7 * vdot_loss,
                    "uR0_loss": 1e3 * uR0_loss,
                    "p0_loss": 1e7 * p0_loss,
                    "mL_loss": 1e8 * mL_loss,
                    "u0_loss": u0_loss,
                    "bRLtau_loss": 1e1 * bRLtau_loss,
                    "u_L_loss": u_L_loss,
                    "bpLtau_loss": 1e6 * bpLtau_loss,
                    "bptau_max_loss": 1e6 * bptau_max_loss,
                    "bRtau_max_loss":  1e1 * bRtau_max_loss,
                    "u_taumax_loss": u_taumax_loss,
                    "bptau_min_loss": 1e7 * bptau_min_loss,
                    "bRtau_min_loss":  1e1 * bRtau_min_loss,
                    "u_taumin_loss": 1e1 * u_taumin_loss,
                    "i0_p_loss": 1e7 * i0_p_loss,
                    "i0_uR_loss": 1e3 * i0_uR_loss,
                    "i0_u_loss": i0_u_loss,
                    "sLdata_p_loss": 1e7 * sLdata_p_loss,
                    "sLdata_R_loss":1e2 * sLdata_R_loss,
                    "sLdata_u_loss": sLdata_u_loss,
                   "arc_length_loss": 1e2 * arc_length_loss

                     }

    del pd_loss, Rd_loss, udot_loss, vdot_loss, uR0_loss, p0_loss, mL_loss,\
        nL_loss, bRLtau_loss, v_L_loss, u_L_loss, bpLtau_loss, \
        bptau_max_loss, bRtau_max_loss, v_taumax_loss, u_taumax_loss, \
        bptau_min_loss, bRtau_min_loss, v_taumin_loss, u_taumin_loss, \
        i0_p_loss, i0_uR_loss, i0_v_loss, i0_u_loss, sLdata_p_loss, \
        sLdata_R_loss, sLdata_v_loss, sLdata_u_loss, arc_length_loss

    gc.collect()

    return losses_dict


def compute_b_sLdata_loss(nn: pnn.NNApproximator, sLdata_values=None)\
                    -> (torch.float, torch.float, torch.float, torch.float):
    s_act, pRvus_sLdata = sLdata_values
    psLdata_ref, RsLdata_ref, \
    vsLdata_ref, usLdata_ref, ssLdata_ref = lf.split_state_vector(pRvus_sLdata)

    nn_pRuv_sLdata = pnn.f(nn, s_act).reshape(-1, nn.out_n, 1)
    p_sLdata = nn_pRuv_sLdata[:, 0:3]
    uR_sLdata = nn_pRuv_sLdata[:, 3:6]
    R_sLdata = pf.rodrigues_uR(uR_sLdata)
    u_sLdata = nn_pRuv_sLdata[:, 6:9]

    v_sLdata = torch.tensor([0, 0, 1]).reshape(1,3,1) * \
               torch.ones((u_sLdata.shape[0],3,1))
    vsLdata_ref = v_sLdata

    dp_sLdata = torch.mean((psLdata_ref - p_sLdata).pow(2), dim=0,
                      keepdim=True)
    p_sLdata_loss = torch.mean(dp_sLdata)

    R_sLdata_loss = lf.compute_SO3_loss_liealg(RsLdata_ref, R_sLdata)
    v_sLdata_loss, u_sLdata_loss = lf.compute_uv_loss(vsLdata_ref,
                                                      usLdata_ref,
                                                      v_sLdata,
                                                      u_sLdata)

    return p_sLdata_loss, R_sLdata_loss, v_sLdata_loss, u_sLdata_loss


def compute_b_s0_loss(nn: pnn.NNApproximator, i0_values=None)\
                    -> (torch.float, torch.float, torch.float, torch.float):

    s_act, pRvus_0 = i0_values
    p0_ref, R0_ref, \
    v0_ref, u0_ref, s0_ref = lf.split_state_vector(pRvus_0)

    nn_pRuv_0 = pnn.f(nn, s_act).reshape(-1, nn.out_n, 1)
    p_0 = nn_pRuv_0[:, 0:3]
    uR_0 = nn_pRuv_0[:, 3:6]
    u_0 = nn_pRuv_0[:, 6:9]

    v_0 = torch.tensor([0, 0, 1]).reshape(1, 3, 1) * \
          torch.ones((u_0.shape[0], 3, 1))
    v0_ref = v_0

    dp_0= torch.mean((p0_ref - p_0).pow(2), dim=0,
                            keepdim=True)
    p_0_loss = torch.mean(dp_0)

    duR_0 = torch.mean((uR_0).pow(2),
                           dim=0, keepdim=True)
    uR_0_loss = torch.mean(duR_0,)

    v_0_loss, u_0_loss = lf.compute_uv_loss(v0_ref, u0_ref, v_0, u_0)


    return p_0_loss, uR_0_loss, v_0_loss, u_0_loss


def compute_p_loss(nn: pnn.NNApproximator,
                   s_act_: torch.Tensor = None)\
                      -> (torch.float, torch.float, torch.float, torch.float):
    """
    Compute the full physics loss function as interior loss.
    """

    # Evaluate ANN
    s_act = s_act_.detach().clone()
    pRuv = pnn.f(nn, s_act)
    uR = pRuv[:, 3:6].reshape(-1, 3, 1)
    R = pf.rodrigues_uR(uR).reshape(-1, 3, 3)
    u = pRuv[:, 6:9].reshape(-1, 3, 1)
    v = torch.tensor([0, 0, 1]).reshape(1, 3, 1) * \
              torch.ones((u.shape[0], 3, 1))
    tau = s_act[:, 1:]
    u_hat = pf.hat_mul(u)
    r1 = pf.compute_r1()

    # Use the same predicted values but with no reference to s_act_
    # Also returns the input vector, which is detached from s_act
    pRuv_d, pRuv_df, s_mL_df = pnn.df_mout(s_act, nn)
    uR_df = pRuv_df[:, 3:6].reshape(-1, 3, 1)
    R_df = pf.rodrigues_uR(uR_df).reshape(-1, 3, 3)

    p_d = pRuv_d[:, 0:3].reshape(-1, 3, 1)
    u_d = pRuv_d[:, 6:9].reshape(-1, 3, 1)

    v_d = torch.tensor([0, 0, 0]).reshape(1, 3, 1) * \
              torch.ones((u.shape[0], 3, 1))
    R_d = pnn.df_R(R_df.reshape(-1, 9, 1), s_mL_df).reshape(-1, 3, 3)

    p_b_d = pf.compute_p_bi_d(u, v, r1)
    A, Ai = pf.compute_Ai_sum(tau, p_b_d)
    B, Bi = pf.compute_Bi_sum(Ai, r1)
    G = pf.compute_G(Ai, r1)
    H = pf.compute_H(Bi, r1)
    a, ai = pf.compute_ai_sum(Ai, u_hat, p_b_d)
    b, bi = pf.compute_bi_sum(r1, ai)

    c, d = lf.compute_cd(u, v, a, b)
    udot_loss, vdot_loss = lf.uv_d_loss(u_d, v_d, d, c, A, G, B, H)

    pd_loss = lf.compute_pd_loss(p_d, R, v)
    Rd_loss = lf.compute_Rd_loss(R_d, R, u_hat)

    return pd_loss, Rd_loss, udot_loss, vdot_loss


def compute_arc_length_loss(nn: pnn.NNApproximator, s_act: torch.Tensor=None)\
    -> torch.float:

    def compute_arc_length(p):
        # Compute the Euclidean distance for all successive points along the
        # backbone
        p0Lm1 = p[:, 0:-1]
        p1L = p[:, 1:]
        Dp = p1L - p0Lm1
        Dp_dists = torch.norm(Dp, dim=2)
        sum_Dp = torch.sum(Dp_dists, dim=1)

        return sum_Dp

    # Compute length loss for all actuations
    s0_arc_len = torch.linspace(0, float(CCRT.L), 200)
    s0L_tau_ = []
    s_act = s_act.detach().clone()
    for idx in range(s_act.shape[0]):
        act = s_act[idx, 1:].reshape(-1, 3, 1) * torch.ones((len(
            s0_arc_len), 1, 1))
        s0L_act = torch.cat((s0_arc_len.reshape(-1, 1, 1), act), dim=1)
        s0L_tau_.append(s0L_act)

    s0L_tau = torch.cat(s0L_tau_, dim=0).reshape(-1, 4)
    pRmn_arc_len = pnn.f(nn, s0L_tau)[:, 0:3].reshape(-1, s0_arc_len.shape[
                                                                    0], 3, 1)

    arc_length = compute_arc_length(pRmn_arc_len)
    darc_length = arc_length - CCRT.L.detach().clone() * torch.ones_like(
        arc_length)
    darc_length_loss = torch.mean(darc_length.pow(2))

    return darc_length_loss

def compute_bv_s0_collocation_points_loss(nn: pnn.NNApproximator,
                                         s_act: torch.Tensor = None) \
        -> (torch.float, torch.float):
    s0_act, _ = hf.compute_s0_sL_act(s_act.detach().clone())
    pRuv_0 = pnn.f(nn, s0_act)
    p0 = pRuv_0[:, 0:3]
    uR0 = pRuv_0[:, 3:6]

    dR0_loss, dp0_loss = lf.compute_b0_loss(uR0, p0)

    return dR0_loss, dp0_loss


def compute_b_loss_sL_collocation_points_u0(nn: pnn.NNApproximator,
                                         s_act: torch.Tensor = None) \
        -> torch.float:
    s0_act, sL_act = hf.compute_s0_sL_act(s_act.detach().clone())
    pRuv_L = pnn.f(nn, sL_act)
    pL = pRuv_L[:, :3].reshape(-1, 3, 1)
    uRL = pRuv_L[:, 3:6].reshape(-1, 3, 1)
    RL = pf.rodrigues_uR(uRL)
    uL = pRuv_L[:, 6:9].reshape(-1, 3, 1)

    vL = torch.tensor([0, 0, 1]).reshape(1, 3, 1) * \
             torch.ones((uL.shape[0], 3, 1))

    pRuv_0 = pnn.f(nn, s0_act)
    u0_nn = pRuv_0[:, 6:9].reshape(-1, 3, 1)

    rL = pf.compute_r1()
    # global frame
    _, lL = pf.compute_fL_lL_vconst(sL_act, vL, rL, RL)

    # Compute moment w.r.t origin due to external forces
    Fext = CCRT.f_ext.detach().clone()
    m0 = lL + torch.cross(pL, Fext, 1)
    u0 = CCRT.Kbt_inv.detach().clone() @ m0
    du0 = u0 - u0_nn
    du0_sq = du0.pow(2)
    du0_sq_mean_batch = torch.mean(du0_sq, 0, keepdim=True)
    du0_loss = torch.mean(du0_sq_mean_batch)

    return du0_loss

def compute_b_loss_sL_collocation_points(
        nn: pnn.NNApproximator, s_act: torch.Tensor = None) \
        -> (torch.float, torch.float):
    # Since we assume no friction, we know the m,n values for each
    # collocation point. But these depend on the current shape, hence they
    # are no boundary value. It is more a constraint, that incorporates the
    # robots shape. This originates from the choice of coordinate system, m,
    # n are described in.
    _, sL_act = hf.compute_s0_sL_act(s_act.detach().clone())

    pRuv_L = pnn.f(nn, sL_act)
    uRL = pRuv_L[:, 3:6].reshape(-1, 3, 1)
    RL = pf.rodrigues_uR(uRL)
    uL = pRuv_L[:, 6:9].reshape(-1, 3, 1)
    vL = torch.tensor([0, 0, 1]).reshape(1, 3, 1) * \
          torch.ones((uL.shape[0], 3, 1))

    rL = pf.compute_r1()
    mL, nL = pf.compute_mn(RL, uL, vL) # uses constitutive law!

    fL, lL = pf.compute_fL_lL_vconst(sL_act, vL, rL, RL)
    
    dmL_loss, dnL_loss = lf.compute_bL1_loss(mL, lL, nL, fL)

    return dmL_loss, dnL_loss


def compute_bv_sL_loss(
        nn: pnn.NNApproximator, boundary_sL=None) -> \
        (torch.float, torch.float, torch.float, torch.float, torch.float):
    # Initial and boundary losses for full PINN optimization
    # Initial boundary values for collocation actuation are already
    # accounted for

    # Boundary losses at L
    s_act, pRvus_L = boundary_sL[:]
    pL_ref, RL_ref, vL_ref, uL_ref, sL_ref = lf.split_state_vector(pRvus_L)

    pRuv_L_tau = pnn.f(nn, s_act).reshape(-1, nn.out_n, 1)
    p_L_tau = pRuv_L_tau[:, 0:3]
    uR_L_tau = pRuv_L_tau[:, 3:6]
    R_L_tau = pf.rodrigues_uR(uR_L_tau)
    u_L_tau = pRuv_L_tau[:, 6:9]

    v_L_tau = torch.tensor([0, 0, 1]).reshape(1, 3, 1) * \
          torch.ones((u_L_tau.shape[0], 3, 1))
    vL_ref = v_L_tau

    v_L_loss, u_L_loss = lf.compute_uv_loss(vL_ref, uL_ref, v_L_tau, u_L_tau)

    # Compute differences
    dpLtau = torch.mean((p_L_tau - pL_ref).pow(2), dim=0, keepdim=True)
    bpLtau_loss = torch.mean(dpLtau)

    bRLtau_loss = lf.compute_SO3_loss_liealg(R_L_tau, RL_ref)

    return bRLtau_loss, None, u_L_loss, bpLtau_loss

def compute_b_tau_max_loss(nn: pnn.NNApproximator, boundary_tau_max=None) \
        -> (torch.float, torch.float, torch.float, torch.float):
    # Boundary losses tau basis
    s_act, pRvus_tau_max = boundary_tau_max
    ptaumax_ref, Rtaumax_ref, \
    vtaumax_ref, utaumax_ref, staumax_ref = lf.split_state_vector(
                                                                pRvus_tau_max)

    nn_pRuv_tau_max = pnn.f(nn, s_act).reshape(-1, nn.out_n, 1)
    p_taumax_tau = nn_pRuv_tau_max[:, 0:3]
    uR_taumax_tau = nn_pRuv_tau_max[:, 3:6]
    R_taumax_tau = pf.rodrigues_uR(uR_taumax_tau)
    u_taumax_tau = nn_pRuv_tau_max[:, 6:9]
    v_taumax_tau = torch.tensor([0, 0, 1]).reshape(1, 3, 1) * \
          torch.ones((u_taumax_tau.shape[0], 3, 1))
    vtaumax_ref = v_taumax_tau

    dp_tau_max = torch.mean((ptaumax_ref - p_taumax_tau).pow(2), dim=0,
                            keepdim=True)
    bptau_max_loss = torch.mean(dp_tau_max)

    bRtau_max_loss = lf.compute_SO3_loss_liealg(R_taumax_tau, Rtaumax_ref)

    v_taumax_loss, u_taumax_loss = lf.compute_uv_loss(vtaumax_ref,
                                                      utaumax_ref,
                                            v_taumax_tau, u_taumax_tau)

    return bptau_max_loss, bRtau_max_loss, v_taumax_loss, u_taumax_loss


def compute_b_tau_min_loss(nn: pnn.NNApproximator, boundary_tau_min=None) \
        -> (torch.float, torch.float, torch.float, torch.float):
    # Boundary losses tau basis
    s_act, pRvus_tau_min = boundary_tau_min
    ptaumin_ref, Rtaumin_ref, \
    vtaumin_ref, utaumin_ref, staumin_ref = lf.split_state_vector(
                                                                pRvus_tau_min)

    nn_pRuv_tau_min = pnn.f(nn, s_act).reshape(-1, nn.out_n, 1)
    p_taumin_tau = nn_pRuv_tau_min[:, 0:3]
    uR_taumin_tau = nn_pRuv_tau_min[:, 3:6]
    R_taumin_tau = pf.rodrigues_uR(uR_taumin_tau)
    u_taumin_tau = nn_pRuv_tau_min[:, 6:9]

    v_taumin_tau = torch.tensor([0, 0, 1]).reshape(1, 3, 1) * \
          torch.ones((u_taumin_tau.shape[0], 3, 1))
    vtaumin_ref = v_taumin_tau

    dp_tau_min = torch.mean((ptaumin_ref - p_taumin_tau).pow(2), dim=0,
                            keepdim=True)
    bptau_min_loss = torch.mean(dp_tau_min)

    bRtau_min_loss = lf.compute_SO3_loss_liealg(R_taumin_tau, Rtaumin_ref)

    v_taumin_loss, u_taumin_loss = lf.compute_uv_loss(vtaumin_ref,
                                                      utaumin_ref,
                                            v_taumin_tau, u_taumin_tau)

    return bptau_min_loss, bRtau_min_loss, v_taumin_loss, u_taumin_loss

