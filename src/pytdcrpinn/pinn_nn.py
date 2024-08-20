import torch
from torch import nn
from tdcrpinn.icra2024.robot_specs import CCRT, CCRnp
import numpy as np

torch.autograd.set_detect_anomaly(True)
dev = "cpu"
DEVICE = torch.device(dev)
CCRTdev = CCRT().device(DEVICE)
S_COMPUTE_LOSS = None

class NNApproximator(nn.Module):
    def __init__(self, num_hidden: int, dim_hidden: int,
                 act=nn.Tanh(),
                 max_tau=2.4):

        super().__init__()
        self.f_ext = CCRT.f_ext.detach().clone()
        self.layer_in = nn.Linear(3, dim_hidden)

        # and m as output
        self.layer_out = nn.Linear(dim_hidden, 9)
        self.out_n = 9

        layers = []
        for idx in range(num_hidden): # 2 * num_middle - 2 due to BN
                layers.append(nn.Linear(dim_hidden, dim_hidden))

        self.middle_layers = nn.ModuleList(layers)
        self.act = act
        self.act_input = nn.SiLU()
        self._output_names = ["p", "R,", "u"]
        self.max_tau = float(max_tau)


    @property
    def output_names(self):
        return self._output_names

    def forward(self, s):
        # normalize
        L = float(CCRT.L)

        s_ = (s[:,0] / L)
        act_ = (s[:, 1:3] / self.max_tau)
        s_act = torch.cat([s_.reshape(-1,1), act_], dim=1)

        out = self.act_input(self.layer_in(s_act))
        for layer in self.middle_layers:
            out = self.act(layer(out))


        return self.layer_out(out)


def f(nn: NNApproximator, s: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    return nn(s)


def df(nn: NNApproximator, s: torch.Tensor = None, order: int = 1,
       ) -> torch.Tensor:
    """
    Compute neural network derivative with respect to input features using
    PyTorch autograd engine
    """

    df_value = f(nn, s)
    for _ in range(order):
        df_value = torch.autograd.grad(
            outputs=df_value,
            inputs=s,
            grad_outputs=torch.ones_like(df_value),
            create_graph=True,
            retain_graph=True,
        )[0]

    return df_value


def df_mout(s_mL: torch.Tensor = None, nn=None) -> \
        torch.Tensor:
    """
    Compute neural network derivative with respect to input features using
    PyTorch autograd engine
    """
    s_mL = s_mL.detach()
    s_mL.requires_grad = True
    s_mL = s_mL[:,:3]
    nn.eval()
    pRuv_df = f(nn, s_mL)
    dudx_ = []

    eye_n = 9
    for idx, ev in enumerate(torch.eye(eye_n)):

        dudx, = torch.autograd.grad(outputs=pRuv_df[:,idx], inputs=s_mL,
                                    grad_outputs=torch.ones_like(pRuv_df[:,
                                                                 idx]),
                                    create_graph=True, retain_graph=True)
        dudx_.append(dudx[:,0].reshape(-1,1))

    dudx_cat = torch.cat(dudx_, dim=1)
    nn.train()
    return dudx_cat, pRuv_df, s_mL


def df_R(R: torch.Tensor, s_mL: torch.Tensor = None) -> torch.Tensor:
    """
    Compute neural network derivative with respect to input features using
    PyTorch autograd engine
    """

    dudx_ = []
    for idx, ev in enumerate(torch.eye(9)):
        dudx, = torch.autograd.grad(outputs=R[:,idx], inputs=s_mL,
                                    grad_outputs=torch.ones_like(R[:,idx]),
                                    create_graph=True, retain_graph=True)
        dudx_.append(dudx[:,0].reshape(-1,1))

    dudx_cat = torch.cat(dudx_, dim=1)

    return dudx_cat

