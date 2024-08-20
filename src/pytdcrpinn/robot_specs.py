import torch
import math
import numpy as np
from copy import deepcopy


# Constants
class CCRnp:
    # Independent Parameters
    E = 60e9  # Young's modulus
    G = E /(2 * 1.3)  # Shear modulus
    r = 0.0005  # Cross-sectional radius
    rho = 8000.  # Density
    g = np.array([9.81, 0, 0]).reshape((3,1))  # Gravitational acceleration

    # Dependent Parameters
    A = math.pi * r ** 2  # Cross-sectional area
    I = math.pi * r ** 4 / 4  # Area moment of inertia
    J = 2 * I  # Polar moment of inertia

    Kse = np.diag([G * A, G * A, E * A])  # Stiffness matrices
    Kse_inv = np.linalg.inv(Kse)
    Kbt = np.diag([E * I, E * I, G * J])
    Kbt_inv = np.linalg.inv(Kbt)
    # Boundary Conditions
    p0 = np.array([0, 0, 0]).T
    R0 = np.eye(3)
    L = 0.2
    r1 = 0.008
    r2 = 0.006
    disks_per_segment= np.array([10, 10])
    f_ext = np.array([-0., 0., 0.])


class CCRT(CCRnp):

    Kse = torch.from_numpy(CCRnp.Kse.astype("float32"))
    Kse.requires_grad = False
    Kse_inv = torch.from_numpy(CCRnp.Kse_inv.astype("float32"))
    Kse_inv.requires_grad = False
    Kbt = torch.from_numpy(CCRnp.Kbt.astype("float32"))
    Kbt.requires_grad = False
    Kbt_inv = torch.from_numpy(CCRnp.Kbt_inv.astype("float32"))
    Kbt_inv.requires_grad = False

    p0 = torch.from_numpy(CCRnp.p0.astype("float32"))
    p0.requires_grad = False
    R0 = torch.from_numpy(CCRnp.R0.astype("float32"))
    R0.requires_grad = False

    g = torch.from_numpy(CCRnp.g.astype("float32"))
    L = torch.tensor([CCRnp.L], requires_grad=False, dtype=torch.float32)
    r1 = torch.tensor([CCRnp.r1], requires_grad=False, dtype=torch.float32)
    r2 = torch.tensor([CCRnp.r2], requires_grad=False, dtype=torch.float32)
    disks_per_segment = torch.from_numpy(CCRnp.disks_per_segment.astype(
        "float32"))
    disks_per_segment.requires_grad = False
    f_ext = torch.from_numpy(CCRnp.f_ext.astype("float32")).reshape(1,3,1)
    f_ext.requires_grad = False
    _device = torch.device("cpu")

    def device(self, device):
        self._device = device
        return self

    def _reset_device(self):
        self.Kse = self.Kse.to(self._device)
        self.Kse_inv = self.Kse_inv.to(self._device)
        self.Kbt = self.Kbt.to(self._device)
        self.Kbt_inv = self.Kbt_inv.to(self._device)
        self.p0 = self.p0.to(self._device)
        self.R0 = self.R0.to(self._device)
        self.g = self.g.to(self._device)
        self.L = self.L.to(self._device)


