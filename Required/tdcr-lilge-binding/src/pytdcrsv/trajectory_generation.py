import pytdcrsv.config as config

import pandas as pd
from scipy.optimize import least_squares
import numpy as np
from pytdcrsv.static_robot_model import StaticRobotModel
from dataclasses import dataclass
import typing as tp
from copy import deepcopy

from multiprocessing import Pool
import datetime
import os
import json
import matplotlib.pyplot as plt
from matplotlib import ticker


# Create a custom logger
logger = config.logging.getLogger(__name__)
# Add handlers to the logger
logger.addHandler(config.C_HANDLER)
logger.addHandler(config.F_HANDLER)

@dataclass
class InitValuesVCRef:
    """
    Dataclass for the initialization of the reference model

    Values from the Lilge paper
    """
    segment_length__m: np.ndarray = np.array([0.15, 0.15])
    f__n: np.ndarray = np.array([0, 0, 0])
    l__Nm: np.ndarray = np.array([0, 0, 0])
    youngs_modulus__n_per_m2: float = 54e9
    pradius_disks__m: np.ndarray = np.array([0.008, 0.006])
    ro__m: float = .7 * 1e-3
    # Initial values
    values = (segment_length__m[0], segment_length__m[1], pradius_disks__m[
        0], pradius_disks__m[1])

    def __repr__(self):
        str1 = f"segment length: {self.segment_length__m}\n"
        str2 = f"youngs modulus: {self.youngs_modulus__n_per_m2}\n"
        str3 = f"pradius_disks:  {self.pradius_disks__m}\n"
        str4 = f"ro:             {self.ro__m}\n"

        return str1 + str2 +str3 + str4

    def __str__(self):
        str0 = "InitialValuesVCRef Class \n"
        str1 = f"segment length: {self.segment_length__m}\n"
        str2 = f"youngs modulus: {self.youngs_modulus__n_per_m2}\n"
        str3 = f"pradius_disks:  {self.pradius_disks__m}\n"
        str4 = f"ro:             {self.ro__m}\n"

        return str0 + str1 + str2 + str3 + str4

    def write_dictionary(self):
        d = {
                "segment_length__m": self.segment_length__m.tolist(),
                "f__n": self.f__n.tolist(),
                "l__Nm": self.l__Nm.tolist(),
                "youngs_modulus__n_per_m2": self.youngs_modulus__n_per_m2,
                "pradius_disks__m": self.pradius_disks__m.tolist(),
                "ro__m": self.ro__m
        }
        return d


class GenerateTrajectory:
    """
    Class for calculating simple trajectories made up of fix points.

    Args:
        configurations: List of numpy arrays representing actuation vectors
        steps: Amount of steps for interpolation between start and end
    """
    def __init__(self, configurations: tp.List[np.ndarray] = None, steps=10):

        if type(configurations) is not list:
            raise TypeError("Expected a list of np.ndarray")

        self._start_conf = np.array([0, 0, 0, 0, 0, 0])
        self.configurations = configurations
        self.steps = steps
        self._calibration_res = None


    def _lin_trajectory(self, start_conf: np.ndarray = None, end_conf: \
        np.ndarray = None) -> \
            np.ndarray:
        """ Calculates one linear trajectory through interpolation"""

        if start_conf is None:
            start_conf = np.zeros((6, 1))

        q1 = np.concatenate([start_conf + s * (end_conf - start_conf) for
                            s in np.linspace(1/(self.steps - 1), 1,
                                             self.steps - 1)]
                            )
        return q1

    def trajectory(self, configurations: list = None) -> np.ndarray:
        """ Concatenates trajectories for multiple target poses"""
        if type(configurations) is not list:
            if configurations is None:
                configurations = self.configurations
            else:
                raise ValueError

        trajectory = np.array([0, 0, 0, 0, 0, 0])

        for c in configurations:
            traj_temp = self._lin_trajectory(trajectory[-6:], c)
            trajectory = np.concatenate((trajectory, traj_temp))

        return trajectory


def main():
    model = "VC"

    p0 = [InitValuesVCRef.segment_length__m * 1.1,
          InitValuesVCRef.pradius_disks__m[0] * 0.9,
          InitValuesVCRef.pradius_disks__m[1] * 0.9,
          InitValuesVCRef.ro__m * 1.05
          ]
    q1 = np.array([2, 0, 0, 0, 0, 0])
    q2 = np.array([0, 0, 0, 0, 6, 6])
    #q3 = np.array([2, 0, 3, 0, 0, 0])
    q = [q1, q2]#, q3]
    steps = 10
    trajectory = GenerateTrajectory(configurations=q, steps=steps)
    q_is = trajectory.trajectory()


if __name__ == "__main__":
    main()