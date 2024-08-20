import numpy as np
import pandas as pd
import pytest as pt
from pytdcrsv.static_robot_model import StaticRobotModel
from pytdcrsv.trajectory_generation import GenerateTrajectory, \
    InitValuesVCRef
from pytdcrsv.visualize_robot import VisualizeRobot
from copy import deepcopy
import time
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def test_GenerateTrajectory():

    q1 = np.array([0, 0, 0, 0, 0, 1])
    q2 = np.array([0, 0, 0, 0, 1, 0])
    q3 = np.array([2, 0, 3, 0, 0, 0])

    q = [q1, q2, q3]
    steps = 20
    trajectory = GenerateTrajectory(configurations=q, steps=steps)
    q_is = trajectory.trajectory()

    q1_exp = [np.array([0, 0, 0, 0, 0, q_]) for q_ in np.linspace(0, 1, steps)]
    q1_exp = np.concatenate(q1_exp, axis=0)

    q2_exp = [np.array([0, 0, 0, 0, 0, 1]) +
              q_ * np.array([0, 0, 0, 0, 1, -1])
              for q_ in np.linspace(0, 1, steps)]
    q2_exp = np.concatenate(q2_exp, axis=0)

    q3_exp = [np.array([0, 0, 0, 0, 1, 0]) +
              np.array([2 * q_, 0, 3 * q_, 0, -q_, 0])
              for q_ in np.linspace(0, 1,steps)]
    q3_exp = np.concatenate(q3_exp, axis=0)

    q_exp = np.concatenate((q1_exp, q2_exp[6:], q3_exp[6:]), axis=0)

    assert np.allclose(q_exp, q_is)

if __name__ == "__main__":
    main4()
