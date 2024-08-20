import numpy as np
import pytest as pt
import tdrpyb as tdr
import os
import csv
from pytdcrsv.static_robot_model import StaticRobotModel
from pytdcrsv.visualize_robot import VisualizeRobot
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion


def test_svec_to_matrix():
    Rz = Rotation.from_euler('z', 60, degrees=True).as_matrix()
    Ry = Rotation.from_euler('y', 50, degrees=True).as_matrix()
    Rx = Rotation.from_euler('x', -30, degrees=True).as_matrix()
    Rot = Rz @ Ry @ Rx

    t = np.array([1, 2, 3]).reshape((3, 1))

    T = np.eye(4)
    T[0:3, 0:3] = Rot
    T[0:3, 3:] = t

    svec, succ = StaticRobotModel.matrix_to_svec(T, [True])
    matrix = VisualizeRobot.svec_to_matrix(svec)

    assert np.allclose(T, matrix)


def test__transform_disk_3dof():
    # Construct a circle of 4 points in the xz plane
    p1 = np.array([.003, 0, 0, 1]).reshape((4, 1))
    p2 = np.array([0, 0, .003, 1]).reshape((4, 1))
    p3 = np.array([-.003, 0, 0, 1]).reshape((4, 1))
    p4 = np.array([0, 0, -.003, 1]).reshape((4, 1))

    points = (p1, p2, p3, p4)

    # Construct test rotation
    Rx = Rotation.from_euler("x", 90, degrees=True)
    rot_mat = Rx.as_matrix()
    t = np.zeros((3, 1))
    T = np.eye(4)
    T[0:3, 0:3] = rot_mat
    T[0:3, 3:4] = t

    svec, succ = StaticRobotModel.matrix_to_svec(T, [True])
    rob = StaticRobotModel()
    rviz = VisualizeRobot(robot=rob)
    transformed_points = rviz._disk_in_base_frame(svec=svec,
                                                  n_points=4,
                                                  disk_radius__m=.003)

    for n, p in enumerate(points):
        tp = transformed_points[:, n].reshape((3, 1))
        assert np.allclose(tp, p[0:3])


def test__transform_disk_6dof():
    # Construct a circle of 4 points in the xz plane
    p1 = np.array([.003, 0, 0, 1]).reshape((4, 1))
    p2 = np.array([0, 0, .003, 1]).reshape((4, 1))
    p3 = np.array([-.003, 0, 0, 1]).reshape((4, 1))
    p4 = np.array([0, 0, -.003, 1]).reshape((4, 1))


    points = (p1, p2, p3, p4)

    # Construct test rotation
    Rx = Rotation.from_euler("x", 90, degrees=True)
    rot_mat = Rx.as_matrix()
    t = np.zeros((4, 1))
    t[0] = 0.1
    t[1] = 0.2
    t[2] = -0.3

    T = np.eye(4)
    T[0:3, 0:3] = rot_mat
    T[:, 3:4] = t

    svec, succ = StaticRobotModel.matrix_to_svec(T, [True])
    rob = StaticRobotModel()
    rviz = VisualizeRobot(robot=rob)
    transformed_points = rviz._disk_in_base_frame(svec=svec,
                                                  n_points=4,
                                                  disk_radius__m=.003)

    for n, p in enumerate(points):
        tp = transformed_points[:, n].reshape((3, 1))
        p = p + t
        assert np.allclose(tp, p[0:3])


def test_draw_robot_ValueError():
    # Create robot
    segment_length__m = 0.2
    f__n = np.array([0, 0, 0])
    l__Nm = np.array([0, 0, 0])
    modelling_approach_vc = "VC"
    # modelling_approach_cc = "CC"

    robot_vc = StaticRobotModel(segment_length__m=segment_length__m,
                                f__n=f__n,
                                l__Nm=l__Nm,
                                modelling_approach=modelling_approach_vc)

    # Actuate robot
    q1 = np.array([1, 2, 0, 3, 0, 1]).reshape((6, 1))
    success_vc = robot_vc.calc_pose_from_ctrl(act=q1)
    print(f"Calulation suceesfull: {success_vc}")

    # Visualilze robot
    vrob = VisualizeRobot(robot=robot_vc)
    with pt.raises(ValueError):
        vrob.draw_robot(indices=2.1)
