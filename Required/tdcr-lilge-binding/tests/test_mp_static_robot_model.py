import numpy as np
import pandas as pd
import pytest as pt
from pytdcrsv.static_robot_model import StaticRobotModel
from pytdcrsv.mp_static_robot_model import multi_processing_calc_pose_from_ctrl, \
    convert_multiprocessing_result_into_dict


def test_multi_processing_calc_pose_from_ctrl():
    rob_model = StaticRobotModel(segment_length__m=np.array([0.15, 0.2]),
                                 modelling_approach="VC")

    act1 = np.array([0, 0, 1, 0, 0, 2])
    act2 = np.array([0, 0, 1, 0, 0, 3])
    act3 = np.array([0, 0, 1, 0, 0, 1])
    act4 = np.array([0, 0, 2, 0, 0, 1])
    act5 = np.array([0, 0, 3, 0, 0, 1])
    act6 = np.array([0, 2, 0, 0, 0, 1])

    frames1_suc = rob_model.calc_pose_from_ctrl(act=act1)
    frames1 = rob_model._tdr_robot.getDiskFrames_storage()
    frames2_suc = rob_model.calc_pose_from_ctrl(act=act2)
    frames2 = rob_model._tdr_robot.getDiskFrames_storage()
    frames3_suc = rob_model.calc_pose_from_ctrl(act=act3)
    frames3 = rob_model._tdr_robot.getDiskFrames_storage()
    frames4_suc = rob_model.calc_pose_from_ctrl(act=act4)
    frames4 = rob_model._tdr_robot.getDiskFrames_storage()
    frames5_suc = rob_model.calc_pose_from_ctrl(act=act5)
    frames5 = rob_model._tdr_robot.getDiskFrames_storage()
    frames6_suc = rob_model.calc_pose_from_ctrl(act=act6)
    frames6 = rob_model._tdr_robot.getDiskFrames_storage()

    frames = multi_processing_calc_pose_from_ctrl(rob_model,
                                                  np.concatenate([act1, act2, act3, act4,
                                                   act5, act6]))

    frames456 = np.concatenate([frames4, frames5, frames6])
    assert np.allclose(frames1, frames[0][0])
    assert np.allclose(frames2, frames[1][0])
    assert np.allclose(frames3, frames[2][0])
    assert np.allclose(frames456, frames[3][0])


def test_multi_processing_calc_pose_from_ctrl_more_processes_than_actuations():
    rob_model = StaticRobotModel(segment_length__m=np.array([0.15, 0.2]),
                                 modelling_approach="VC")

    act1 = np.array([0, 0, 1, 0, 0, 2])
    act2 = np.array([0, 0, 1, 0, 0, 3])
    act3 = np.array([0, 0, 1, 0, 0, 1])
    act4 = np.array([0, 0, 2, 0, 0, 1])
    act5 = np.array([0, 0, 3, 0, 0, 1])
    act6 = np.array([0, 2, 0, 0, 0, 1])

    frames1_suc = rob_model.calc_pose_from_ctrl(act=act1)
    frames1 = rob_model._tdr_robot.getDiskFrames_storage()
    frames2_suc = rob_model.calc_pose_from_ctrl(act=act2)
    frames2 = rob_model._tdr_robot.getDiskFrames_storage()
    frames3_suc = rob_model.calc_pose_from_ctrl(act=act3)
    frames3 = rob_model._tdr_robot.getDiskFrames_storage()
    frames4_suc = rob_model.calc_pose_from_ctrl(act=act4)
    frames4 = rob_model._tdr_robot.getDiskFrames_storage()
    frames5_suc = rob_model.calc_pose_from_ctrl(act=act5)
    frames5 = rob_model._tdr_robot.getDiskFrames_storage()
    frames6_suc = rob_model.calc_pose_from_ctrl(act=act6)
    frames6 = rob_model._tdr_robot.getDiskFrames_storage()

    frames = multi_processing_calc_pose_from_ctrl(rob_model,
                                                  np.concatenate([act1, act2, act3, act4,
                                                   act5, act6]),
                                                  7)

    assert np.allclose(frames1, frames[0][0])
    assert np.allclose(frames2, frames[1][0])
    assert np.allclose(frames3, frames[2][0])
    assert np.allclose(frames4, frames[3][0])
    assert np.allclose(frames5, frames[4][0])
    assert np.allclose(frames6, frames[5][0])

def test_convert_multiprocessing_result_into_dict():
    act1 = np.array([0, 0, 1, 0, 0, 2])
    act2 = np.array([0, 0, 1, 0, 0, 3])
    act3 = np.array([0, 0, 1, 0, 0, 1])
    act4 = np.array([0, 0, 2, 0, 0, 1])
    act5 = np.array([0, 0, 3, 0, 0, 1])
    act6 = np.array([0, 2, 0, 0, 0, 1])

    acts = [act1, act2, act3, act4, act5, act6,
            act1, act2, act3, act4, act5, act6,
            act1, act2, act3, act4, act5, act6,
            act1, act2, act3, act4, act5, act6]

    rob_model = StaticRobotModel(segment_length__m=np.array([0.15, 0.2]),
                                 modelling_approach="VC")

    # Actuations will split into 4 equally sized actuation vectors
    frames_multi = multi_processing_calc_pose_from_ctrl(rob_model, np.concatenate(acts),
                                                        4)

    result_dict = convert_multiprocessing_result_into_dict(frames_multi)

    frames_res = result_dict["frames"]
    success_res = result_dict["success"]
    full_state_res = result_dict["full_state"]
    actID_res = result_dict["actuations"]

    # Compare dict representation with raw result
    assert np.allclose(frames_multi[0][0], result_dict["frames"][0])
    assert np.allclose(full_state_res[0][0][1], result_dict["full_state"][0][0][1])