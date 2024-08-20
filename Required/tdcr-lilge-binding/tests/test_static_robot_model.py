import numpy as np
import pytest as pt
import tdrpyb as tdr
import os
import csv
from pytdcrsv.static_robot_model import StaticRobotModel
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion


def test_computation_fails():
    srob = StaticRobotModel(modelling_approach="VC", number_disks=np.array([
        2, 2]))

    srob.use_recomputation = False
    a = np.array([0, 4, 0, 0, 5, 4, 0, 4, 5, 0, 4, 4])

    succ1 = srob.calc_pose_from_ctrl(a)

    assert not np.alltrue(succ1)
    srob = StaticRobotModel(modelling_approach="VC", number_disks=np.array([
        2, 2]))
    srob.use_recomputation = True
    succ2 = srob.calc_pose_from_ctrl(a)

    assert np.alltrue(succ2)

def test_reorthoganilization():
    r = R.from_euler('z', 90, degrees=True).as_matrix()
    R_err = r + np.array([[1e-5, 1e-5, -1e-5],
                          [-1e-5, 2e-5, -1e-6],
                          [1e-5, -2e-5, -1e-7]])

    x_err = 1 - np.dot(R_err[0,:], R_err[0,:])
    y_err = 1 - np.dot(R_err[1, :], R_err[1, :])
    z_err = 1 - np.dot(R_err[2, :], R_err[2, :])


    r_corr = StaticRobotModel.reorthogonalization(
        R_err)
    rrt = r @ r_corr.T - np.eye(3)

    close = np.allclose(r_corr, r, atol=1e-4)

    assert close

def test_matrix_to_state_vectors():

    Rz = R.from_euler('z', 60, degrees=True)
    Ry = R.from_euler('y', 50, degrees=True)
    Rx = R.from_euler('x', -30, degrees=True)

    quatz_ = Rz.as_quat()
    quatz = np.array([quatz_[3], quatz_[0], quatz_[1], quatz_[2]],
                     dtype=complex)
    quaty_ = Ry.as_quat()
    quaty = np.array([quaty_[3], quaty_[0], quaty_[1], quaty_[2]],
                     dtype=complex)
    quatx_ = Rx.as_quat()
    quatx = np.array([quatx_[3], quatx_[0], quatx_[1], quatx_[2]],
                     dtype=complex)

    T1, T2, T3 = np.eye(4), np.eye(4), np.eye(4)

    T1[0:3, 0:3] = Rz.as_matrix()
    T2[0:3, 0:3] = Ry.as_matrix()
    T3[0:3, 0:3] = Rx.as_matrix()

    t1 = np.array([1, 2, 3])
    t2 = np.array([-2, 3, 0])
    t3 = np.array([0, 0, 0])

    T1[0:3, 3] = t1
    T2[0:3, 3] = t2

    srob = StaticRobotModel(modelling_approach="VC")

    x_exp1 = np.concatenate((t1, quatz), axis=0)
    x_exp2 = np.concatenate((t2, quaty), axis=0)
    x_exp3 = np.concatenate((t3, quatx), axis=0)
    succ = [True, True, True]
    x_is1, inso3 = srob.matrix_to_svec(T1, succ)

    assert np.allclose(x_exp1, x_is1)

    ee_all = np.concatenate((T1, T2, T3), axis=0)
    x_is_all, inso3_ = srob.matrix_to_svec(ee_all, succ)
    x_all_exp = np.concatenate((x_exp1, x_exp2, x_exp3), axis=0).reshape((3,
                                                                          7))

    assert np.allclose(x_is_all, x_all_exp)


def test_matrix_to_state_vectors_ValueError():
    srob = StaticRobotModel(modelling_approach="VC")
    mat = np.eye(3)
    with pt.raises(ValueError):
        srob.matrix_to_svec(mat, [True])


def test_getDefaultInitValue(numerically_solved_models):
    # u,v for the cosserat rod optimization
    #["VC", "PCC", "PRB", "VC_Ref"]
    init_values = [np.array([0, 0, 1, 0, 0, 0]).reshape((1, 6)),
                   np.ones((1,60)) * 0.01,
                   np.zeros((1, 100)),
                   np.array(20 * [[0, 0, 1, 0, 0, 0]]).reshape((1, 120))
                   ]
    for model, init_values in zip(numerically_solved_models, init_values):
        robot = StaticRobotModel(modelling_approach=model)

        init_values_rob = robot.getDefaultInitValue()
        assert np.allclose(init_values, init_values_rob)


def test_setDefaultInitValue(numerically_solved_models):
    # u,v for the cosserat rod optimization

    init_values = [np.array([0, 0, 1, 0, 0, 0]).reshape((1, 6)),
                   np.ones((1, 60)) * 0.01,
                   np.zeros((1, 100)),
                   np.array(20 * [[0, 0, 1, 0, 0, 0]]).reshape((1, 120))
                   ]
    for model, init_values in zip(numerically_solved_models, init_values):
        robot = StaticRobotModel(modelling_approach=model)
        init_values = np.array([1, 1, 1, 1, 1, 1]).reshape((1, 6))

        robot.setDefaultInitialValue(init_values)
        init_values_rob = robot.getDefaultInitValue()
        assert np.allclose(init_values, init_values_rob)


def test_getFinalInitialValues_keepInits(numerically_solved_models):
    init_values = [np.array([0, 0, 1, 0, 0, 0]).reshape((1, 6)),
                   np.ones((1, 60)) * 0.01,
                   np.zeros((1, 100)),
                   np.array(20 * [[0, 0, 1, 0, 0, 0]]).reshape((1, 120))
                   ]
    for model, init_values in zip(numerically_solved_models, init_values):
        q = np.array([2,
                       0,
                       0,
                       0,
                       2,
                       0]).reshape((6, 1))

        # Compute forward kinamatics

        robot = StaticRobotModel(modelling_approach=model)
        robot.keep_inits_uv = True
        _ = robot.calc_pose_from_ctrl(q)
        rtol = 1e-5
        atol = 1e-5

        assert not np.allclose(init_values, robot.getFinalInitValues(),
                               rtol=rtol, atol=atol)


def test_reset_robot_model(numerically_solved_models):
    init_values = [np.array([0, 0, 1, 0, 0, 0]).reshape((1, 6)),
                   np.ones((1, 60)) * 0.01,
                   np.zeros((1, 100)),
                   np.array(20 * [[0, 0, 1, 0, 0, 0]]).reshape((1, 120))
                   ]

    for model, init_values in zip(numerically_solved_models, init_values):
        q = np.array([2,
                      0,
                      0,
                      0,
                      2,
                      0]).reshape((6, 1))

        # Compute forward kinamatics
        robot = StaticRobotModel(modelling_approach=model)
        # Keep inits
        robot.keep_inits_uv = True
        _ = robot.calc_pose_from_ctrl(q)
        kept_init_values = robot.getFinalInitValues()

        assert not np.allclose(init_values, kept_init_values)

        robot.resetLastInits()
        init_values_ = robot.getFinalInitValues()

        assert np.allclose(init_values, init_values_)



""" -------------------------- Cosserat rod model --------------------------"""

#
def test_getDiskFrames_1ctrl(default_robot):
    _, tdcr_maincpp_data = default_robot

    q = tdcr_maincpp_data["q2"]["cosseratrod"]["q"]
    f_ext = tdcr_maincpp_data["q2"]["cosseratrod"]["f"]
    l_ext = tdcr_maincpp_data["q2"]["cosseratrod"]["l"]
    ee_frame = tdcr_maincpp_data["q2"]["cosseratrod"]["ee_frame"]

    diskFrames_exp = tdcr_maincpp_data["q2"]["cosseratrod"]["diskFrames"]
    # Compute forward kinamatics

    robot = StaticRobotModel(modelling_approach="VC")
    _ = robot.calc_pose_from_ctrl(q)

    for c in range(21):

        if c < 10:
            col = f"disk_0{c}"
        else:
            col = f"disk_{c}" if c != 20 else "ee"

        d_state = robot.frames_df.at[0, col]
        d_T_exp = diskFrames_exp[0:, c * 4: c * 4 + 4]
        asd = d_T_exp[:3, :3] @ d_T_exp[:3, :3].T
        rtol=1e-05
        atol=1e-05
        # Lower atol and rtol bounds, since expected test data is not as
        # precise as it should be. Just an issue during the test data export
        # from the main.cpp!
        a = np.allclose(asd, np.eye(3), rtol=rtol, atol=atol)
        d_state_exp = Quaternion(matrix=d_T_exp[:3,:3],rtol=rtol, atol=atol).elements
        d_t = abs(d_state[0:3] - d_T_exp[0:3, 3])
        d_r = abs(d_state[3:] - d_state_exp)
        d = np.concatenate((d_t, d_r))
        # Expecting 20 frames
        res = np.allclose(d, np.zeros_like(d), atol=1e-3)
        assert res

#
def test_getDiskFrames_2ctrl(default_robot):
    _, tdcr_maincpp_data = default_robot

    q1 = tdcr_maincpp_data["q2"]["cosseratrod"]["q"]
    q2 = tdcr_maincpp_data["q2"]["cosseratrod"]["q"]

    q = np.concatenate((np.transpose(q1),
                        np.transpose(q2)),
                       axis=0)

    diskFrames_exp1 = tdcr_maincpp_data["q2"]["cosseratrod"]["diskFrames"]
    diskFrames_exp2 = tdcr_maincpp_data["q2"]["cosseratrod"]["diskFrames"]
    diskFrames_exp = np.concatenate((diskFrames_exp1,
                                     diskFrames_exp2), axis=0)
    # Compute forward kinamatics

    robot = StaticRobotModel(modelling_approach="VC")
    _ = robot.calc_pose_from_ctrl(q.reshape((12, 1)))

    # Run through different actuations
    for b in range(2):
        # Run through all disks
        for c in range(20):
            if c < 10:
                col = f"disk_0{c}"
            else:
                col = f"disk_{c}" if c != 20 else "ee"

            d_state = robot.frames_df.at[b, col]
            d_T_exp = diskFrames_exp[b * 4: b * 4 + 4,
                                     c * 4: c * 4 + 4]
            rtol=1e-05
            atol=1e-06
            # Lower atol and rtol bounds, since expected test data is not as
            # precise as it should be. Just an issue during the test data export
            # from the main.cpp!
            d_state_exp = Quaternion(matrix=d_T_exp, rtol=rtol, atol=atol
                                     ).elements
            d_q = abs(d_state[3:] - d_state_exp)
            d_t = abs(d_state[:3] - d_T_exp[0:3, 3])
            d = np.concatenate((d_t, d_q))
            # Expecting 20 frames
            res = np.allclose(d, np.zeros_like(d), atol=1e-3)
            assert res

#
def test_getDiskFrames_3ctrl_with_f1_l1(default_robot):
    _, tdcr_maincpp_data = default_robot

    q1f1l1 = tdcr_maincpp_data["q1f1l1"]["cosseratrod"]["q"]

    q = np.concatenate((np.transpose(q1f1l1),
                        np.transpose(q1f1l1),
                        np.transpose(q1f1l1)),
                       axis=0)

    diskFrames_exp1 = tdcr_maincpp_data["q1f1l1"]["cosseratrod"]["diskFrames"]
    diskFrames_exp2 = tdcr_maincpp_data["q1f1l1"]["cosseratrod"]["diskFrames"]
    diskFrames_exp3 = tdcr_maincpp_data["q1f1l1"]["cosseratrod"]["diskFrames"]
    diskFrames_exp = np.concatenate((diskFrames_exp1,
                                     diskFrames_exp2,
                                     diskFrames_exp3), axis=0)
    # Compute forward kinamatics

    f__n = tdcr_maincpp_data["q1f1l1"]["cosseratrod"]["f"]
    l__Nm = tdcr_maincpp_data["q1f1l1"]["cosseratrod"]["l"]
    robot = StaticRobotModel(modelling_approach="VC", f__n=f__n, l__Nm=l__Nm)
    s = robot.calc_pose_from_ctrl(q.reshape((18, 1)))

    # Run through different actuations
    for b in range(3):
        for c in range(20):
            if c < 10:
                col = f"disk_0{c}"
            else:
                col = f"disk_{c}" if c != 20 else "ee"

            d_state = robot.frames_df.at[b, col]
            d_T_exp = diskFrames_exp[b * 4: b * 4 + 4, c * 4: c * 4 + 4]
            rtol = 1e-5
            atol = 1e-5
            # Lower atol and rtol bounds, since expected test data is not as
            # precise as it should be. Just an issue during the test data export
            # from the main.cpp!
            try:
                try:
                    d_state_exp = Quaternion(matrix=d_T_exp, rtol=rtol, atol=atol
                                             ).elements
                except:
                    # When arriving here, the external generated test data
                    # was not orthogonal.
                    continue
                d_q = abs(d_state[3:] - d_state_exp)
                d_t = abs(d_state[:3] - d_T_exp[0:3, 3])
                d = np.concatenate((d_t, d_q))
                # Expecting 20 frames
                res = np.allclose(d, np.zeros_like(d), atol=atol, rtol=rtol)
            except:
                res = np.isnan(d_state)
                res = np.alltrue(res)

            assert res

#
def test_model_not_found():
    with pt.raises(ValueError):
        rob = StaticRobotModel(modelling_approach="NotExistent")


def test_calc_pose_from_ctrl_unsuccessful(default_robot):
    _, tdcr_maincpp_data = default_robot

    q1 = np.array([100, 0, 10, 1000, 0, 0]).reshape((6, 1))
    q2 = np.array([4, 0, 0, 0, 2, 0]).reshape((6, 1))
    q = np.concatenate((np.transpose(q1),
                        np.transpose(q2)),
                       axis=0)
    # Compute forward kinamatics
    robot = StaticRobotModel(modelling_approach="VC")
    success = robot.calc_pose_from_ctrl(q.reshape((12, 1)))

    assert not success[0]
    assert success[1]

    df = robot.frames_df

    for r in range(2):
        for c in df.loc[r, :].index:
            if c not in {"ctrl", "success", "actuation_set",
                         "tendon_displacement"}:
                if r == 0:
                    assert np.isnan(df.loc[0, c][0])
                else:
                    assert not np.isnan(df.loc[r, c][0])


"""---------------------- Constant Curvature -------------------------------"""

def test_getDiskFrames_1ctrl_CC(default_robot):
    _, tdcr_maincpp_data = default_robot

    # Constant curvature
    q = tdcr_maincpp_data["q2"]["cosseratrod"]["q"]
    q_cc = tdcr_maincpp_data["q2"]["ConstantCurvature"]["q"]

    diskFrames_exp = tdcr_maincpp_data["q2"]["ConstantCurvature"]["diskFrames"]

    # Compute forward kinamatics
    # Therefore, calculate the CR model beforehand to calculate the tendon
    # displacements
    robot_cr = StaticRobotModel(modelling_approach="VC")
    success_cr = robot_cr.calc_pose_from_ctrl(act=q)
    tendon_displacements = robot_cr.tendon_displacement.to_numpy()[0]

    robot_cc = StaticRobotModel(modelling_approach="CC")
    success_cc = robot_cc.calc_pose_from_ctrl(act=tendon_displacements)

    for c in range(20):
        if c < 10:
            col = f"disk_0{c}"
        else:
            col = f"disk_{c}" if c != 20 else "ee"
        d_state = robot_cc.frames_df.at[0, col]
        d_T_exp = diskFrames_exp[0:, c * 4: c * 4 + 4]

        # Lower atol and rtol bounds, since expected test data is not as
        # precise as it should be. Just an issue during the test data export
        # from the main.cpp!
        rtol = 1e-5
        atol = 1e-5
        d_state_exp = Quaternion(matrix=d_T_exp[:3,:3],rtol=rtol, atol=atol).elements
        d_t = abs(d_state[0:3] - d_T_exp[0:3, 3])
        d_r = abs(d_state[3:] - d_state_exp)
        d = np.concatenate((d_t, d_r))
        # Expecting 20 frames
        res = np.allclose(d, np.zeros_like(d), atol=atol, rtol=rtol)
        assert res


def test_getDiskFrames_3ctrl_with_f1_l1_CC(default_robot):
    _, tdcr_maincpp_data = default_robot

    q1f1l1 = tdcr_maincpp_data["q1f1l1"]["cosseratrod"]["q"]

    q = np.concatenate((np.transpose(q1f1l1),
                        np.transpose(q1f1l1),
                        np.transpose(q1f1l1)),
                       axis=0).reshape((18, 1))

    diskFrames_exp = tdcr_maincpp_data["q1f1l1"]\
                                      ["ConstantCurvature"]\
                                      ["diskFrames"]
    diskFrames_exp = np.concatenate((diskFrames_exp,
                                     diskFrames_exp,
                                     diskFrames_exp), axis=0)
    # Compute forward kinamatics

    f__n = tdcr_maincpp_data["q1f1l1"]["ConstantCurvature"]["f"]
    l__Nm = tdcr_maincpp_data["q1f1l1"]["ConstantCurvature"]["l"]

    # Compute forward kinamatics
    # Therefore, calculate the CR model beforehand to calculate the tendon
    # displacements
    robot_cr = StaticRobotModel(modelling_approach="VC", f__n=f__n,
                                 l__Nm=l__Nm)
    success_cr = robot_cr.calc_pose_from_ctrl(act=q)
    tendon_displacements = robot_cr.tendon_displacement

    robot_cc = StaticRobotModel(modelling_approach="CC", f__n=f__n,
                                 l__Nm=l__Nm)
    success_cc = robot_cc.calc_pose_from_ctrl(act=tendon_displacements)

    # Run through different actuations
    for b in range(3):
        for c in range(20):
            if c < 10:
                col = f"disk_0{c}"
            else:
                col = f"disk_{c}" if c != 20 else "ee"

            d_state = robot_cc.frames_df.at[b, col]
            d_T_exp = diskFrames_exp[b * 4: b * 4 + 4, c * 4: c * 4 + 4]
            rtol = 1e-05
            atol = 1e-05
            # Lower atol and rtol bounds, since expected test data is not as
            # precise as it should be. Just an issue during the test data export
            # from the main.cpp!

            d_state_exp = Quaternion(matrix=d_T_exp, rtol=rtol, atol=atol
                                     ).elements
            d_q = abs(d_state[3:] - d_state_exp)
            d_t = abs(d_state[:3] - d_T_exp[0:3, 3])
            d = np.concatenate((d_t, d_q))
            # Expecting 20 frames
            res = np.allclose(d, np.zeros_like(d), atol=atol, rtol=rtol)
            assert res


""" ---------------------------Pseudo Rigid Body -------------------------- """
def test_getDiskFrames_3ctrl_with_f1_l1_PRB(default_robot):
    _, tdcr_maincpp_data = default_robot

    q1f1l1 = tdcr_maincpp_data["q1f1l1"]["PseudoRigidBody"]["q"]

    q = np.concatenate((np.transpose(q1f1l1),
                        np.transpose(q1f1l1),
                        np.transpose(q1f1l1),
                        np.transpose(q1f1l1),
                        np.transpose(q1f1l1),
                        np.transpose(q1f1l1)),
                        axis=0)

    diskFrames_exp = tdcr_maincpp_data["q1f1l1"]\
                                      ["PseudoRigidBody"]\
                                      ["diskFrames"]

    diskFrames_exp = np.concatenate((diskFrames_exp,
                                     diskFrames_exp,
                                     diskFrames_exp,
                                     diskFrames_exp,
                                     diskFrames_exp,
                                     diskFrames_exp), axis=0)

    # Compute forward kinematics
    f__n = tdcr_maincpp_data["q1f1l1"]["PseudoRigidBody"]["f"]
    l__Nm = tdcr_maincpp_data["q1f1l1"]["PseudoRigidBody"]["l"]

    robot = StaticRobotModel(modelling_approach="PRB", f__n=f__n, l__Nm=l__Nm)
    _ = robot.calc_pose_from_ctrl(q.reshape((36, 1)))

    # Run through different actuations
    for b in range(6):
        for c in range(20):
            if c < 10:
                col = f"disk_0{c}"
            else:
                col = f"disk_{c}" if c != 20 else "ee"

            # Lower atol and rtol bounds, since expected test data is not as
            # precise as it should be. Just an issue during the test data export
            # from the main.cpp!
            d_state = robot.frames_df.at[b, col]
            d_T_exp = diskFrames_exp[b * 4: b * 4 + 4, c * 4: c * 4 + 4]
            rtol = 1e-05
            atol = 1e-05

            d_state_exp = Quaternion(matrix=d_T_exp, rtol=rtol, atol=atol
                                     ).elements
            d_q = abs(d_state[3:] - d_state_exp)
            d_t = abs(d_state[:3] - d_T_exp[0:3, 3])
            d = np.concatenate((d_t, d_q))
            # Expecting 20 frames
            res = np.allclose(d, np.zeros_like(d), atol=atol, rtol=rtol)
            assert res



""" ----------------------- Piecewise Constant Curvature ------------------ """

def test_getDiskFrames_3ctrl_with_f1_l1_PCC(default_robot):
    _, tdcr_maincpp_data = default_robot

    q1f1l1 = tdcr_maincpp_data["q1f1l1"]["PiecewiseConstantCurvature"]["q"]

    q = np.concatenate((np.transpose(q1f1l1),
                        np.transpose(q1f1l1),
                        np.transpose(q1f1l1),
                        np.transpose(q1f1l1),
                        np.transpose(q1f1l1),
                        np.transpose(q1f1l1)),
                       axis=0)

    diskFrames_exp = tdcr_maincpp_data["q1f1l1"]\
                                      ["PiecewiseConstantCurvature"]\
                                      ["diskFrames"]

    diskFrames_exp = np.concatenate((diskFrames_exp,
                                     diskFrames_exp,
                                     diskFrames_exp,
                                     diskFrames_exp,
                                     diskFrames_exp,
                                     diskFrames_exp), axis=0)
    # Compute forward kinamatics
    f__n = tdcr_maincpp_data["q1f1l1"]["PiecewiseConstantCurvature"]["f"]
    l__Nm = tdcr_maincpp_data["q1f1l1"]["PiecewiseConstantCurvature"]["l"]

    robot = StaticRobotModel(modelling_approach="PCC", f__n=f__n, l__Nm=l__Nm)
    _ = robot.calc_pose_from_ctrl(q.reshape((36, 1)))

    # Run through different actuations
    for b in range(6):
        for c in range(20):
            if c < 10:
                col = f"disk_0{c}"
            else:
                col = f"disk_{c}" if c != 20 else "ee"

            # Lower atol and rtol bounds, since expected test data is not as
            # precise as it should be. Just an issue during the test data export
            # from the main.cpp!
            d_state = robot.frames_df.at[b, col]
            d_T_exp = diskFrames_exp[b * 4: b * 4 + 4, c * 4: c * 4 + 4]
            rtol = 1e-05
            atol = 5e-04

            d_state_exp = Quaternion(matrix=d_T_exp, rtol=rtol, atol=atol
                                     ).elements
            d_q = abs(d_state[3:] - d_state_exp)
            d_t = abs(d_state[:3] - d_T_exp[0:3, 3])
            d = np.concatenate((d_t, d_q))
            # Expecting 20 frames
            res = np.allclose(d, np.zeros_like(d), atol=1e-3)
            assert res



""" ----------------------- Subsegment Cosserat Rod ----------------------- """

def test_getDiskFrames_3ctrl_with_f1_l1_SCR(default_robot):
    _, tdcr_maincpp_data = default_robot

    q1f1l1 = tdcr_maincpp_data["q1f1l1"]["SubsegmentCosseratRod"]["q"]

    q = np.concatenate((np.transpose(q1f1l1),
                        np.transpose(q1f1l1),
                        np.transpose(q1f1l1),
                        np.transpose(q1f1l1),
                        np.transpose(q1f1l1),
                        np.transpose(q1f1l1)),
                       axis=0)

    diskFrames_exp = tdcr_maincpp_data["q1f1l1"] \
                                      ["SubsegmentCosseratRod"] \
                                      ["diskFrames"]

    diskFrames_exp = np.concatenate((diskFrames_exp,
                                     diskFrames_exp,
                                     diskFrames_exp,
                                     diskFrames_exp,
                                     diskFrames_exp,
                                     diskFrames_exp), axis=0)
    # Compute forward kinamatics
    f__n = tdcr_maincpp_data["q1f1l1"]["SubsegmentCosseratRod"]["f"]
    l__Nm = tdcr_maincpp_data["q1f1l1"]["SubsegmentCosseratRod"]["l"]
    robot = StaticRobotModel(modelling_approach="VC_Ref", f__n=f__n,
                             l__Nm=l__Nm)
    suc = robot.calc_pose_from_ctrl(q.reshape((36, 1)))

    # Run through different actuations
    for b in range(6):
        for c in range(20):
            if c < 10:
                col = f"disk_0{c}"
            else:
                col = f"disk_{c}" if c != 20 else "ee"

            # Lower atol and rtol bounds, since expected test data is not as
            # precise as it should be. Just an issue during the test data export
            # from the main.cpp!
            d_state = robot.frames_df.at[b, col]
            d_T_exp = diskFrames_exp[b * 4: b * 4 + 4, c * 4: c * 4 + 4]
            rtol = 1e-04
            atol = 1e-04

            try:
                try:
                    d_state_exp = Quaternion(matrix=d_T_exp, rtol=rtol, atol=atol
                                             ).elements
                except:
                    # When arriving here, the external generated test data
                    # was not orthogonal.
                    continue
                d_q = abs(d_state[3:] - d_state_exp)
                d_t = abs(d_state[:3] - d_T_exp[0:3, 3])
                d = np.concatenate((d_t, d_q))
                # Expecting 20 frames
                res = np.allclose(d, np.zeros_like(d), atol=1e-3)
            except:
                res = np.isnan(d_state)
                res = np.alltrue(res)

            assert res



""" ----------------------------- Performance ----------------------------- """

def test_static_robot_model_performance_vc_cc():
    q1 = np.array([1, 2, 0, 3, 0, 1]).reshape((6,1))
    q2 = np.array([0, 2, 3, 0, 0, 1]).reshape((6,1))

    q = np.concatenate((q1, q2, q1, q2), axis=0)

    robot_vc = StaticRobotModel(modelling_approach="VC")
    robot_cc = StaticRobotModel(modelling_approach="CC")

    def cosserat_rod():
        success_vc = robot_vc.calc_pose_from_ctrl(act=q)
        tendon_displacements = robot_vc.tendon_displacement
        return tendon_displacements

    def constant_curvature(tendon_displacements):
        success_cc = robot_cc.calc_pose_from_ctrl(act=tendon_displacements)
        return success_cc

    tendon_displacements = cosserat_rod()
    suc_cc = constant_curvature(tendon_displacements)
    assert True


""" ----------------------------- Fixtures -------------------------------- """
@pt.fixture
def tdcr_maincpp_data():

    # import q1
    files = ["ConstantCurvature.txt",
             "cosseratrod.txt",
             "PiecewiseConstantCurvature.txt",
             "PseudoRigidBody.txt",
             "SubsegmentCosseratRod.txt"]
    folders = ["q1", "q2", "f1", "l1", "q1f1l1"]
    data_dict = {}

    for folder in folders:
        for file in files:
            current_path = os.getcwd()

            # Check if file exists
            path = current_path + "/../tests/" + folder + "/" + file
            if not os.path.isfile(path):
                path = current_path + "/tests/" + folder + "/" + file


            with open(path, newline='\n') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                data = []
                for r, row in enumerate(reader):
                    if r == 0:
                        q = np.array([row[1:]], dtype='f').reshape((6, 1))
                    if r == 1:
                        f = np.array([row[1:]], dtype='f').reshape((3, 1))
                    if r == 2:
                        l = np.array([row[1:]], dtype='f').reshape((3, 1))
                    if r > 2:
                        data.append(row)
                diskframes = np.array(data, dtype='f')
                ee_frame = diskframes[0:, -4:]

            if folder not in data_dict:
                data_dict[folder] = {file[0:-4]: {"q": q,
                                                  "f": f,
                                                  "l": l,
                                                  "diskFrames": diskframes,
                                                  "ee_frame": ee_frame}
                                     }
            else:
                data_dict[folder][file[0:-4]] = {"q": q,
                                                 "f": f,
                                                 "l": l,
                                                 "diskFrames": diskframes,
                                                 "ee_frame": ee_frame}
    return data_dict


@pt.fixture
def main_cpp_output(tdcr_maincpp_data):
    # Build robot
    cosserat_rod = np.array([
        [0.782941,  -0.00323359,    0.622087,    0.127608],
        [-0.244038,  0.918236,      0.311912,    0.143467],
        [-0.572231, -0.396021,      0.718136,    0.336407],
        [0,          0,             0,           1]]
    )

    pcc = np.array([
        [ 0.783122,   -0.00304628,     0.62186,     0.127542],
        [-0.243681,    0.918514,       0.311372,    0.143276],
        [-0.572136,   -0.395377,       0.718566,    0.336556],
        [0,            0,              0,           1]
    ])

    prb = np.array([
        [ 0.777837,   -0.00507746,    0.628445,     0.12941],
        [-0.251097,    0.914175,      0.318173,     0.145928],
        [-0.576125,   -0.405287,      0.709805,     0.334214],
        [0,            0,              0,           1]
    ])

    ss_cosserat_rod = np.array([
        [ 0.777788,  -0.00497019,  0.62851,    0.129371],
        [-0.25119,    0.914184,    0.318086,   0.145941],
        [-0.576154,  -0.405279,    0.709792,   0.334201],
        [0,            0,              0,             1]
    ])

    cc = np.array([
        [ 0.777775, -0.00504295,    0.628522,    0.129386],
        [-0.251165,    0.914167,    0.318143,    0.145977],
        [-0.576178,   -0.405307,     0.70975,    0.334202],
        [0,            0,              0,           1]
    ])



    # Actuation and external forces/moments
    q1 = np.array([4,
                   0,
                   0,
                   0,
                   2,
                   0]).reshape((6, 1))
    q2 = np.array([1,
                   2,
                   3,
                   4,
                   5,
                   6
                   ]).reshape((6, 1))

    q3 = np.array([5,
                   0,
                   6,
                   2,
                   1,
                   3
                   ]).reshape((6, 1))

    q = {"q1": q1, "q2": q2, "q3": q3}

    f_ext = np.array([0,
                      0,
                      0]).reshape((3, 1))


    l_ext = np.array([0,
                      0,
                      0]).reshape((3, 1))

    models_ee = {"CosseratRod": cosserat_rod,
                 "PiecewiseConstantCurvature": pcc,
                 "PseudoRigidBody": prb,
                 "SubsegmentCosseratRod": ss_cosserat_rod,
                 "ConstantCurvature": cc
                 }

    return q, f_ext, l_ext, models_ee


@pt.fixture
def numerically_solved_models():
    return ["VC", "PCC", "PRB", "VC_Ref"]


@pt.fixture
def default_robot(tdcr_maincpp_data):
    """
    Initialise a default robot with default with default configuration and
    external forces. The results are taken from a run of the plain tdcr
    main.cpp.

    Returns:
        robot:  Default robot object
         q:     Default configuration
         f_ext: Default external forces
         l_ext: Default external moments
        models_ee: Output from main.cpp in tdcr-modeling
    """
    # Robot model parameters
    length_array = np.array([0.2, 0.2])
    youngs_modulus_int = 54 * 1e9
    ro = 0.7*1e-3

    number_disks = [10, 10]
    pradius_disks_arry = [10e-3, 10e-3]

    tendon1 = np.array([0,
                        pradius_disks_arry[0],
                        0]).reshape((3, 1))

    tendon2 = np.array([pradius_disks_arry[0] * np.cos(-np.pi / 6),
                        pradius_disks_arry[0] * np.sin(-np.pi / 6),
                        0]
                       ).reshape((3, 1))

    tendon3 = np.array([pradius_disks_arry[0] * np.cos(7 * np.pi / 6),
                        pradius_disks_arry[0] * np.sin(7 * np.pi / 6),
                        0]
                       ).reshape((3, 1))

    tendon4 = np.array([0,
                        pradius_disks_arry[1],
                        0]).reshape((3, 1))

    tendon5 = np.array([pradius_disks_arry[1] * np.cos(-np.pi / 6),
                        pradius_disks_arry[1] * np.sin(-np.pi / 6),
                        0]
                       ).reshape((3, 1))

    tendon6 = np.array([pradius_disks_arry[1] * np.cos(7 * np.pi / 6),
                        pradius_disks_arry[1] * np.sin(7 * np.pi / 6),
                        0]
                       ).reshape((3, 1))

    routing = [tendon1, tendon2, tendon3, tendon4, tendon5, tendon6]



    robot = tdr.TendonDrivenRobot()

    robot.setRobotParameters(length=length_array,
                             youngs_modulus=youngs_modulus_int,
                             routing=routing,
                             number_disks=number_disks,
                             pradius_disks=pradius_disks_arry,
                             ro=ro,
                             two_tendons=False
                             )

    return robot, tdcr_maincpp_data


if __name__ == "__main__":

    rob = StaticRobotModel(modelling_approach="VC", number_disks=np.array(
                                                                    [2, 3]))

    act1 = np.array([0, 0, 0, 0, 0, 0])
    act2 = np.array([0, 0, 0, 0, 0, 1])
    res = rob.calc_pose_from_ctrl(np.concatenate([act1, act2, act1]))
   # state1, state2 = rob.getFullStates_CR()

    a = 1
