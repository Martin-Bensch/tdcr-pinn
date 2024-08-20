import pytest as pt
import os

import tdrpyb as tdr
import numpy as np
import csv

@pt.fixture
def tdcr_maincpp_data():

    # import q1
    files = ["ConstantCurvature.txt",
             "cosseratrod.txt",
             "PiecewiseConstantCurvature.txt",
             "PseudoRigidBody.txt"]
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
                try:
                    diskframes = np.array(data, dtype='f')
                except:
                    a = 1
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



def test_TendonDrivenRobot():
    """ Test object creation"""
    try:
        r = tdr.TendonDrivenRobot()
        assert True
    except:
        assert False


def test_setRobotParameters(default_robot):

    try:
        robot, tdcr_maincpp_data = default_robot
        assert True
    except:
        assert False


def test_forwardKinematics_cosseratrod_1_pose(default_robot):

    robot, tdcr_maincpp_data = default_robot

    q = tdcr_maincpp_data["q1"]["cosseratrod"]["q"]
    f_ext = tdcr_maincpp_data["q1"]["cosseratrod"]["f"]
    l_ext = tdcr_maincpp_data["q1"]["cosseratrod"]["l"]

    ee_frame = tdcr_maincpp_data["q1"]["cosseratrod"]["diskFrames"][0:, -4:]

    # Compute forward kinamatics
    success = robot.forwardKinematics_pyb(1, q, f_ext, l_ext,
                                         tdr.TendonDrivenRobot.CosseratRod)
    ee_pyb = robot.getEEFrames()

    assert np.allclose(np.round(ee_pyb[0:, -4:], 6), ee_frame, atol=1e-3)

def test_forwardKinematics_cosseratrod_2_poses(default_robot):

    robot, tdcr_maincpp_data = default_robot

    q1 = tdcr_maincpp_data["q1"]["cosseratrod"]["q"]
    q2 = tdcr_maincpp_data["q2"]["cosseratrod"]["q"]
    f_ext = tdcr_maincpp_data["q1"]["cosseratrod"]["f"]
    l_ext = tdcr_maincpp_data["q1"]["cosseratrod"]["l"]

    ee_frame1 = tdcr_maincpp_data["q1"]["cosseratrod"]["diskFrames"][0:, -4:]
    ee_frame2 = tdcr_maincpp_data["q2"]["cosseratrod"]["diskFrames"][0:, -4:]

    q = np.concatenate((q1, q2))
    ee_frame = np.round(np.concatenate((ee_frame1, ee_frame2)), 3)

    # Compute forward kinamatics
    print("getEEFrames():")
    success = robot.forwardKinematics_pyb(2, q, f_ext, l_ext,
                                          tdr.TendonDrivenRobot.CosseratRod)
    ee_pyb = np.round(robot.getEEFrames(), 3)
    de = abs(ee_pyb - ee_frame)

    assert np.allclose(de, np.zeros_like(de), atol=1e-5)


def test_cosseratrod_getDiskFrames_storage(default_robot):
    robot, tdcr_maincpp_data = default_robot

    q = tdcr_maincpp_data["q2"]["cosseratrod"]["q"]
    f_ext = tdcr_maincpp_data["q2"]["cosseratrod"]["f"]
    l_ext = tdcr_maincpp_data["q2"]["cosseratrod"]["l"]
    ee_frame = tdcr_maincpp_data["q2"]["cosseratrod"]["ee_frame"]

    diskFrames_exp = tdcr_maincpp_data["q2"]["cosseratrod"]["diskFrames"]
    # Compute forward kinamatics
    ee_pyb = robot.forwardKinematics_pyb(1, q, f_ext, l_ext,
                                         tdr.TendonDrivenRobot.CosseratRod)

    diskFrames = robot.getDiskFrames_storage()

    D = abs(diskFrames_exp - diskFrames)
    # Expecting 20 frames
    assert np.allclose(np.round(diskFrames, 6), diskFrames_exp, atol=1e-3)

def test_cosseratrod_forwardKinematics_error_qsize(default_robot):

    robot, tdcr_maincpp_data = default_robot

    q = tdcr_maincpp_data["q2"]["cosseratrod"]["q"]
    q = np.array([1, 2, 3, 4, 5]).reshape((5, 1))
    f_ext = tdcr_maincpp_data["q2"]["cosseratrod"]["f"]
    l_ext = tdcr_maincpp_data["q2"]["cosseratrod"]["l"]
    ee_frame = tdcr_maincpp_data["q2"]["cosseratrod"]["ee_frame"]
    # Compute forward kinamatics
    with pt.raises(ValueError):
        ee_pyb = robot.forwardKinematics_pyb(1, q, f_ext, l_ext,
                                         tdr.TendonDrivenRobot.CosseratRod)


def test_cosseratrod_forwardKinematics_error_f_ext(default_robot):

    robot, tdcr_maincpp_data = default_robot

    q = tdcr_maincpp_data["q1"]["cosseratrod"]["q"]

    l_ext = tdcr_maincpp_data["q1"]["cosseratrod"]["l"]
    ee_frame = tdcr_maincpp_data["q1"]["cosseratrod"]["ee_frame"]
    f_ext = np.array([1, 2])
    # Compute forward kinamatics
    with pt.raises(ValueError):
        ee_pyb = robot.forwardKinematics_pyb(1, q, f_ext, l_ext,
                                             tdr.TendonDrivenRobot.CosseratRod)


def test_cosseratrod_call_by_const_reference():
    q1 = np.array([1,2,3])

    qplus1 = tdr.call_by_reference_eigenref(q1)

    for n, q in enumerate(q1):
        assert q + 1 == qplus1[n]


def test_m_uv_s0_storage(default_robot):
    robot, tdcr_maincpp_data = default_robot

    q1 = np.array([0, 0, 3, 0, 0, 0])
    q2= np.array([0, 0, 1, 0, 0, 1])
    f_ext = tdcr_maincpp_data["q2"]["cosseratrod"]["f"]
    l_ext = tdcr_maincpp_data["q2"]["cosseratrod"]["l"]
    ee_frame = tdcr_maincpp_data["q2"]["cosseratrod"]["ee_frame"]

    # Compute forward kinamatics
    ee_pyb = robot.forwardKinematics_pyb(1, q1, f_ext, l_ext,
                                         tdr.TendonDrivenRobot.CosseratRod)
    uv01 = robot.getFinalInitValues_uv(tdr.TendonDrivenRobot.CosseratRod)
    uv01_storage = robot.get_uv_s0_storage()

    ee_pyb = robot.forwardKinematics_pyb(1, q2, f_ext, l_ext,
                                         tdr.TendonDrivenRobot.CosseratRod)
    uv02 = robot.getFinalInitValues_uv(tdr.TendonDrivenRobot.CosseratRod)
    uv02_storage = robot.get_uv_s0_storage()

    # COmpute all actuations with one pass
    ee_pyb = robot.forwardKinematics_pyb(2, np.concatenate([q1, q2]), f_ext, l_ext,
                                         tdr.TendonDrivenRobot.CosseratRod)
    uv_12_storage = robot.get_uv_s0_storage()

#    D = abs(diskFrames_exp - diskFrames)
    # Expecting 20 frames
    assert np.allclose(uv_12_storage, np.concatenate([uv01, uv02], axis=0), atol=1e-8)