# Bachelorarbeit von Jakob Wenner, Betreut von Martin Bensch am imes (LUH 2021)
import pytdcrsv.config as config

import numpy as np
import pandas as pd
from pyquaternion import Quaternion
import tdrpyb as tdr
from copy import deepcopy
from pytdcrsv.visualize_robot import animate_pose
import pytdcrsv.visualize_robot as vr
import time
import matplotlib.pyplot as plt

# Create a custom logger
logger = config.logging.getLogger(__name__)
# Add handlers to the logger
logger.addHandler(config.C_HANDLER)
logger.addHandler(config.F_HANDLER)


class StaticRobotModel:
    """
    This class provides different implementations of modeling approaches for
    tendon driven continuum robots.

    Results are written to the object's data frame

    For usage see the /Code/jupyter_lab/minimal_example.ipynb jupyter lab.

    For more information see https://github.com/SvenLilge/tdcr-modeling

    Args:
        segment_length__m:
                        Segment lengths for each segment in meters
                        np.array([l1, l2])
        f__n:
            External force vector acting on the tip
        l__Nm:
            External moment vector acting on the tip (w.r.t base frame)
        youngs_modulus__n_per_m2:
                                Youngs module for the backbone in N per m^2
        pradius_disks__m:
                        Radii for the tendon routing holes in m, for each
                        segment as a np.ndarray([r1, r2])
        ro__m:
            Radius of the backbone in m
        modelling_approach:
                        Choose modelling approach

            .. table:: Available models and their abbreviations
                :widths: auto

                ======== =================== ==============
                Argument  Model              Tendon Routing
                ======== =================== ==============
                "VC"      Cosserat Rod        fc
                "VC_Ref"  Cosserat Rod        pc
                "CC"      Constant Curvature  fc
                "PCC"     Piecewise CC        pc
                "PRB"     Pseudo rigid body   pc
                ======== =================== ==============

    Attributes:
        g__m_per_s2: Gravitational acceleration
        frames_df: pandas data frame holing all poses (state vector) per
                   control
    """

    def __init__(self,
                 segment_length__m: np.ndarray = None,
                 f__n: np.ndarray = np.array([0, 0, 0]),
                 l__Nm: np.ndarray = np.array([0, 0, 0]),
                 youngs_modulus__n_per_m2: float = 54e9,
                 pradius_disks__m: np.ndarray = np.array([0.01, 0.01]),
                 ro__m: float = .7 * 1e-3,
                 modelling_approach: str ="VC",
                 number_disks = np.array([10, 10]),
                 gxf_tol=1e-11,
                 step_size=1e-4):

        if segment_length__m is None:
            segment_length__m = np.array([0.1, 0.1])
        elif type(segment_length__m) is not np.ndarray:
            if type(segment_length__m) is float or type(segment_length__m) is float:
                segment_length__m = np.array([segment_length__m, segment_length__m])
            elif type(segment_length__m) is list:
                if len(segment_length__m) == 2:
                    segment_length__m = np.array(segment_length__m)
            else:
                raise TypeError("segment_length__m should be an numpy ndarray")

        if len(segment_length__m) < 2:
            segment_length__m = [segment_length__m, segment_length__m]


        # Only used for the _td_job model
        self._computation_n = 0
        self.time_dict = {}
        self.segment_length__m = segment_length__m
        self.f__n = f__n
        self.l__Nm = l__Nm
        self.youngs_modulus__n_per_m2 = youngs_modulus__n_per_m2
        self.pradius_disks__m = pradius_disks__m
        self.ro__m = ro__m
        # For models from the Sven Lilge repo
        self._number_disks = number_disks
        self._modelling_approach = modelling_approach
        self.gxf_tol = gxf_tol
        self.step_size = step_size

        self._model_properties = {"segment_length__m": segment_length__m,
                                  "f__n": self.f__n,
                                  "l__Nm": self.l__Nm,
                                  "youngs_modulus__n_per_m2":
                                      self.youngs_modulus__n_per_m2,
                                  "pradius_disks__m": self.pradius_disks__m,
                                  "ro__m": self.ro__m,
                                  "modelling_approach": modelling_approach,
                                  "number_disks": self._number_disks,
                                  "gxf_tol": self.gxf_tol,
                                  "step_size": self.step_size
                                  }

        # For models from the Sven Lilge repo
       # self._number_disks = [10, 10]

        self._modelling_approach = modelling_approach
        self._model_enum = None

        # Tendon driven robot instance
        self._tdr_robot = None

        # Init values u and v for numerically solved models
        self._keep_inits_uv = None

        # Modeling approaches
        # Not an instance of a specific model class!
        # All model class instances are access through the TendonDrivenRobot
        # class.
        self._model = self._choose_model()
        self._poses_count = 0
        self.frames_df = None
        self.tendon_displacement = None

        # state = [p, R_vec, v, u, s]
        # full_state1_lst is a list, where one entry
        self.full_state1_lst_tmp = None # only for the Cosserat rod model
        self.full_state2_lst_tmp = None
        self.full_state_lst_tmp = None

        self.full_state1_lst = None  # only for the Cosserat rod model
        self.full_state2_lst = None
        self.full_state_lst = None
        self.disk_frames_array = None
        self.use_recomputation = True
        self.time_lst = []

    @property
    def number_disks(self):
        return self._number_disks
    @property
    def model_properties(self):
        return self._model_properties
    @property
    def modeling_approach(self):
        return self._modelling_approach

    @property
    def model_enum(self):
        model_enum_dict = {
                        "VC": tdr.TendonDrivenRobot.CosseratRod,
                        "CC": tdr.TendonDrivenRobot.ConstantCurvature,
                        "PCC": tdr.TendonDrivenRobot.PiecewiseConstantCurvature,
                        "PRB": tdr.TendonDrivenRobot.PseudoRigidBody,
                        "VC_Ref": tdr.TendonDrivenRobot.SubsegmentCosseratRod
                        }

        return model_enum_dict[self._modelling_approach]

    @property
    def keep_inits_uv(self):
        return self._keep_inits_uv

    @keep_inits_uv.setter
    def keep_inits_uv(self, keep: bool):
        """
            Use last computed init values uv for the next iteration
        """
        if self._modelling_approach in {"VC", "VC_Ref", "PCC", "PRB"}:
            if isinstance(keep, bool):
                self._tdr_robot.keepInits(keep)
                self._keep_inits_uv = keep
            else:
                raise ValueError

    def resetLastInits(self):
        """
        Reset the last initial values of after a model was run.
        """
        if self._modelling_approach in {"VC", "VC_Ref", "PCC", "PRB"}:
            #init_values = np.concatenate((u, v))
            self._tdr_robot.reset_last_inits()

    def _getFullStates_CR(self):

        if self._modelling_approach == "VC":
            full_state1 = self._tdr_robot.getStateSegmentN_CR(1)
            full_state2 = self._tdr_robot.getStateSegmentN_CR(2)

            full_state1_lst = []
            for pose in range(full_state1.shape[0]):
                pose_states = full_state1[pose,:].reshape(-1, 19)
                full_state1_lst.append(pose_states)

            full_state2_lst = []
            for pose in range(full_state2.shape[0]):
                pose_states = full_state2[pose, :].reshape(-1, 19)
                full_state2_lst.append(pose_states)

            self.full_state1_lst_tmp = full_state1_lst
            self.full_state2_lst_tmp = full_state2_lst
            self.full_state_lst_tmp = [[s1, s2] for s1, s2 in zip(full_state1_lst, full_state2_lst)]

        else:
            raise NotImplementedError("Only implemented for VC model")

    def getDefaultInitValue(self) -> [float]:
        """
        Get Default init values

        Returns:
            init values
        """
        if self._modelling_approach in {"VC", "VC_Ref", "PCC", "PRB"}:
            init_values = self._tdr_robot.\
                                getDefaultInitValues_uv(self.model_enum)
            return init_values

    def setDefaultInitialValue(self, init_values_uv: np.ndarray=None):
        """
        Set the default initial values

        Args:
            init_values_uv: initial values. uv as np.array, first 3 entries
            are v, remaining are u

        Returns:
            None
        """
        if self._modelling_approach in {"VC", "VC_Ref", "PCC", "PRB"}:
            #init_values = np.concatenate((u, v))
            self._tdr_robot.setDefaultInitValues_uv(init_values_uv)

    def getFinalInitValues(self) -> np.ndarray:
        """
        Receive final initial Values.

        Returns:
            Final initial values
        """
        if self._modelling_approach in {"VC", "VC_Ref", "PCC", "PRB"}:
            init_values = self._tdr_robot. \
                getFinalInitValues_uv(self.model_enum)
            return init_values

    def setRobotParameters(self, pradius_disks__m: [float, float],
                           youngs_modulus__n_per_m2: float,
                           ro__m: float):
        """
        Set the robot parameters

        Args:
            pradius_disks__m: Disks radius
            youngs_modulus__n_per_m2: youngs modulus
            ro__m: rod radius

        Returns:
            None
        """

        self.pradius_disks__m = pradius_disks__m
        self.youngs_modulus__n_per_m2 = youngs_modulus__n_per_m2
        self.ro__m = ro__m

        length_array = self.segment_length__m

        tendon1 = np.array([0,
                            self.pradius_disks__m[0],
                            0]).reshape((3, 1))

        tendon2 = np.array([self.pradius_disks__m[0] * np.cos(-np.pi / 6),
                            self.pradius_disks__m[0] * np.sin(-np.pi / 6),
                            0]
                           ).reshape((3, 1))

        tendon3 = np.array([self.pradius_disks__m[0] * np.cos(7 * np.pi / 6),
                            self.pradius_disks__m[0] * np.sin(7 * np.pi / 6),
                            0]
                           ).reshape((3, 1))

        tendon4 = np.array([0,
                            self.pradius_disks__m[1],
                            0]).reshape((3, 1))

        tendon5 = np.array([self.pradius_disks__m[1] * np.cos(-np.pi / 6),
                            self.pradius_disks__m[1] * np.sin(-np.pi / 6),
                            0]
                           ).reshape((3, 1))

        tendon6 = np.array([self.pradius_disks__m[1] * np.cos(7 * np.pi / 6),
                            self.pradius_disks__m[1] * np.sin(7 * np.pi / 6),
                            0]
                           ).reshape((3, 1))

        routing = [tendon1, tendon2, tendon3, tendon4, tendon5, tendon6]

        print("\n Set parameters: ", [length_array,
                                          self.youngs_modulus__n_per_m2,
                                            self.pradius_disks__m[0],
                                            self.pradius_disks__m[1],
                                          self.ro__m])
        self._tdr_robot.setRobotParameters(length=length_array,
                                           youngs_modulus=
                                           self.youngs_modulus__n_per_m2,
                                           routing=routing,
                                           number_disks=self._number_disks,
                                           pradius_disks=self.pradius_disks__m,
                                           ro=self.ro__m,
                                           two_tendons=False
                                           )

    def _choose_model(self):#
        """
        Helper function to select the modeling approach.
        """
        self._init_robot()
        if self._modelling_approach == "VC":
            return self._cosserat_rod
        elif self._modelling_approach == "CC":
            return self._constant_curvature
        elif self._modelling_approach == "PCC":
            return self._piecewise_constant_curvature
        elif self._modelling_approach == "PRB":
            return self._pseudo_rigid_body
        elif self._modelling_approach == "VC_Ref":
            return self._subsegment_cosserat_rod
        elif self._modelling_approach == "test":
            return self._calibration_test
        else:
            raise ValueError(f"Model not found: "
                             f"{self._modelling_approach}.")

    def calc_pose_from_ctrl(self, act: np.ndarray = None,
                            actuation_set: int = 0) -> [bool]:
        """
        Calculates the end effector pose from a given actuation vector.

        Args:
            act:
                np.array(shape=(X, 1)) actuation vector. Units depend on the
                chosen model (forces in N or tendon length differences in
                meters)
            actuation_set:
                Bending direction for the given actuation
        Returns:
                a list of boolean values indicating a successful computation
                for a given actuation.
        """
        if self.frames_df is not None:
            self.frames_df = None

        if self.tendon_displacement is not None:
            #self.logger.info("Resetting tendon displacement data frame.")
            self.tendon_displacement = None

        if act is None:
            logger.error("Argument Error")
            raise ValueError

        if isinstance(act, pd.Series):
            logger.info("Convert Pandas Series to numpy array")
            act_lst = []
            for a in act:
                act_lst.append(a)
            act = np.array(act_lst).reshape((-1, 1))

        # Success per pose
        act = np.round(act, 8) # 100 -> error
        #a = self.getFinalInitValue()
        success_vec = self._model(act)

        # Store disk frames
        disk_frames_array = self._tdr_robot.getDiskFrames_storage()

        if self._modelling_approach == "VC":
            self.full_state1_lst = self.full_state1_lst_tmp
            self.full_state2_lst = self.full_state2_lst_tmp
            self.full_state_lst = self.full_state_lst_tmp

        if self.use_recomputation and not np.alltrue(success_vec) and \
                self._modelling_approach == "VC":
            second_try = []
            poses_count = self._poses_count
            for idxs, s in enumerate(success_vec):
                if not s:
                    a = 1
                    # Generate trajectory
                    a = []
                    act_tmp = act[6 * idxs: 6 * (idxs + 1)]
                    a_half = 0.5 * act_tmp
                    act_traj_tmp = np.concatenate([a_half, act_tmp])

                    self.resetLastInits()
                    self.keep_inits_uv = True
                    success_vec_tmp = self._model(act_traj_tmp)
                    self.keep_inits_uv = False
                    self.resetLastInits()
                    if np.alltrue(success_vec_tmp):
                        # Get new calculated pose
                        disk_frames_array_tmp = \
                                        self._tdr_robot.getDiskFrames_storage()
                        df_tmp = disk_frames_array_tmp[4: 4 * (2), :]
                        # Change entry in df
                        disk_frames_array[4 * idxs: 4 * (idxs + 1), :] = df_tmp[:,:]
                        success_vec[idxs] = True
                        self.full_state1_lst[idxs] = self.full_state1_lst_tmp[-1]
                        self.full_state2_lst[idxs] = self.full_state2_lst_tmp[-1]
                        self.full_state_lst[idxs] = self.full_state_lst_tmp[-1]
            self._poses_count = poses_count
        frames_array = {}
        frames_array["actuation_set"] = [actuation_set for _ in
                                         range(len(success_vec))]

        act_reshape = act.reshape((self._poses_count, 6))
        frames_array["ctrl"] = [q_ for q_ in act_reshape]

        # Run through all disks
        for m in range(self._number_disks[0] + self._number_disks[1] + 1):
            frames_disk_n = []
            # Convert matrices to state vectors per disk
            sv_disk_m, inso3 = self.matrix_to_svec(
                                        disk_frames_array[:, 4 * m: 4 * m + 4],
                                        success_vec
                                                  )
            for n in range(self._poses_count):
                # Insert disk frames into pandas data frame
                sv = sv_disk_m[n, :]
                frames_disk_n.append(sv)

            if m == self._number_disks[0] + self._number_disks[1]:
                str_ = "ee"
            else:
                str_ = f"disk_{m}" if m > 9 else f"disk_0{m}"

            frames_array[str_] = frames_disk_n

        frames_array["success"] = inso3
        self.frames_df = pd.DataFrame(frames_array)

        self.frames_df = pd.concat([self.frames_df, self.tendon_displacement],
                                   axis=1)
        self.disk_frames_array = disk_frames_array
        if not np.alltrue(success_vec):
            print("-----Error-------")
            print(success_vec)
            for n in range(len(success_vec)):
                 print(act[n * 6:6 * (n + 1)])
            print("-----------")
        #else:
         #   print("-----Success-------")
           # print(success_vec)
           # for n in range(len(success_vec)):
           #     print(act[n * 6:6 * (n + 1)])
           # print("-----------")

        return success_vec

    def _cosserat_rod(self, act):
        """
        Uses the cosserat rod model from Sven Lilges Repository (see README.md)
        through binding code.

        Args:
            act:
                np.array(shape=(X, 1)) actuation vector. Units depend on the
                chosen model (forces in N or tendon length differences in
                meters)

        Returns:
                np.array(shape=(7, 1)) mit der Pose als Informationsgehalt in
                kartesischen Koordinaten und Quaternionen

        """
        # Count poses
        if len(act) % 6 != 0:
            raise ValueError("Size of actuation vector is not a multiple of "
                             "6.")
        self._poses_count = int(len(act)/6)

        # Calculate tip poses for all actuation vectors
        time_s = time.time()
        success = self._tdr_robot.forwardKinematics_pyb(int(self._poses_count),
                                              act,
                                              self.f__n,
                                              self.l__Nm,
                                              tdr.TendonDrivenRobot.CosseratRod)
        time_e = time.time()

        self.time_dict[self._computation_n] ={"poses_n": int(self._poses_count),
                                              "seconds":time_e - time_s }
        self._computation_n += 1
        # Store the full state
        self._getFullStates_CR()
        # In order to calculate the CC model later on, one has to receive
        # the current tendon length from the VC model and store them.
        self._insert_tendon_displacements_into_df(success)

        return success

    def _pseudo_rigid_body(self, act):
        """
        Uses the pseudo rigid body model from Sven Lilges Repository (see
        README.md) through binding code.

        Args:
            act:
                np.array(shape=(X, 1)) actuation vector. Units depend on the
                chosen model (forces in N or tendon length differences in
                meters)

        Returns:
                np.array(shape=(7, 1)) mit der Pose als Informationsgehalt in
                kartesischen Koordinaten und Quaternionen

        """
        # Count poses
        if len(act) % 6 != 0:
            raise ValueError("Size of actuation vector is not a multiple of "
                             "6.")
        self._poses_count = int(len(act)/6)

        # Calculate tip poses for all actuation vectors
        success = self._tdr_robot.forwardKinematics_pyb(int(self._poses_count),
                                                        act,
                                                        self.f__n,
                                                        self.l__Nm,
                                                        tdr.TendonDrivenRobot.
                                                        PseudoRigidBody)

        # Check if all poses where calculated successfully

        # In order to calculate the CC model later on, one has to receive
        # the current tendon length from the VC model and store them.
        #self._insert_tendon_displacements_into_df(success)

        return success

    def _subsegment_cosserat_rod(self, act):
        """
        Makes use of the subsegment cosserat rod model from Sven Lilges
        repository (see README.md) using binding code. This model will be
        the reference model within our corresponding paper, just like it was
        in the paper associated with the aforementioned repository.

        Args:
            act:
                np.array(shape=(X, 1)) actuation vector. Units depend on the
                chosen model (forces in N or tendon length differences in
                meters)

        Returns:
                np.array(shape=(7, 1)) mit der Pose als Informationsgehalt in
                kartesischen Koordinaten und Quaternionen

        """
        # Count poses
        if len(act) % 6 != 0:
            raise ValueError("Size of actuation vector is not a multiple of "
                             "6.")
        self._poses_count = int(len(act)/6)

        # Calculate tip poses for all actuation vectors
        success = self._tdr_robot.forwardKinematics_pyb(int(self._poses_count),
                                                        act,
                                                        self.f__n,
                                                        self.l__Nm,
                                                        tdr.TendonDrivenRobot.
                                                        SubsegmentCosseratRod)

        # In order to calculate the CC model later on, one has to receive
        # the current tendon length from the VC model and store them.
        self._insert_tendon_displacements_into_df(success)

        return success

    def _constant_curvature(self, act=None):
        """
        Uses the constant curvature model from Sven Lilges Repository (see
        README.md) through binding code.

        In contrast to the different models, this approach depends on the
        cosserat rod approach, since it is actuated through tendon
        displacements while the remaining models use tendon forces.

        Args:
            act:
                np.array(shape=(X, 1)) actuation vector. Units depend on the
                chosen model (forces in N or tendon length differences in
                meters)

        Returns:
                np.array(shape=(N, 1)) ruterns an array consisting of
                boolean values, which denote successful or un-successful
                calculations, in the same ordering as the actuation vectors
                in act.

        """
        # Count poses
        if len(act) % 6 != 0:
            raise ValueError("Size of actuation vector is not a multiple of "
                             "6.")

        self._poses_count = int(len(act)/6)

        # Calculate tip poses for all actuation vectors
        success = self._tdr_robot.forwardKinematics_pyb(self._poses_count,
                                                        act,
                                                        self.f__n,
                                                        self.l__Nm,
                                                        tdr.TendonDrivenRobot.
                                                        ConstantCurvature)

        # Check if all poses where calculated successfully

        # In order to calculate the CC model later on, one has to receive
        # the current tendon lengths from the VC model and store them.
        #self._insert_tendon_displacements_into_df(success)

        return success

    def _piecewise_constant_curvature(self, act):
        """
        Makes use of the piecewise constant curvature model from Sven Lilges
        repository (see README.md) using binding code.

        Args:
            act:
                np.array(shape=(X, 1)) actuation vector. Units depend on the
                chosen model (forces in N or tendon length differences in
                meters)

        Returns:
                np.array(shape=(7, 1)) mit der Pose als Informationsgehalt in
                kartesischen Koordinaten und Quaternionen

        """
        # Count poses
        if len(act) % 6 != 0:
            raise ValueError("Size of actuation vector is not a multiple of "
                             "6.")
        self._poses_count = int(len(act)/6)

        # Calculate tip poses for all actuation vectors
        success = self._tdr_robot.forwardKinematics_pyb(int(self._poses_count),
                                                        act,
                                                        self.f__n,
                                                        self.l__Nm,
                                                    tdr.TendonDrivenRobot.
                                                    PiecewiseConstantCurvature)

        # Check if all poses where calculated successfully

        # In order to calculate the CC model later on, one has to receive
        # the current tendon length from the VC model and store them.
        #self._insert_tendon_displacements_into_df(success)

        return success

    def _insert_tendon_displacements_into_df(self, success):
        """
        Inserts all tendon displacements into the data frame
        """
        dl = self._tdr_robot.getTendonDisplacements_storage()
        dl_dict = {}
        for n, b in enumerate(success):
            if b:
                dl_dict[n] = deepcopy(dl[:, n])
            else:
                logger.info("Unsuccessful calculation. Writing np.nan to "
                            "data frame")
                dl_dict[n] = [np.nan] * 6

        self.tendon_displacement = pd.Series(dl_dict, name=
                                                         "tendon_displacement")

    def _init_robot(self):
        """
        Initializes robot with some hard coded values.
        """
        length_array = np.array(self.segment_length__m)

        tendon1 = np.array([0,
                            self.pradius_disks__m[0],
                            0]).reshape((3, 1))

        tendon2 = np.array([self.pradius_disks__m[0] * np.cos(-np.pi / 6),
                            self.pradius_disks__m[0] * np.sin(-np.pi / 6),
                            0]
                           ).reshape((3, 1))

        tendon3 = np.array([self.pradius_disks__m[0] * np.cos(7 * np.pi / 6),
                            self.pradius_disks__m[0] * np.sin(7 * np.pi / 6),
                            0]
                           ).reshape((3, 1))

        tendon4 = np.array([0,
                            self.pradius_disks__m[1],
                            0]).reshape((3, 1))

        tendon5 = np.array([self.pradius_disks__m[1] * np.cos(-np.pi / 6),
                            self.pradius_disks__m[1] * np.sin(-np.pi / 6),
                            0]
                           ).reshape((3, 1))

        tendon6 = np.array([self.pradius_disks__m[1] * np.cos(7 * np.pi / 6),
                            self.pradius_disks__m[1] * np.sin(7 * np.pi / 6),
                            0]
                           ).reshape((3, 1))



        routing = [tendon1, tendon2, tendon3, tendon4, tendon5, tendon6]

        self._tdr_robot = tdr.TendonDrivenRobot()
        self._tdr_robot.setRobotParameters(length=length_array,
                                           youngs_modulus=
                                           self.youngs_modulus__n_per_m2,
                                           routing=routing,
                                           number_disks=self._number_disks,
                                           pradius_disks=self.pradius_disks__m,
                                           ro=self.ro__m,
                                           two_tendons=False
                                           )

        self._tdr_robot.set_gxf_tol(self.gxf_tol, self.step_size)

    @staticmethod
    def matrix_to_svec(t_end, success_vec):
        """
        Calculates state vectors from given homogeneous sub-matrix in t_end

        Args:
            t_end:
                np.array(shape=(4 * N, 4)) matrix, sub-matrices describe
                the end effector in the robots base frame
            success_vec:
                Indicates the successful computed poses

        Returns:
                np.array((shape=(N, 7)) transposed state vector consisting of
                cartesian position and orientation as quaternion.
                The quaternion is in scalar first order (w, a, b, b)!
                x = [pos, quat] = [x,y,z, w, a,b,c]
        """

        if t_end.shape[0] % 4 != 0:
            raise ValueError("Wrong amount of end effector transformations.")

        count_t_end = t_end.shape[0] / 4

        state_vectors = np.zeros((int(count_t_end), 7))

        for n in range(int(count_t_end)):
            # Access t_end for n-th pose
            t_ = t_end[n * 4: (n + 1) * 4, :]
            if not success_vec[n]:
                state_vectors[n, :] = np.nan
                success_vec[n] = False
                continue

            try:
                quaternion = Quaternion(matrix=t_, atol=1e-8, rtol=1e-8)
            except:
                try:
                    r = t_[0:3, 0:3]
                    r_corr = StaticRobotModel.reorthogonalization(r)
                    t_[0:3, 0:3] = r_corr
                    quaternion = Quaternion(matrix=t_, atol=1e-8, rtol=1e-8)
                except:
                    try:
                        #r = t_[0:3, 0:3]
                        r_corr_2 = StaticRobotModel.reorthogonalization(r_corr)
                        t_[0:3, 0:3] = r_corr_2
                        quaternion = Quaternion(matrix=t_, atol=1e-8, rtol=1e-8)
                    except:
                        logger.warning(f"Rotation matrix for pose {n} is not in SO("
                                       "3).")
                        state_vectors[n, :] = np.nan
                        success_vec[n] = False
                        continue
            x = t_[0, 3]
            y = t_[1, 3]
            z = t_[2, 3]
            w = quaternion.scalar
            a = quaternion.vector[0]
            b = quaternion.vector[1]
            c = quaternion.vector[2]
            # Store into numpy array
            state_vectors[n, :] = [x, y, z, w, a, b, c]
            success_vec[n] = True

        return state_vectors, success_vec

    @staticmethod
    def reorthogonalization(r):
        """
        Reorthogonalizes a given matrix R

        [See here](https://stackoverflow.com/questions/23080791/eigen-re
        -orthogonalization-of-rotation-matrix)

        Args:
            r:
              Rotation matrix with R R^T != I

        Returns:
            R: Rotation matrix that fullfills R R^T = I
            success: Indicator for R being orthogonal
        """

        x = r[0, :]
        y = r[1, :]

        error = np.dot(x, y)
        x_ort = x - (error / 2) * y
        y_ort = y - (error / 2) * x
        z_ort = np.cross(x_ort, y_ort)

        #x_new = 0.5 * (3 - np.dot(x_ort, x_ort)) * x_ort
        #y_new = 0.5 * (3 - np.dot(y_ort, y_ort)) * y_ort
        #z_new = 0.5 * (3 - np.dot(z_ort, z_ort)) * z_ort

        x_new_ = x_ort / np.linalg.norm(x_ort)
        y_new_ = y_ort / np.linalg.norm(y_ort)
        z_new_ = z_ort / np.linalg.norm(z_ort)
        #R = np.eye(3)
        #R[0, :] = x_new[:]
        #R[1, :] = y_new[:]
        #R[2, :] = z_new[:]

        R_ = np.eye(3)
        R_[0, :] = x_new_[:]
        R_[1, :] = y_new_[:]
        R_[2, :] = z_new_[:]

        # Test property
        #rrt = R @ R.T
        rrt_ = R_ @ R_.T
        #success = np.allclose(R @ R.T, np.eye(3))
        #success_ = np.allclose(R_ @ R_.T, np.eye(3), atol=1e-7)

        return R_#, success_


if __name__ == "__main__": #pragma : no cover

    rd = 0.0008
    E= 2.11 * 1e11
    #I =

    #T_M = E * I * np.pi * 0.5 / (2 * r_d * L)
    robot_vc = StaticRobotModel(
        segment_length__m=np.array(
            [float(.120), float(.120)]),
        youngs_modulus__n_per_m2=64 * 1e9,
        pradius_disks__m=np.array([0.016, 0.016]),
        ro__m=0.001,
        modelling_approach="VC",
        f__n=np.array([0., 0., 0.]),
        gxf_tol=1e-4,
        step_size=0.001
    )

    robot_vc.use_recomputation = True
    # print(f"Started computation for {acts.shape[0] / 6}")
    a = np.concatenate([np.array([0, 0, 10, 0, 0, 0])])
    success = robot_vc.calc_pose_from_ctrl(act=a)

    print(robot_vc.time_dict)
    vrob = vr.VisualizeRobot(robot_vc)
    vrob.draw_robot(animate=False, scale_disk=1)
    vrob.show()
    plt.show()

"""
    f_ext = np.array([-0.1, 0., 0.])
    srob = StaticRobotModel(modelling_approach="VC", number_disks=np.array([
                            10, 10]), gxf_tol=1e-8, step_size= 1e-2,
                            segment_length__m=np.array([0.1, 0.1]),
                            f__n=f_ext)

    a1 = np.array([0, 0, 0, 2, 0, 0])
    a = np.concatenate([a1])
    succ = srob.calc_pose_from_ctrl(a)
    print(srob.frames_df)
    print(succ)

    E = srob.youngs_modulus__n_per_m2  # Young's modulus
    G = E / (2 * 1.3)  # Shear modulus
    r = srob.ro__m  # Cross-sectional radius
    rho = 8000.  # Density
    g = np.array([9.81, 0, 0]).reshape((3, 1))  # Gravitational acceleration
    # L = 0.5  # Length(before strain)

    # Dependent Parameters
    A = np.pi * r ** 2  # Cross-sectional area
    I = np.pi * r ** 4 / 4  # Area moment of inertia
    J = 2 * I  # Polar moment of inertia

    Kse = np.diag([G * A, G * A, E * A])  # Stiffness matrices
    Kse_inv = np.linalg.inv(Kse)
    Kbt = np.diag([E * I, E * I, G * J])
    Kbt_inv = np.linalg.inv(Kbt)

    Tee = srob.disk_frames_array
    R = Tee[0:3, 80:83].reshape(3,3)
    p = Tee[0:3, 83].reshape(3,1)
    L0 = np.array([-2 * 0.01, 0, 0]).reshape(3,1) + np.cross(p.reshape(3),
                                                                 f_ext).reshape(3,1)
    u0 = Kbt_inv @ L0
    u_rob = srob.full_state1_lst[0][0][15:18]
    a = 1

"""