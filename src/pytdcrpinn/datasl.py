from torch.utils.data import Dataset, DataLoader
import torch
import tdcrpinn.icra2024.robot_specs as rs
from pytdcrsv.static_robot_model import StaticRobotModel
import pytdcrsv.mp_static_robot_model as mpsrm
import numpy as np
import time
import datetime
import tdcrpinn.icra2024.pinn_nn as pnn
from tdcrpinn.icra2024.pinn_nn import DEVICE
import pickle
import json
from torch.distributions import MultivariateNormal as gaussian_dist
from copy import deepcopy
import pandas as pd


class RunReference():
    """
    Run the reference statics model to obtain boundary values

    full_state_res = [p, R, v, u, s]
    """

    def __init__(self, boundary_s_n : int = 100,
                 boundary_tau_max_n : int = 100,
                 tau_max : float = 2.0,
                 phi_range: tuple = (300,360),
                 processes : int =4,
                 ref_name : str = None,
                 verbose=False,
                 ):

        if ref_name is None:
            ref_name = "no_name_provided"
        self.ref_name = ref_name
        self.act_tau_max = None
        self.act_tau_min = None
        self.frames_res_sL = None
        self.act_sL = None
        self.success_res_sL = None
        self.full_state_res_sL = None
        self.actID_res_sL = None

        self.frames_res_tau_max = None
        self.acts_np_tau_max = None
        self.success_res_tau_max = None
        self.full_state_res_tau_max = None
        self.actID_res_tau_max = None

        self._boundary_s = boundary_s_n
        self._boundary_tau_max = boundary_tau_max_n
        self.tau_max = tau_max
        self.tau_max_generated = None

        self.phi_range = phi_range
        self.processes = processes

        self.pRvus_s0_torch = None
        self.pRvus_sL_torch = None
        self.pRvus_sL_data_torch = None
        self.pRvus_tau_max_torch = None
        self.p_test_bench_data = None

        self.test_val_train = None
        self.verbose = verbose

        self.s_p_act_tensor_test_bench = None

    def save_to_disk(self, name):
        path = "./reference/"

        torch.save(self.pRvus_s0_torch, path + name + "_s0.pt")
        torch.save(self.pRvus_sL_torch, path + name + "_sL.pt")
        torch.save(self.pRvus_sL_data_torch, path + name + "_sL_data.pt")
        torch.save(self.pRvus_tau_max_torch, path + name + "_tau_max.pt")
        torch.save(self.pRvus_tau_min_torch, path + name + "_tau_min.pt")

        if self.s_p_act_tensor_test_bench is not None:
            torch.save(self.s_p_act_tensor_test_bench, path + name + "_test_bench.pt")

        np.save(path + name + "_act_sL.npy", self.act_sL)
        np.save(path + name + "_act_tau_max.npy", self.act_tau_max)
        np.save(path + name + "_act_tau_min.npy", self.act_tau_min)


    def load_from_disk(self, folder="./reference/", name=None, verbose=True,
                       name_config="", folder_config=None):

        fname = folder + name
        # load tensors
        self.pRvus_s0_torch = torch.load(fname + "_s0.pt")
        self.pRvus_sL_torch = torch.load(fname + "_sL.pt")
        self.pRvus_sL_data_torch = torch.load(fname + "_sL_data.pt")
        self.pRvus_tau_max_torch = torch.load(fname + "_tau_max.pt")
        self.pRvus_tau_min_torch = torch.load(fname + "_tau_min.pt")

        # load numpy arrays
        self.act_sL = np.load(fname + "_act_sL.npy")
        self.act_tau_max = np.load(fname + "_act_tau_max.npy")
        self.act_tau_min = np.load(fname + "_act_tau_min.npy")

        self.tau_max_generated = np.max(self.act_tau_max)

        # Load indices for training, validation, testing data
        data_type = ["testing", "validation", "training"]
        boundary_type = ["i0", "sL", "dataSL", "tau_max", "tau_min",
                         "test_bench"]
        self.test_val_train = {}
        for dt in data_type:
            dict_tmp = {}
            for bt in boundary_type:
                try:
                    data_name = bt + "_" + dt
                    fn = fname + "_" + data_name +".pt"
                    data = torch.load(fn)
                    dict_tmp[bt] = data
                except Exception as e:
                    print("No test bench data.")

            self.test_val_train[dt] = dict_tmp

        # Load config
        if folder_config is None:
            folder_config = folder[:-10]
            file_name_config = name_config
            path_config = folder_config + "configs/" + file_name_config
        else:
            path_config = folder_config + name_config

        with open(path_config) as user_file:
            parsed_json = json.load(user_file)

            self.phi_range = parsed_json["phi_range"]
            self.collocation_tau_phi = parsed_json["collocation_tau_phi"]
            self.collocation_tau = parsed_json["collocation_tau"]
            self.collocation_s = parsed_json["collocation_s"]
            self.tau_max = parsed_json["tau_max"]
            self.config= parsed_json

        if verbose:
            print("+------------------------------------+")
            print("Loaded data ", fname)
            print("Successful s=L, s=0: ", self.pRvus_s0_torch.shape[0])
            print("Successful s=L remaining data",
                  self.pRvus_sL_data_torch.shape[0])
            print("Successful tau_max data: ", self.pRvus_tau_max_torch.shape[0])

            print("+------------------------------------+")

    def generate_actuations_grid(self, phi_range : tuple, phi_n : int,
                                 Fr : float,
                                 Fr_n : int,
                                 fr_low = -1) -> np.array:
        """
        Generates actuations for location phi and magnitude Fr.

        Args:
            phi_range: tuple specifying lower and upper bound
            phi_n: number of steps
            Fr: maximum force
            Fr_n: number steps

        returns:
            actuation vectors
        """
        if fr_low == -1:
            fr_low = 0.1 * Fr

        acts = []
        if phi_range[0] == phi_range[1] and self.verbose:
            print(f"Generating {phi_n} actuation for the same phi. phi "
                  f"range specifies only a bending plane.")
        for phi_ in np.linspace(phi_range[0], phi_range[1], phi_n):
            for fr in np.linspace(fr_low, Fr, Fr_n):
                if Fr_n == 1:
                    fr = Fr
                act_tmp = self.compute_forces_pairing(fr, phi_)
                acts.append(act_tmp)

        acts_np = np.concatenate(acts)

        # Get maximum tau
        self.tau_max_generated = np.max(acts_np)
        return acts_np

    @staticmethod
    def compute_forces_pairing(Fr: float, phi: float) -> np.array:
        """
        Computes the tendon tension when actuation is given in tension-angle
        representation.

         Counting from tension one counter clock-wise. phi specifies the
         location where the virtual force Fr should be applied on the disks
         radius r. For example, phi=0 would result in Fr = F1, one tendon
         actuation. phi=120° --> Fr=F2 and
         phi = 60° --> F1 = F2, Fr = (F1^2 + F2^2)^(1/2)

         phi is given in degree

        Args:
            Fr: aquivalent single tendon tension
            phi: location on tendon attachment radius
            alpha: angle between attachments

        Returns:
            actuation vector
        """

        assert 0 <= phi <= 360
        if phi <= 120:
            # between tendons 1 and 2
            phi_ = np.deg2rad(phi)
            act1 = np.array([0, 0, 0, 1, 0, 0])
            act2 = np.array([0, 0, 0, 0, 0, 1])
        elif 120 < phi < 240:
            # between tendons 2 and 3
            phi_ = np.deg2rad(phi - 120)
            act1 = np.array([0, 0, 0, 0, 0, 1])
            act2 = np.array([0, 0, 0, 0, 1, 0])
        elif 240 <= phi <= 360:
            # between tendons 3 and 1
            phi_ = np.deg2rad(phi - 240)
            act1 = np.array([0, 0, 0, 0, 1, 0])
            act2 = np.array([0, 0, 0, 1, 0, 0])
        else:
            raise ValueError

        salpha = np.sin(np.deg2rad(120))
        talpha = np.tan(np.deg2rad(120))
        F2 = np.sin(phi_) * Fr / salpha
        F1 = (np.cos(phi_) - np.sin(phi_) / talpha) * Fr

        act = F1 * act1 + F2 * act2

        return act

    def generate_reference(self):

        # Build robot
        L = rs.CCRnp.L
        robot_vc = StaticRobotModel(segment_length__m=np.array([L / 2, L / 2]),
                                    youngs_modulus__n_per_m2=rs.CCRnp.E,
                                    pradius_disks__m=np.array(
                                        [rs.CCRnp.r1, rs.CCRnp.r1]),
                                    ro__m=rs.CCRnp.r,
                                    modelling_approach="VC",
                                    f__n=rs.CCRnp.f_ext)
        robot_vc.use_recomputation = True

        # Compute reference for s=L
        # Therefore, divide actuation space of tau and its magnitude into a
        # grid generate actuations
        sL_Fr_n = int(np.sqrt(self._boundary_s))
        sL_phi_n = int(np.sqrt(self._boundary_s))
        if self.phi_range[0] == self.phi_range[1]:
            sL_Fr_n = self._boundary_s
            sL_phi_n = 1

        self.act_sL = self.generate_actuations_grid(phi_range=self.phi_range,
                                                   phi_n=sL_phi_n,
                                               Fr=self.tau_max, Fr_n=sL_Fr_n)

        print(f"Computing {self.act_sL.shape[0] / 6} actuations for reference "
              f"at s=L")
        start = time.time()
        frames_multi = mpsrm.multi_processing_calc_pose_from_ctrl(robot_vc,
                                                                self.act_sL,
                                                             self.processes)
        result_dict = mpsrm.convert_multiprocessing_result_into_dict(
                                                                frames_multi)
        end = time.time()
        self.frames_res_sL = result_dict["frames"]
        self.success_res_sL = result_dict["success"]
        self.full_state_res_sL = result_dict["full_state"]
        self.actID_res_sL = result_dict["actuations"]

        print(f"Calculation took {datetime.timedelta(seconds=end - start)}")

        # Compute reference for tau_max boundary
        tau_max_Fr_n = 1
        tau_max_phi_n = self._boundary_tau_max

        self.act_tau_max = self.generate_actuations_grid(phi_range=self.phi_range,
                                                     phi_n=tau_max_phi_n,
                                               Fr=self.tau_max, Fr_n=tau_max_Fr_n)

        print(f"Computing {self.act_tau_max.shape[0] / 6} actuations for "
              f"reference tau_max boundary")
        start = time.time()
        frames_multi = mpsrm.multi_processing_calc_pose_from_ctrl(robot_vc,
                                                                  self.act_tau_max,
                                                                  self.processes)
        result_dict = mpsrm.convert_multiprocessing_result_into_dict(
            frames_multi)
        end = time.time()
        self.frames_res_tau_max = result_dict["frames"]
        self.success_res_tau_max = result_dict["success"]
        self.full_state_res_tau_max = result_dict["full_state"]
        self.actID_res_tau_max = result_dict["actuations"]

        print(f"Calculation took {datetime.timedelta(seconds=end - start)}")

        # Compute reference for tau_min boundary
        tau_min_Fr_n = 3
        tau_min_phi_n = 1

        self.act_tau_min = self.generate_actuations_grid(
            phi_range=(0, 0),
            phi_n=tau_min_phi_n,
            Fr=0.0, Fr_n=tau_min_Fr_n)

        print(f"Computing {self.act_tau_min.shape[0] / 6} actuations for "
              f"reference tau_min boundary")
        start = time.time()
        frames_multi = mpsrm.multi_processing_calc_pose_from_ctrl(robot_vc,
                                                                  self.act_tau_min,
                                                                  n_processes=1)

        result_dict = mpsrm.convert_multiprocessing_result_into_dict(
                                                                frames_multi)
        end = time.time()
        self.frames_res_tau_min = result_dict["frames"]
        self.success_res_tau_min = result_dict["success"]
        self.full_state_res_tau_min = result_dict["full_state"]
        self.actID_res_tau_min = result_dict["actuations"]

        print(f"Calculation took {datetime.timedelta(seconds=end - start)}")
        self._to_torch()

    def _to_torch(self):
        """
        Transforms generated reference full_state from numpy to torch
        """

        # s=L reference. Process reference data and extract reference values
        # at s=L
        full_state_sL = []
        full_state_sL_data = []
        full_state_s0 = []
        # Run through results from multiprocessing
        for  success_res, process_res in zip(self.success_res_sL,
                                    self.full_state_res_sL):
            # Run through actuatiions
            for suc, res in  zip(success_res, process_res):
                if not suc:
                    continue
                seg1 = res[0]
                seg2 = res[1]
                full_state_s0.append(seg1[0,:])
                full_state_sL.append(seg2[-1,:])
                seg1seg2 = np.concatenate(res)
                full_state_sL_data.append(seg1seg2[1:-1,:])


        self.pRvus_sL_torch = torch.from_numpy(
                                            np.concatenate(full_state_sL).
                                                            astype("float32").
                                                            reshape(-1, 19, 1))

        self.pRvus_s0_torch = torch.from_numpy(
                                            np.concatenate(full_state_s0).
                                                            astype("float32").
                                                            reshape(-1, 19, 1))

        self.pRvus_sL_data_torch = torch.from_numpy(
                                            np.concatenate(full_state_sL_data)
                                                            .astype("float32").
                                                            reshape(-1, 19, 1))
        # Transform full state for tau_max
        # Run through results from multiprocessing
        full_state_tau_max = []
        for success_ref, process_res in zip(self.success_res_tau_max,
                                            self.full_state_res_tau_max):
            # Run through actuatiions
            for success, res in zip(success_ref, process_res):
                if not success:
                    continue
                seg1seg2 = np.concatenate(res)
                full_state_tau_max.append(seg1seg2)

        self.pRvus_tau_max_torch = torch.from_numpy(
                                            np.concatenate(full_state_tau_max).
                                                    astype(
                                                    "float32").
                                                    reshape(-1, 19, 1)
                                                )
        # Transform full state for tau_min
        # Run through results from multiprocessing
        full_state_tau_min = []
        for success_ref, process_res in zip(self.success_res_tau_min,
                                            self.full_state_res_tau_min):
            # Run through actuatiions
            for success, res in zip(success_ref, process_res):
                if not success:
                    continue
                seg1seg2 = np.concatenate(res)
                full_state_tau_min.append(seg1seg2)

        self.pRvus_tau_min_torch = torch.from_numpy(
            np.concatenate(full_state_tau_min).
                astype(
                "float32").
                reshape(-1, 19, 1)
        )

        print("Computed reference")
        print("Successful s=L, s=0: ", self.pRvus_s0_torch.shape[0])
        print("Successful tau_max data: ",
              self.pRvus_tau_max_torch.shape[0])
        print("Successful tau_min data: ",
              self.pRvus_tau_min_torch.shape[0])

    def store_training_validation_testing_indices(self, ref_name,
                                                  train_ds_i0, val_ds_i0,
                                                  test_ds_i0,
                                                  train_ds_bv_sL, val_ds_bv_sL,
                                                  test_ds_bv_sL,
                                                  train_ds_bv_data,
                                                  val_ds_bv_data,
                                                  test_ds_bv_data,
                                                  train_ds_bv_tau_max,
                                                  val_ds_bv_tau_max,
                                                  test_ds_bv_tau_max,
                                                  train_ds_bv_tau_min,
                                                  val_ds_bv_tau_min,
                                                  test_ds_bv_tau_min,
                                                  train_ds_test_bench=None,
                                                  val_ds_test_bench=None,
                                                  test_ds_test_bench=None
                                                  ):

        i0 = {"training": train_ds_i0,
              "validation": val_ds_i0,
              "testing": test_ds_i0
              }
        sL = {"training": train_ds_bv_sL,
              "validation": val_ds_bv_sL,
              "testing": test_ds_bv_sL
              }
        data = {"training": train_ds_bv_data,
                "validation": val_ds_bv_data,
                "testing": test_ds_bv_data
                }
        tau_max = {"training": train_ds_bv_tau_max,
                   "validation": val_ds_bv_tau_max,
                   "testing": test_ds_bv_tau_max
                   }
        tau_min = {"training": train_ds_bv_tau_min,
                   "validation": val_ds_bv_tau_min,
                   "testing": test_ds_bv_tau_min
                   }
        test_bench = {"training": train_ds_test_bench,
                   "validation": val_ds_test_bench,
                   "testing": test_ds_test_bench
                   }
        data_dict = {"i0": i0,
                     "sL": sL,
                     "dataSL": data,
                     "tau_max": tau_max,
                     "tau_min": tau_min,
                     "test_bench": test_bench
                     }

        for key in data_dict:
            for key2 in data_dict[key]:
                data_name = key + "_" + key2
                fn = "./reference/" + ref_name + "_" + data_name + ".pt"
                torch.save(data_dict[key][key2], fn)

        return data_dict

# Dataset for Initial values s0
class InitialValuesS0(Dataset):
    """
    Dataset holding the initial values, that are computed by the
    RunReference class.
    """
    def __init__(self, rr : RunReference):
        self._s0_values = rr.pRvus_s0_torch.detach().clone()
        self._actuation_np = rr.act_sL
        self.act_torch = None
        self.s_act_torch = None

        self._act_np_to_torch()

    def __len__(self):
        return self._s0_values.shape[0]

    def __getitem__(self, item):
        return self.s_act_torch[item], self._s0_values[item]

    def _act_np_to_torch(self):

        act_np = self._actuation_np.astype("float32").reshape((-1, 6))
        act_np_reduced = act_np[:, 3:]

        self.act_torch = torch.from_numpy(act_np_reduced).reshape(-1, 3)
        self.s_act_torch = torch.cat([self._s0_values[:,18],
                                     self.act_torch], axis=1)


# Dataset for Boundary values sL
class BoundaryValuesSL(Dataset):
    """
    Dataset holding the boundary values for s=L, that are computed by the
    RunReference class.
    """
    def __init__(self, rr : RunReference):
        self._sL_values = rr.pRvus_sL_torch.detach().clone()
        self._actuation_np = rr.act_sL
        self.act_torch = None

        self._act_np_to_torch()

    def __len__(self):
        return self._sL_values.shape[0]

    def __getitem__(self, item):
        return self.s_act_torch[item], self._sL_values[item]

    def _act_np_to_torch(self):

        act_np = self._actuation_np.astype("float32").reshape((-1, 6))
        act_np_reduced = act_np[:, 3:]

        self.act_torch = torch.from_numpy(act_np_reduced).reshape(-1, 3)
        self.s_act_torch = torch.cat([self._sL_values[:, 18],
                                     self.act_torch], axis=1)


class BoundaryValuesDatasl(Dataset):
    """
    Dataset for remaining data points byproduct of Initial/boundary value
    computation s=0, s=L
    """
    def __init__(self, rr : RunReference):
        self._sL_data_values = rr.pRvus_sL_data_torch.detach().clone()
        self._actuation_np = rr.act_sL
        self.act_torch = None

        self._act_np_to_torch()

    def __len__(self):
        return self._sL_data_values.shape[0]

    def __getitem__(self, item):
        # data sl has two entries less than the full data (s0, sL are
        # missing)
        # find the correct actuation index. item access' a disk in the data set.
        # There will be sum(rs.CCRnp.disks_per_segment) disks per actuation.
        # Use floor division // to find the correct index
        # 17 // 19 = 0 for 20 -2 + 1
        # disks_per_segment does not include the basis, but the result does,
        # hence +1
        # s0 and sL are not in the data, therefore subtract 2
        actuation_index = item // (sum(rs.CCRnp.disks_per_segment) - 1)

        # Concat actuation and s
        s = self._sL_data_values[item, -1]
        s_act = torch.cat([s, self.act_torch[actuation_index]])
        return s_act, self._sL_data_values[item]

    def _act_np_to_torch(self):

        act_np = self._actuation_np.astype("float32").reshape((-1, 6))
        act_np_reduced = act_np[:, 3:]

        self.act_torch = torch.from_numpy(act_np_reduced).reshape(-1, 3)


class BoundaryValuesTauMax(Dataset):
    """
    Dataset holding the boundary values for s = L, that are computed by the
    RunReference class.
    """
    def __init__(self, rr : RunReference):
        self._pRvus_tau_max = rr.pRvus_tau_max_torch.detach().clone()
        self._actuation_np = rr.act_tau_max
        self.act_torch = None

        self._act_np_to_torch()

    def __len__(self):
        return self._pRvus_tau_max.shape[0]

    def __getitem__(self, item):
        # data sl has two entries less than the full data (s0, sL are
        # missing)
        # find the correct actuation index. item access a disk in the data set.
        # There will be sum(rs.CCRnp.disks_per_segment) disks per actuation.
        # Use floor division // to find find the correct index
        # 17 // 21 = 0 for 20 + 1
        # disks_per_segment does not include the basis, but the result does,
        # hence +1
        # s0 and sL are in the data, different to the sL data values!
        actuation_index = item // (sum(rs.CCRnp.disks_per_segment) + 1)

        # Concat actuation and s. Since s is associated to the
        # _pRvus_tau_max data tensor, it has to be accessed with the item
        # index. But the act_torch actuationi tensor is much smaller (
        # multiple disks/results for one actuation)
        s = self._pRvus_tau_max[item, -1]
        s_act = torch.cat([s, self.act_torch[actuation_index]])
        return s_act, self._pRvus_tau_max[item]

    def _act_np_to_torch(self):

        act_np = self._actuation_np.astype("float32").reshape((-1, 6))
        act_np_reduced = act_np[:, 3:]

        self.act_torch = torch.from_numpy(act_np_reduced).reshape(-1, 3)


# Dataset for tau_max
class BoundaryValuesTauMin(Dataset):
    """
    Dataset holding the boundary values for s=L, that are computed by the
    RunReference class.
    """
    def __init__(self, rr : RunReference):
        self._pRvus_tau_min = rr.pRvus_tau_min_torch.detach().clone()
        self._actuation_np = rr.act_tau_min
        self.act_torch = None

        self._act_np_to_torch()

    def __len__(self):
        return self._pRvus_tau_min.shape[0]

    def __getitem__(self, item):
        # data sl has two entries less than the full data (s0, sL are
        # missing)
        # find the correct actuation index. item access a disk in the data set.
        # There will be sum(rs.CCRnp.disks_per_segment) disks per actuation.
        # Use floor division // to find find the correct index
        # 17 // 21 = 0 for 20 + 1
        # disks_per_segment does not include the basis, but the result does,
        # hence +1
        # s0 and sL are in the data, different to the sL data values!
        actuation_index = item // (sum(rs.CCRnp.disks_per_segment) + 1)

        # Concat actuation and s. Since s is associated to the
        # _pRvus_tau_max data tensor, it has to be accessed with the item
        # index. But the act_torch actuationi tensor is much smaller (
        # multiple disks/results for one actuation)
        s = self._pRvus_tau_min[item, -1]
        s_act = torch.cat([s, self.act_torch[actuation_index]])
        return s_act, self._pRvus_tau_min[item]

    def _act_np_to_torch(self):

        act_np = self._actuation_np.astype("float32").reshape((-1, 6))
        act_np_reduced = act_np[:, 3:]

        self.act_torch = torch.from_numpy(act_np_reduced).reshape(-1, 3)


# Dataset for collocation points
class CollocationPoints(Dataset):
    def __init__(self, rr : RunReference, collocation_tau_phi=72,
                                            collocation_tau=30,
                                            collocation_s=100,
                 verbose=True):
        self.verbose = verbose
        self.collocation_tau_phi = collocation_tau_phi
        self.collocation_tau = collocation_tau
        self.collocation_s = collocation_s
        self._rr = deepcopy(rr)
        self.L_max = rs.CCRnp.L
        self.col_s = None
        self.act = None
        self.generate_collocation_points_grid()
        if self.verbose:
            print(f"Generated {len(self)} collocation points")

    def __getitem__(self, item):
        return self.s_act_torch[item]

    def __len__(self):
        return self.s_act_torch.shape[0]

    def generate_collocation_points_grid(self):

        if self._rr.phi_range[0] == self._rr.phi_range[1] and self.verbose:
            print("phi_range specifies one bending plane.")
        acts_np = self._rr.generate_actuations_grid(
                                                phi_range=self._rr.phi_range,
                                               phi_n=self.collocation_tau_phi,
                                               Fr=self._rr.tau_max,
                                               Fr_n=self.collocation_tau)
        act_np = acts_np.astype("float32").reshape((-1, 6))
        act_np_reduced = act_np[:, 3:]

        self.col_s = np.linspace(0, self.L_max, self.collocation_s).astype("float32")

        self.act_torch = torch.from_numpy(act_np_reduced).reshape(-1, 3)
        self.col_s_torch = torch.from_numpy(self.col_s).reshape(-1, 1)
        s_act_lst = []
        for s_ in self.col_s:
            s__ = s_ * torch.ones((self.act_torch.shape[0], 1))
            s_act_tmp = torch.cat([s__, self.act_torch], axis=1)
            s_act_lst.append(s_act_tmp)

        self.s_act_torch = torch.cat(s_act_lst)

    def generate_collocation_points_importance_sampling(self, res_df,
                                                        tau_centers_n=10,
                                                        s_centers_n=10,
                                                        s_scale=0.1 *
                                                                rs.CCRnp.L,
                                                        tau_scale =None):
        if tau_scale is None:
            tau_scale = self._rr.tau_max
        # Get centers from res_df (which holds the evaluation result,
        # by first selecting tau_center_n tendon actuations
        n_largest_values = res_df.groupby("max", sort=True).indices

        # Sample gaussian for all pairs
        # Sample for tau
        # Build mean and covariance matrices from worst_tau_s_pairs
        cov_mat = torch.zeros((4,4))
        cov_mat[0][0] = s_scale
        cov_mat[1][1] = tau_scale
        cov_mat[2][2] = tau_scale
        cov_mat[3][3] = tau_scale

        # Get mean
        collocation_lst = []
        max_values = [v for v in n_largest_values]

        for idx, i_row in enumerate(max_values[:-(tau_centers_n+1):-1]):

            df_tau = res_df[res_df["max"] == i_row]
            df_tau_n_largest = df_tau.nlargest(s_centers_n, ["euclid"], "all")

            if len(df_tau_n_largest) < s_centers_n:
                s_sampled = (torch.rand(s_centers_n - len(df_tau_n_largest))
                             * self.L_max)
                s_dict = {"s": s_sampled,
                          "t1": [df_tau["t1"].iloc[0]] * len(s_sampled),
                          "t2": [df_tau["t2"].iloc[0]] * len(s_sampled),
                          "t3": [df_tau["t3"].iloc[0]] * len(s_sampled)}
                s_df = pd.DataFrame(s_dict)
                df_tau_n_largest = pd.concat([df_tau_n_largest, s_df], axis=0)


            for j_row in range(len(df_tau_n_largest)):
                s = df_tau_n_largest.iloc[j_row]["s"]
                t1 = df_tau_n_largest.iloc[j_row]["t1"]
                t2 = df_tau_n_largest.iloc[j_row]["t2"]
                t3 = df_tau_n_largest.iloc[j_row]["t3"]
                s_act = torch.from_numpy(np.array([s, t1, t2, t3]).astype(
                                                                    "float32"))
                data_points = int((self.collocation_tau_phi *
                                   self.collocation_tau *
                                   self.collocation_s) /
                                   (tau_centers_n * s_centers_n))
                m = gaussian_dist(loc=s_act, covariance_matrix=cov_mat)
                s_act_sampled = m.sample((data_points,))

                # Set zero tendon to zero again
                failure = True
                for n in [1, 2, 3]:
                    if res_df[f"t{n}"].max() == res_df[f"t{n}"].min() == 0.0:
                        # n equals zero tendon
                        s_act_sampled[:, n] = 0.0
                        failure = False

                if failure:
                    print("Something went wrong.")
                    print(res_df)
                # If sampled tendon tension exceeds tau_max or is negative,
                # Map it to the boundary value
                s1_mask = s_act_sampled[:,0] > rs.CCRnp.L
                t1_mask = s_act_sampled[:, 1] > self._rr.tau_max_generated
                t2_mask = s_act_sampled[:, 2] > self._rr.tau_max_generated
                t3_mask = s_act_sampled[:, 3] > self._rr.tau_max_generated

                s_act_sampled[s1_mask, 0] = rs.CCRnp.L
                s_act_sampled[t1_mask, 1] = self._rr.tau_max_generated
                s_act_sampled[t2_mask, 2] = self._rr.tau_max_generated
                s_act_sampled[t3_mask, 3] = self._rr.tau_max_generated

                s1_mask = s_act_sampled[:, 0] < 0.
                t1_mask = s_act_sampled[:, 1] < 0
                t2_mask = s_act_sampled[:, 2] < 0
                t3_mask = s_act_sampled[:, 3] < 0

                s_act_sampled[s1_mask, 0] = 0.0
                s_act_sampled[t1_mask, 1] = 0.0
                s_act_sampled[t2_mask, 2] = 0.0
                s_act_sampled[t3_mask, 3] = 0.0

                collocation_lst.append(s_act_sampled)

        self.s_act_torch = torch.cat(collocation_lst)


    def save_to_disk(self, ref_name):
        data_dict = {
                     "collocation_tau_phi": self.collocation_tau_phi,
                     "collocation_tau": self.collocation_tau,
                     "collocation_s": self.collocation_s
                     }
        with open("./reference/" + ref_name + "_collocation_config.json",
                  "w") as file:
            json.dump(data_dict, file)


