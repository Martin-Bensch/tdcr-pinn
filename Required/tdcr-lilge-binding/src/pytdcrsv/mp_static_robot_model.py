import pytdcrsv.config as config

from pytdcrsv.static_robot_model import StaticRobotModel
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
from copy import deepcopy
import pandas as pd

def multi_processing_calc_pose_from_ctrl(robot: StaticRobotModel,
                                         act: np.ndarray,
                                         n_processes: int = 4):
    """
    Computes poses for actuation vector act by using multiprocessing.

    full_state = [p, R, v, u, s]
    Args:
        robot: StaticRobotModel
        act: actuation vector
        n_processes: number of processes

    Returns:
        list of results per process, each entry consists of frames,
        success, full_state, actuation id
    """
    if type(act) is list:
        act = np.concatenate(act)
        raise Warning("Changed input list to numpy array, using np.concatenate()")
    act = act.reshape((-1, 6, 1))

    # Devide act into chunks for parallellisation
    actuations_n = act.shape[0]
    rest = actuations_n % n_processes
    chunk_size = (actuations_n - rest) / n_processes
    if chunk_size > 0:
        actuations = []
        ids_chunks = []
        for n in range(n_processes):
            lidx = int(n * chunk_size)
            uidx = int((n + 1) * chunk_size)
            if n < n_processes - 1:
                a = act[lidx:uidx,:,:].reshape(-1,1)
            else:
                a = act[lidx:, :, :].reshape(-1,1)
            actuations.append(a)
    else:
        actuations = []
        ids_chunks = []
        chunk_size = 1
        for n in range(actuations_n):
            lidx = int(n * chunk_size)
            uidx = int((n + 1) * chunk_size)
            if n < actuations_n - 1:
                a = act[lidx:uidx, :, :].reshape(-1, 1)
            else:
                a = act[lidx:, :, :].reshape(-1, 1)
            actuations.append(a)

    modelling_approach = robot.modeling_approach
    model_properties = robot.model_properties
    processes = []
    with Pool(processes=n_processes) as pool:
        # Add processes to the pool
        # For each direction add a new process
        for idx, a in enumerate(actuations):
            # access actuations
            if modelling_approach == "CC":
               raise NotImplementedError

            processes.append(pool.apply_async(
                _compute_pose,
                args=(model_properties, a
                      )
            )
            )
        # Collect all results
        df_succ_fullstate_idsc_results = [p.get() for p in processes]

    return df_succ_fullstate_idsc_results


def convert_multiprocessing_result_into_dict(
        df_succ_fullstate_idsc_results):
    """
    Converts the results from multiprocessing into a readable dictionary
    """
    frames_lst = []
    success_lst = []
    fullstate_lst = []
    actuations_lst = []
    for entry in df_succ_fullstate_idsc_results:
        frames_lst.append(entry[0])
        success_lst.append(entry[1])
        fullstate_lst.append(entry[2])
        actuations_lst.append(entry[3])

    result_dict = {
                    "frames": frames_lst,
                    "success": success_lst,
                    "full_state": fullstate_lst,
                    "actuations": actuations_lst
                   }
    #result_frame = pd.DataFrame(result_dict)

    return result_dict

def _compute_pose(model_properties, act):

    robot = StaticRobotModel(**model_properties)
    success = robot.calc_pose_from_ctrl(act)

    frames = robot.disk_frames_array
    full_state = robot.full_state_lst

    return [frames, success, full_state, act]