from train_nn import train_model
import numpy as np
import robot_specs as rs
from torch.utils.tensorboard import SummaryWriter
import pinn_nn as pnn
import os
import torch
import datetime
from multiprocessing import Pool, RLock
from copy import deepcopy
import json
import datasl as ds
import traceback


def dict_unpack_mp(config_dict):
    ## save config
    name = f"{os.path.basename(__file__)[:-3]}_" + config_dict["name_nn"]

    with open("./configs/" + name + "_config.json", "w") as file:
        json.dump(config_dict, file)

    config_dict["name_nn"] = name
    try:
        # Run training
        run_training_config(**config_dict)
    except Exception as e:
        print(e)
        print("---")
        print(traceback.format_exc())
        return None


def run_training_config(
                        hidden_layer_n=3,
                        layer_width=250,
                        boundary_s_n=100,
                        boundary_tau_max_n=100,
                        collocation_tau_phi=72,
                        collocation_tau=30,
                        collocation_s=100,
                        tau_max=0.7,
                        processes=4,
                        percentage_training_data=70,
                        batch_size_collocation=512,
                        batch_size_iv_s0=512,
                        batch_size_bv_sL=512,
                        batch_size_bv_sL_data=512,
                        batch_size_bv_tau_max=512,
                        learning_rate: float = 0.01,
                        max_epochs: int = 1_000,
                        save_reference=True,
                        load_reference=True,
                        ref_name=None,
                        name_nn="",
                        phi_range=(0, 360),
                        importance_sampling=True,
                        tau_centers_n=10,
                        s_centers_n=10,
                        s_scale=0.1 * rs.CCRnp.L,
                        tau_scale =None,
                        epoch_resume_training=-1,
                        nn_opt_name=None
                        ):
    # train the PINN
    nn_approximator_trained = train_model(
        hidden_layer_n=hidden_layer_n,
        layer_width=layer_width,
        boundary_s_n=boundary_s_n,
        boundary_tau_max_n=boundary_tau_max_n,
        collocation_tau_phi=collocation_tau_phi,
        collocation_tau=collocation_tau,
        collocation_s=collocation_s,
        tau_max=tau_max,
        processes=processes,
        percentage_training_data=percentage_training_data,
        batch_size_collocation=batch_size_collocation,
        batch_size_iv_s0=batch_size_iv_s0,
        batch_size_bv_sL=batch_size_bv_sL,
        batch_size_bv_sL_data=batch_size_bv_sL_data,
        batch_size_bv_tau_max=batch_size_bv_tau_max,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        save_reference=save_reference,
        load_reference=load_reference,
        ref_name=ref_name,
        phi_range=phi_range,
        name=name_nn,
        importance_sampling=importance_sampling,
        tau_centers_n = tau_centers_n,
        s_centers_n = s_centers_n,
        s_scale =s_scale,
        tau_scale = tau_scale,
        epoch_resume_training=epoch_resume_training,
        nn_opt_name=nn_opt_name
    )

    # Save approximator and samples/collocation points
    approximator_name = f"state_dict_phi{phi_range[0]}" \
                        f"_{phi_range[1]}__{name_nn}"
    try:
        torch.save(nn_approximator_trained.state_dict(),
                   "./state_dicts/" + approximator_name + ".pth")
    except:
        os.mkdir("./state_dicts/")
        torch.save(nn_approximator_trained.state_dict(),
                   "./state_dicts/" + approximator_name + ".pth")

