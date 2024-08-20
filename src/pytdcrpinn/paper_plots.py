import time
from evaluate_model import evaluate_euclid
import numpy as np
import robot_specs as rs
from pytdcrsv.static_robot_model import StaticRobotModel
import pytdcrsv.mp_static_robot_model as mpsrm
import matplotlib.pyplot as plt
import datasl as dsl
import tqdm
import torch
import os
import json
import pinn_nn as pnn
import pandas as pd
import pickle

import scienceplots

plt.style.use('ieee')
plt.rcParams.update({'figure.dpi': '300'})

def load_evaluation_data(ref_name=None, name=None):
    """
    Loads evaluation_data

    Args:
        ref_name: Name of the reference data
        name: name of the nn configuration

    Returns:
        used collocation points, test/val/train data, config
    """
    # Load validation data
    rr = dsl.RunReference()

    rr.load_from_disk(name=ref_name,
                      name_config=name + "_config.json")

    ds_collocation = dsl.CollocationPoints(rr,
                                   collocation_tau_phi=rr.collocation_tau_phi,
                                   collocation_tau=rr.collocation_tau,
                                   collocation_s=rr.collocation_s)

    return ds_collocation, rr.test_val_train, rr.config


def generate_evaluation_actuation(ref_name=None, name=None, df=0.1, dphi=1.,
                                        testing_data=True,
                                    validation_data=True,
                                    training_data=False,
                                  fr_low=0.5,
                                  act_name=""):

    ds_collocation, test_val_train, config = load_evaluation_data(
                                                            ref_name=ref_name,
                                                            name=name)
    rr = dsl.RunReference()
    phi_range = config["phi_range"]
    tau_max = config["tau_max"] -0.011
    diff_phi = phi_range[1] - phi_range[0]
    phi_n = int(diff_phi / dphi)
    fr_n = int(tau_max / df)
    act_np = rr.generate_actuations_grid(phi_range, phi_n, tau_max,
                                         fr_n,
                                         fr_low=fr_low).reshape(-1, 6)

    # Now, remove unwanted actuations
    data_type = [training_data, validation_data, testing_data]
    data_key = ["training", "validation", "testing"]
    indices = []
    for dt, dk in zip(data_type, data_key):
        if dt:
            act_dt = test_val_train[dk]
            act_dt_full = torch.concat([act_dt["i0"].dataset.act_torch,
                                        act_dt["sL"].dataset.act_torch,
                                        act_dt["dataSL"].dataset.act_torch,
                                        act_dt["tau_max"].dataset.act_torch])
            act_dt_np = act_dt_full.numpy()
            act_dt_np = np.concatenate([np.zeros_like(act_dt_np), act_dt_np],
                                  axis=1)
            for idx, act in tqdm.tqdm(enumerate(act_np)):
                for act_dt_ in act_dt_np:
                    d = act - act_dt_
                    if np.sqrt(d[3]**2) < 1e-2 and \
                        np.sqrt(d[4]**2) < 1e-2 and \
                        np.sqrt(d[5] ** 2) < 1e-2:
                        indices.append(idx)
                        break

    # Remove actuations
    mask = [True] * act_np.shape[0]
    for idx in indices:
        mask[idx] = False

    act_np_new = act_np[mask]

    # Store actuations to disk
    name_plot_acts = "plot_acts" + act_name
    print("./evaluate_models/" + name_plot_acts + ".npy")
    try:
        np.save("./evaluate_models/" + name_plot_acts + ".npy", act_np_new)
    except:
        os.mkdir("./evaluate_models/")
        np.save("./evaluate_models/" + name_plot_acts + ".npy", act_np_new)
    print(f"Generated {act_np_new.shape[0]} actuations")

    return rr.tau_max_generated, rr.tau_max

def compute_reference_to_pkl_and_profile(path, runs=2, eps_sz=None):
    # Load actuations
    acts = np.load(path + "plot_acts.npy").reshape(-1)
    time_dict = {}
    for esz in tqdm.tqdm(eps_sz):
        print("eps sz ", esz)
        print("")
        for n in tqdm.tqdm(range(runs)):
            try:
                robot_vc = StaticRobotModel(
                    segment_length__m=np.array(
                        [float(rs.CCRnp.L) / 2, float(rs.CCRnp.L) / 2]),
                    youngs_modulus__n_per_m2=rs.CCRnp.E,
                    pradius_disks__m=np.array([rs.CCRnp.r1, rs.CCRnp.r1]),
                    ro__m=rs.CCRnp.r,
                    modelling_approach="VC",
                    f__n=rs.CCRnp.f_ext,
                    gxf_tol=esz[0],
                    step_size=esz[1]
                    )

                robot_vc.use_recomputation = True
                success = robot_vc.calc_pose_from_ctrl(act=acts)
                time_dict[n] = robot_vc.time_dict
            except Exception as e:
                print(f"Error during calculation.{esz}")
                print(e)

        try:
            with open("./evaluate_models/pos_rt/"  +
                      f"_runtime_lst_ref_{esz[0]}_{esz[1]}.json",
                      "w") as file:
                json.dump(time_dict, file)
        except:
            os.mkdir("evaluate_models/pos_rt/")
            with open("./evaluate_models/pos_rt/" +
                      f"_runtime_lst_ref_{esz[0]}_{esz[1]}.json",
                      "w") as file:
                json.dump(time_dict, file)

        frames = robot_vc.frames_df
        pkl_path = "./evaluate_models/pos_rt/" + f"df_ref_{esz[0]}_"\
                    + f"{esz[1]}.pkl"
        frames.to_pickle(path=pkl_path)


def compute_nn_to_npy_and_profile(path, nn_path, eval_n=20,
                                  disks_n=21,layer=4, width=250,
                                  tau_max_generated=2.8):
    # Load actuations
    acts = np.load(path +"plot_acts.npy")

    # Generate s points for each actuation
    s = torch.linspace(0, rs.CCRnp.L, disks_n).reshape(-1,1,1)
    # Generate input tensor, by concatenating s with actuations
    s_acts = []
    s_full = []
    for a in acts:
        a_long = torch.from_numpy(a[3:].astype("float32")).reshape(1, 3, 1) * \
              torch.ones((s.shape[0], 3, 1))
        s_a = torch.concat([s, a_long], dim=1)
        s_acts.append(s_a)
        s_full.append(s)

    s_act_full = torch.concat(s_acts, dim=0).reshape(-1, 4)
    s_full_torch = torch.concat(s_full, dim=0).reshape(-1, 1)

    nn = pnn.NNApproximator(layer, width,
                            max_tau=tau_max_generated,
                            )
    try:
        nn.load_state_dict(torch.load(nn_path))
    except:
        nn.load_state_dict(torch.load(nn_path)["model_state_dict"])
    def eval(mod, inp):
       # mod.eval()
        with torch.no_grad():
            return pnn.f(mod, inp)

    output = pnn.f(nn, s_act_full)
    time_dict = {}
    with torch.no_grad():
        nn.eval()
        for n in range(eval_n):
            time_s = time.time()
            output = pnn.f(nn, s_act_full)
            time_e = time.time()
            time_dict[n] = time_e - time_s
    print(time_dict)
    df_dict = {"nn_rt": time_dict}
    # Store time array
    try:
        with open("./evaluate_models/pos_rt/" + "_runtime_lst_nn.json",
                  "w") as file:
            json.dump(df_dict, file)
    except:
        os.mkdir("evaluate_models/pos_rt/")
        with open("./evaluate_models/pos_rt/" + "_runtime_lst_nn.json",
                  "w") as file:
            json.dump(df_dict, file)

    # store positions
    p_torch = output[:,:3].reshape(-1, 3)

    p_s_act_torch = torch.concat([s_full_torch, p_torch, s_act_full[:,1:]],
                               dim=1)

    p_np = p_s_act_torch.numpy()
    name_plot_acts = "s_p_nn"
    try:
        np.save("./evaluate_models/pos_rt/" + name_plot_acts + ".npy", p_np)
    except:
        os.mkdir("evaluate_models/pos_rt/")
        np.save("./evaluate_models/pos_rt/" + name_plot_acts + ".npy", p_np)

    return nn, acts


def compare_models_precision(path="./evaluate_models/pos_rt/",
                            nn_s_pos_npy=None, acts=None, eps_sz=None,
                             start_disk=1, each_n_disk=1):
    ref_df_file_lst = [f"df_ref_{esz[0]}_{esz[1]}.pkl" for esz in eps_sz]
    df_dict = {}
    # Load all dataframes
    for fname, esz in zip(ref_df_file_lst, eps_sz):
        df = pd.read_pickle(path + fname)
        df_dict[f"esz{esz[0]}_{esz[1]}_euclid"] = df

    df_pos_lst = []
   # acts = np.load(acts)

    for df_key in df_dict:
        # Get positions from all frames for row_i actuation
        df = df_dict[df_key]
        x_ = []
        y_ = []
        z_ = []
        s_ = []
        act_ = []
        disks = [n for n in range(start_disk, 21, each_n_disk)]
        for row_i in range(len(df)):
            row = df.iloc[row_i]
            for n in range(df.shape[1] - 4):
                a_ = acts[row_i]
                assert np.linalg.norm(a_ - row["ctrl"]) < 1e-2

                column = "disk_" + f"{n:02d}"
                if n == (df.shape[1] - 4) - 1:
                    column = "ee"

                if n not in disks:
                    continue
                state = row[column]
                x_.append(state[0])
                y_.append(state[1])
                z_.append(state[2])
                s_.append(n * (rs.CCRnp.L / (df.shape[1] - 5)))
                act_.append(row["ctrl"][3:])

        df_pos_lst.append(pd.DataFrame({"s": s_,
                                        "x": x_,
                                        "y": y_,
                                        "z": z_,
                                        "act": act_}))

    positions_nn = np.load(path + nn_s_pos_npy)
    nn_df = pd.DataFrame(positions_nn,columns=["s", "x", "y", "z",
                                               "t1", "t2", "t3"])

    for df_p, name in zip(df_pos_lst, ref_df_file_lst):
        picle_path = ("./evaluate_models/dX/" + name[:-4] +
                      "_df_processed.pkl")
        df_p.to_pickle(path=picle_path)

    pickle_path = ("./evaluate_models/dX/" + nn_s_pos_npy[:-4] +
              "_df_processed.pkl")
    nn_df.to_pickle(path=pickle_path)

    # Assuming the first entry in the ref_df_file_lst is the reference model
    ref_model = df_pos_lst[0]

    # Compute accuracy of nn (compared to ref_model)
    mask_ = [False] * 21
    for n in range(start_disk, 21, each_n_disk):
        mask_[n] = True
    mask = []
    for n in range(len(acts)):
        mask += mask_
    acc_df = pd.DataFrame()
    nn_df_masked = nn_df[mask].copy().reset_index()
    acc_df["nn_dx_sq"] = (ref_model["x"] - nn_df_masked["x"]).pow(2)
    acc_df["nn_dy_sq"] = (ref_model["y"] - nn_df_masked["y"]).pow(2)
    acc_df["nn_dz_sq"] = (ref_model["z"] - nn_df_masked["z"]).pow(2)
    acc_df["nn_ds"] = ref_model["s"] - nn_df_masked["s"]
    acc_df["nn_euclid"] = (acc_df["nn_dx_sq"] + acc_df["nn_dy_sq"]
                           + acc_df["nn_dz_sq"]).pow(0.5)
    acc_df["act"] = ref_model["act"]

    # Compare ref model with lower accuracy model
    for es, df in zip(eps_sz[1:],df_pos_lst[1:]):
        assert len(ref_model) == len(df)
        assert (df["act"] - ref_model["act"]).pow(2).sum().sum() < 1e-6
        acc_df[f"esz{es[0]}_{es[1]}_dx_sq"] = (ref_model["x"] - df["x"]).pow(2)
        acc_df[f"esz{es[0]}_{es[1]}_dy_sq"] = (ref_model["y"] - df["y"]).pow(2)
        acc_df[f"esz{es[0]}_{es[1]}_dz_sq"] = (ref_model["z"] - df["z"]).pow(2)
        acc_df[f"esz{es[0]}_{es[1]}_euclid"] = (
                                            acc_df[f"esz{es[0]}_{es[1]}_dx_sq"]
                                          + acc_df[f"esz{es[0]}_{es[1]}_dy_sq"]
                                          + acc_df[f"esz{es[0]}_{es[1]}_dz_sq"]
                                              ).pow(0.5)
    acc_df["s"] = ref_model["s"]
    acc_df["act"] = ref_model["act"]

    pickle_path = ("./evaluate_models/dX/"+
                   "eval_df_processed.pkl")
    acc_df.to_pickle(path=pickle_path)
    return acc_df

def compare_models_runtime(path="./evaluate_models/", ref_rt=None,
                            nn_rt=None, eps_sz=None):
    # Load runtime dictionaries
    df_lst = []
    # Load all dataframes
    for fname in ref_rt:
        with open(path + fname, "r") as file:
            # each df holds n runs of the respective model, holding m
            # runtimes, since each model is run m times.
            df = json.load(file)
            df_lst.append(df)

    nn_df = pd.read_json(path + nn_rt)

    # Build data frame
    df_runtime_lst = []
    for df, esz in zip(df_lst, eps_sz[1:]):
        # Go through all runs for one model configuration
        runtimes_lst = []
        poses_lst = []
        for run in df:
            # sum all runtimes and poses
            pose_sum = 0
            runtime_sum = 0
            for k in df[run]:
                pose_sum += df[run][k]["poses_n"]
                runtime_sum += df[run][k]["seconds"]
            runtimes_lst.append(runtime_sum)
            poses_lst.append(pose_sum)
        df_runtime_lst.append(pd.DataFrame(
                                   {f"eps{esz[0]}_sz{esz[1]}_rt":runtimes_lst,
                                    f"eps{esz[0]}_sz{esz[1]}_poses": poses_lst}
                                         )
                             )
    runtime_df = pd.concat(df_runtime_lst, axis=1)
    return runtime_df, nn_df


def performance_plot(runtime_ref_df, runtime_nn_df, dX_df, eps_sz):
    df_full_rt = pd.concat([runtime_ref_df, runtime_nn_df], axis=1)

    df_precision_lst = []
    df_runtime_dict = {}
    eps_sz_dict = {}
    for esz in eps_sz[1:]:
        name_euclid = f"esz{esz[0]}_{esz[1]}_euclid"
        name_rt = f"eps{esz[0]}_sz{esz[1]}_rt"
        df_precision_lst.append(dX_df[name_euclid])
        df_runtime_dict[name_euclid] = df_full_rt[name_rt].mean()
        eps_sz_dict[name_euclid] = esz

    df_precision_lst.append(dX_df["nn_euclid"])
    df_precision_lst.append(dX_df["act"])
    df_precision = pd.concat(df_precision_lst, axis=1)
    df_runtime_dict["nn_euclid"] = runtime_nn_df["nn_rt"].mean()


    # Plot results
    ## Median

    fig, ax = plt.subplots()

    switch = True
    min_rt = 10
    for k, esz in zip(df_precision, eps_sz):
        if k == "act":
            continue
        x_rt = [df_runtime_dict[k]] * df_precision.shape[0]
        y_rt = df_precision[k].to_list()
        if x_rt[0] < min_rt:
            min_rt = x_rt[0]
        if k == "nn_euclid":
            ax.plot(x_rt, y_rt, '+', c="r", alpha=0.05, markersize=.5)
        else:
            if switch:
                ax.plot(x_rt, y_rt, 'v', c='b', alpha=0.05, markersize=.5)
                switch = False
            else:
                ax.plot(x_rt, y_rt, '^', c='b', alpha=0.05, markersize=.5)
                switch = True
    print(df_runtime_dict)
    for n, k in enumerate(df_precision):
        if k == "act":
            continue
        x_rt = df_runtime_dict[k]
        y_rt = df_precision[k].median()
        if n == len(df_runtime_dict) - 1:
            ax.plot(x_rt, y_rt,"_",markersize=8,c="black",
                       label=r"Median",
                )
        else:
            ax.plot(x_rt, y_rt, "_", markersize=10,c="black"
                    )

    ax.set_yscale("log")
    ax.set_xscale("log")
    # Draw 1% boundary
    lns3 = ax.axhline(0.002, c='black', linestyle='--', alpha=0.5,
                      label="1 \% Deviation")
    # Set axes limits
    max_rt = 0.
    min_rt = 100
    for k in df_runtime_dict:
        if k == "act":
            continue
        if df_runtime_dict[k] > max_rt:
            max_rt = df_runtime_dict[k]
        if df_runtime_dict[k] < min_rt:
            min_rt = df_runtime_dict[k]

    ax.set_xlim([0.5 * min_rt, max_rt * 1.2])
    ax.set_ylim([5e-9, dX_df["nn_euclid"].max() * 1.2])

    # Set y label
    ax.set_ylabel(r"Error $\| \vec{x}_{ref} - \vec{x}_i\|$ [m]")
    ax.set_xlabel(r"Computation time [s]")

    ax.legend()
    ax.grid(True, axis="both", ls="--", linewidth=0.2, which="both")
    fig.tight_layout()
    plt.tight_layout()
    plt.savefig("PrecisionRT.png",
             pad_inches=0.1,
            facecolor='auto', edgecolor='auto'
            )


def compute_evaluate_models_precision():
    # Load actuations
    acts = np.load(path + "plot_actsprecision.npy").reshape(-1)
    # Read dictionary pkl file
    print("")
    robot_vc = StaticRobotModel(
        segment_length__m=np.array(
            [float(rs.CCRnp.L) / 2, float(rs.CCRnp.L) / 2]),
        youngs_modulus__n_per_m2=rs.CCRnp.E,
        pradius_disks__m=np.array([rs.CCRnp.r1, rs.CCRnp.r1]),
        ro__m=rs.CCRnp.r,
        modelling_approach="VC",
        f__n=rs.CCRnp.f_ext,
        gxf_tol=1e-11,
        step_size=1e-4
    )

    frames_multi = mpsrm.multi_processing_calc_pose_from_ctrl(robot_vc,
                                                              acts,
                                                              5)
    result_dict = mpsrm.convert_multiprocessing_result_into_dict(frames_multi)
    robot_vc.use_recomputation = True
    pkl_path = "evaluate_models/pos_rt/precision.pkl"
    with open(pkl_path, 'wb') as fp:
        pickle.dump(result_dict, fp)
        print('dictionary saved successfully to file')


def evaluate_models_precision(nn_path, tau_max_generated=2.8, layer=3, width=100):
    pkl_path = "evaluate_models/pos_rt/precision.pkl"
    with open(pkl_path, 'rb') as fp:
        mpresdict = pickle.load(fp)

    nn = pnn.NNApproximator(layer, width,
                            max_tau=tau_max_generated
                            )
    nn.load_state_dict(torch.load(nn_path))

    # Evaluate nn per process
    s = torch.linspace(0, 0.2, 21).reshape(-1,1)
    # Results for all processes
    res_per_process = []
    for act, frame in zip(mpresdict["actuations"], mpresdict["frames"]):
        act_reshape = act.reshape((-1, 6))
        s_act = act_reshape[:, 3:].astype("float32")
        res_per_act = []
        # Build actuation for one shape
        for row in range(s_act.shape[0]):
            act_lst = [torch.from_numpy(s_act[row,:]).reshape(-1, 3)] * len(s)
            act_torch = torch.concat(act_lst, dim=0)
            s_act_torch = torch.concat([s, act_torch], dim=1)
            # predict shape
            with torch.no_grad():
                nn.eval()
                pRuv_1act_shape = pnn.f(nn, s_act_torch).numpy()
            # Compare shape with reference
            # Run through disks
            res_per_shape =[]
            for n in range(int(frame.shape[1] / 4)):
                pos_ref = frame[4 * row:4 * row + 3, 3 + 4 * n]
                pos_nn = pRuv_1act_shape[n, :3]
                euclid = np.linalg.norm(pos_ref - pos_nn)
                # Results for each disk for one actuation
                res_per_shape.append(euclid)
            # Results for each actuation
            res_per_act.append(res_per_shape)
        res_per_process.append(res_per_act)

    # Collect everything in a dataframe
    res_df_lst = []
    # run through all processes
    for a, res_proc in zip(mpresdict["actuations"], res_per_process):
        act_reshape = a.reshape((-1, 6))
        act = act_reshape[:, 3:]
        # run through actuations
        for an, act_res in enumerate(res_proc):
            actuation = act[an,:].reshape(1,3)
            act_shape_long = np.concatenate([actuation] * len(act_res),axis=0)
            t1 = act_shape_long[:,0]
            t2 = act_shape_long[:,1]
            t3 = act_shape_long[:, 2]
            res_df_lst.append(pd.DataFrame({"s": np.linspace(0, rs.CCRnp.L,
                                                             len(act_res)),
                                            "t1": t1,
                                            "t2": t2,
                                            "t3": t3,
                                            "euclid": act_res
                                            }
                                           )
                              )

    res_df = pd.concat(res_df_lst, axis=0, ignore_index=True)
    res_df.to_pickle("./evaluate_models/precision_df.pkl")
    return res_df


if __name__ == "__main__":
    path = "./evaluate_models/"
    name = "main69_vconst_cid_-1000no_force_vconst"
    name = "main76_pub_prec_ref"
    #ref_name = "bsn192_btm192_phirange_300_360_taum2.51_forces_pairing_v2"
    #ref_name = "bsn96_btm96_phirange_300_360_taum3.51_"
    #ref_name_train ="bsn384_btm192_phirange_0_10_taum2
    # .51forces_pairing_v2_1e-11_1e-4"
    nn_path = ("./evaluate_models/main69_vconst_cid_"
               "-1000no_force_vconst_10_27_26.pth")
    nn_path = ("./evaluate_models/main76_resume400_cid_"
               "-1000prec_ref_vconst_04_16_02.pth")
    nn_path = ("./state_dicts/main76_cid_-1000prec_ref_vconst_4"
               ".338_epoch_76002_09_11nn_opt.pth")
    nn_path = ("./state_dicts/main76_cid_-1000prec_ref_vconst_3"
               ".887_epoch_108922_04_48nn_opt.pth")
    nn_path =("./state_dicts/main76_pub_prec_ref__epoch_780_17_22_49_24"
              ".832nn_opt.pth")

    nn_path = ("./state_dicts/run_training_prec_ref__epoch_5310_09_14_20_15.322nn_opt.pth")
    nn_path = ("./state_dicts/run_training_prec_ref__epoch_6070_19_38_49_15"
               ".244nn_opt.pth")

    ref_name = "bsn96_btm96_phirange_300_360_taum3.51_"
    layer = 3
    width = 100
    eps_sz = [(1e-11, 1e-4),
              (1e-4, 0.001),
              (1e-4, 0.005),
              (1e-3, 0.01),
              (5e-3, 0.01),
              (1e-2, 0.01),
              ]

    tau_max_generated, tau_max = generate_evaluation_actuation(
                                      ref_name=ref_name, name=name,
                                      df=.125, dphi=10, fr_low=0.2,
                                      training_data=True,
                                      validation_data=False,
                                      testing_data=False)

    if False:
        compute_reference_to_pkl_and_profile(path=path,
                                             runs=5,
                                             eps_sz=eps_sz)

    nn, acts = compute_nn_to_npy_and_profile(path=path,
                                             nn_path=nn_path, eval_n=5,
                                             layer=layer, width=width,
                                             tau_max_generated=tau_max_generated)

    nn_s_pos_npy = ("s_p_nn.npy")
    dX_df = compare_models_precision(path="evaluate_models/pos_rt/",
                                     nn_s_pos_npy=nn_s_pos_npy,
                                     acts=acts,
                                     eps_sz=eps_sz,
                                     start_disk=1, each_n_disk=1)

    ref_rt = [f"_runtime_lst_ref_{esz[0]}_{esz[1]}.json" for esz in eps_sz[
     1:]]
    nn_rt = ("_runtime_lst_nn.json")
    runtime_ref_df, runtime_nn_df = compare_models_runtime(
                                                path="evaluate_models/pos_rt/",
                                               ref_rt=ref_rt,
                                               nn_rt=nn_rt, eps_sz=eps_sz)

    performance_plot(runtime_ref_df, runtime_nn_df, dX_df, eps_sz)

    plt.show()

