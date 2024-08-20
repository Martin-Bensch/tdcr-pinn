import pandas as pd
from pytdcrsv.static_robot_model import StaticRobotModel
import torch
import numpy as np
from tdcrpinn.icra2024.pinn_nn import NNApproximator
import tdcrpinn.icra2024.datasl as dsl
from torch.utils.data import Dataset, DataLoader
import tdcrpinn.icra2024.robot_specs as rs
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'figure.autolayout': True
})


def _load_data_for_evaluation(folder=None, filename=None,
                             data_type="validation",
                              name_config="", folder_config=None,
                              ds_collocation_given=False):
    """
    Loads training, validation and testing data from disk

    Args:
        folder: path to the folder containing data
        filename: name of the approximator.

    Returns:

    """

    # Instantiate RunReference object, since data loading is already
    # implemented there.
    rr = dsl.RunReference(verbose=False)
    rr.load_from_disk(folder=folder, name=filename, verbose=False,
                      name_config=name_config, folder_config=folder_config,
                      )

    collocation_tau_phi = rr.collocation_tau_phi
    collocation_tau = rr.collocation_tau
    collocation_s = rr.collocation_s

    if not ds_collocation_given:
        ds_collocation = dsl.CollocationPoints(rr,
                                           collocation_tau_phi=collocation_tau_phi,
                                           collocation_tau=collocation_tau,
                                           collocation_s=collocation_s)
    else:
        ds_collocation = None

    data_return = rr.test_val_train[data_type]

    return data_return, ds_collocation, rr.tau_max_generated, rr.tau_max


def evaluate_euclid(nn_model, data_type="validation",
                    folder="./", filename=None,
                    name_config="", folder_config=None,
                    phi_range=(0,30),scatter_plot=True,
                    heatmap=True,
                    shape=True,
                    collocation_points=True,
                    alpha=0.3, ds_collocation=None,
                    close=True):
    """
    Evaluates PINN.

    Args:
        nn_model: trained model
        test_data:
                test/validation data that was generated during training.
    Returns:
        None
    """
    tendons = [0, 0]
    if 0 <= phi_range[0] <= 120:
        tendons[0] = 1
        tendons[1] = 3
    elif 120 < phi_range[0] <= 240:
        tendons[0] = 3
        tendons[1] = 2
    elif 240 < phi_range[0] < 360:
        tendons[0] = 2
        tendons[1] = 1

    nn_model.eval()
    with (((torch.no_grad()))):
        if ds_collocation is not None:
            data_dict, _, tau_max_generated, tau_max = _load_data_for_evaluation(
                                                      folder=folder,
                                                       filename=filename,
                                                      data_type=data_type,
                                                      name_config=name_config,
                                                      folder_config=folder_config)
            gaussian_collocation = True
        else:
            data_dict, ds_collocation, tau_max_generated, tau_max =_load_data_for_evaluation(
                                                        folder=folder,
                                                     filename=filename,
                                                     data_type=data_type,
                                                     name_config=name_config,
                                                     folder_config=folder_config)
            gaussian_collocations = False

        # Run nn for the test data
        # i0, sL, data
        s_act_ref_p_nn_p_sLs0data = []
        for dtype in data_dict:
            if dtype is "test_bench":
                continue
            i0sLdata = data_dict[dtype]
            # Change batch size
            dl = DataLoader(i0sLdata, batch_size=len(i0sLdata))
            # Make it an iterator
            dl_iter = iter(dl)
            # Get data s_act, reference
            s_act, ref = next(dl_iter)
            assert s_act.shape[0] == len(i0sLdata)
            # Use nn for prediction on s_act
            nn_res_i0 = nn_model(s_act).detach().clone()

            tmp_dct = {"s": s_act[:, 0].numpy(),
                       "t1": s_act[:, 1].numpy(),
                       "t2": s_act[:, 2].numpy(),
                       "t3": s_act[:, 3].numpy(),
                       "x_ref": ref[:, 0].reshape(-1,).numpy(),
                       "y_ref": ref[:, 1].reshape(-1,).numpy(),
                       "z_ref": ref[:, 2].reshape(-1,).numpy(),
                       "x_nn": nn_res_i0[:,0].numpy(),
                       "y_nn": nn_res_i0[:,1].numpy(),
                       "z_nn":nn_res_i0[:,2].numpy(),
                       "datatype": [dtype] * len(nn_res_i0[:,0])
                       }
            tmp_df = pd.DataFrame.from_dict(tmp_dct)
            s_act_ref_p_nn_p_sLs0data.append(tmp_df)

        res_df = pd.concat(s_act_ref_p_nn_p_sLs0data, axis=0, ignore_index=True)
        res_df["x_dist"] = res_df["x_ref"] - res_df["x_nn"]
        res_df["y_dist"] = res_df["y_ref"] - res_df["y_nn"]
        res_df["z_dist"] = res_df["z_ref"] - res_df["z_nn"]
        res_df["square_sum"] = res_df["x_dist"].pow(2) +  res_df["y_dist"].pow(2)\
                               + res_df["z_dist"].pow(2)

        res_df["euclid"] = res_df["square_sum"].pow(1./2.)

        # Plot multiple visualisations
        # For each tendon combination find max, min, median
        for index, row in res_df.iterrows():
            flt_tmp = res_df[(res_df[f"t{tendons[0]}"] == row[f"t{tendons[0]}"]) &
                             (res_df[f"t{tendons[1]}"] == row[f"t{tendons[1]}"])]

            max_tmp = flt_tmp["euclid"].max()
            min_tmp = flt_tmp["euclid"].min()
            median_tmp = flt_tmp["euclid"].median()

            res_df.loc[(res_df[f"t{tendons[0]}"] == row[f"t{tendons[0]}"]) &
                    (res_df[f"t{tendons[1]}"] == row[f"t{tendons[1]}"]),
                    "min"] = min_tmp
            res_df.loc[(res_df[f"t{tendons[0]}"] == row[f"t{tendons[0]}"]) &
                    (res_df[f"t{tendons[1]}"] == row[f"t{tendons[1]}"]),
                    "max"] = max_tmp
            res_df.loc[(res_df[f"t{tendons[0]}"] == row[f"t{tendons[0]}"]) &
                    (res_df[f"t{tendons[1]}"] == row[f"t{tendons[1]}"]),
                    "median"] = median_tmp

        figs = []
        fig_names = []
        ax_hdl = []
        if scatter_plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(res_df[f"t{tendons[0]}"], res_df[f"t{tendons[1]}"],
                            res_df['max'], marker="+")
            ax.scatter(res_df[f"t{tendons[0]}"], res_df[f"t{tendons[1]}"],
                            res_df['median'])
            ax.scatter(res_df[f"t{tendons[0]}"], res_df[f"t{tendons[1]}"],
                            res_df['min'], marker="v")
            ax.set_xlabel(rf"Tendon $t_{tendons[0]}$")
            ax.set_ylabel(rf"Tendon $t_{tendons[1]}$")
            ax.set_zlabel("Euclidean Distance")
            ax.set_zlim([0, 0.005])
            ax_hdl.append(ax)
            figs.append(fig)
            fig_names.append("scatter plot")

        if heatmap:
            fig2, (a1, a2, a3) = plt.subplots(1,3, sharey="all",
                                              tight_layout=True,
                                              figsize=(13,3.9))

            a1_r = a1.scatter(res_df[f"t{tendons[0]}"], res_df[f"t{tendons[1]}"],
                       c=res_df['max'], s=5)
            a2_r = a2.scatter(res_df[f"t{tendons[0]}"], res_df[f"t{tendons[1]}"],
                       c=res_df['median'], s=5)
            a3_r = a3.scatter(res_df[f"t{tendons[0]}"], res_df[f"t{tendons[1]}"],
                       c=res_df['min'], s= 5)
            for a_r, a, title in zip([a1_r, a2_r, a3_r], [a1, a2, a3],
                                     ["Maximal", "Median", "Minimal"]):
                ax_hdl.append(a)
                a.set_aspect("equal")
                fig2.colorbar(a_r, ax=a, location="right", shrink=0.82)
                a.set_title(title + " Euclidean Distance")

            fig2.supxlabel(rf"Tendon $t_{tendons[0]}$")
            fig2.supylabel(rf"Tendon $t_{tendons[1]}$")
            figs.append(fig2)
            fig_names.append("heat map")

        if shape:
            # Get the worst actuation
            max_euclid = res_df["max"].max()
            min_euclid = res_df["min"].min()
            median_euclid = res_df["median"].median()

            # Max: Get reference positions
            xyz_ref_max = res_df.loc[res_df["max"] == max_euclid,
                                    ["s","x_ref", "y_ref", "z_ref"]].sort_values("s")
            xyz_nn_max = res_df.loc[res_df["max"] == max_euclid,
                                    ["s","x_nn", "y_nn", "z_nn"]].sort_values("s")
            fig3 = plt.figure()
            ax4 = fig3.add_subplot(111, projection="3d")
            ax4.scatter(xyz_ref_max["x_ref"],xyz_ref_max["y_ref"],
                        xyz_ref_max["z_ref"], label="Ref. Max Euclid",
                        s=15, color="orange", depthshade=0, alpha=alpha)

            ax4.scatter(xyz_nn_max["x_nn"], xyz_nn_max["y_nn"],
                        xyz_nn_max["z_nn"], marker="+", color="orange",
                        label="NN Max Euclid", depthshade=0)

            ## Compute robot model for worst actuation
            tendons_max = res_df.loc[res_df["max"] == max_euclid,
                                     ["s", "t1", "t2", "t3"]].sort_values("s")
            # Check if all entries are equal in each column
            if not tendons_max["t1"].max() == tendons_max["t1"].min() or \
                not tendons_max["t1"].max() == tendons_max["t1"].min() or \
                not tendons_max["t1"].max() == tendons_max["t1"].min():
                print("wtf?")

            t1 = tendons_max["t1"].max()
            t2 = tendons_max["t2"].max()
            t3 = tendons_max["t3"].max()
            act_max = np.array([0, 0, 0, t1, t2, t3])
            L = rs.CCRnp.L
            robot_vc = StaticRobotModel(segment_length__m=np.array([L / 2, L / 2]),
                                        youngs_modulus__n_per_m2=rs.CCRnp.E,
                                        pradius_disks__m=np.array(
                                            [rs.CCRnp.r1, rs.CCRnp.r1]),
                                        ro__m=rs.CCRnp.r,
                                        modelling_approach="VC",
                                        f__n=rs.CCRnp.f_ext,
                                        gxf_tol=1e-6,
                                        step_size=1e-2
                                        )
            robot_vc.use_recomputation = True
            success = robot_vc.calc_pose_from_ctrl(act_max)
            if np.alltrue(success):
                frames_max = robot_vc.disk_frames_array
                # Get all disk positions
                disks_n = frames_max.shape[1] / 4
                disks_x = []
                disks_y = []
                disks_z = []
                for n in range(int(disks_n)):
                    disk_idx = 4 * n
                    p_tmp = frames_max[0:3,disk_idx + 3]
                    disks_x.append(p_tmp[0])
                    disks_y.append(p_tmp[1])
                    disks_z.append(p_tmp[2])
                ax4.plot3D(disks_x, disks_y, disks_z,'-',alpha=alpha,
                           color="orange")

                disks_x_ts = [disks_x[-1], disks_x[-1]]
                disks_y_ts = [disks_y[-1], disks_y[-1]]
                disks_z_ts = [disks_z[-1], disks_z[0]]

                ax4.plot3D(disks_x_ts, disks_y_ts, disks_z_ts,  "--",
                           alpha =alpha, color="orange")
                disks_x_ts = [ disks_x[-1], rs.CCRnp.L / 1.5]
                disks_y_ts = [ disks_y[-1], disks_y[-1]]
                disks_z_ts = [disks_z[0], disks_z[0]]

                ax4.plot3D(disks_x_ts, disks_y_ts, disks_z_ts, "--",
                           alpha=alpha, color="orange")

                disks_x_ts = [disks_x[-1], disks_x[-1]]
                disks_y_ts = [disks_y[-1], -rs.CCRnp.L / 1.5]
                disks_z_ts = [disks_z[0], disks_z[0]]

                ax4.plot3D(disks_x_ts, disks_y_ts, disks_z_ts, "--",
                           alpha=alpha, color="orange")

                disks_x_ts = [disks_x[-1], -rs.CCRnp.L / 1.5]
                disks_y_ts = [disks_y[-1], disks_y[-1]]
                disks_z_ts = [disks_z[-1], disks_z[-1]]

                ax4.plot3D(disks_x_ts, disks_y_ts, disks_z_ts, "--",
                           alpha=alpha, color="orange")

                s_torch = torch.linspace(0, float(rs.CCRnp.L), 21).reshape(
                                                                        -1, 1, 1)
                act_torch = torch.from_numpy(act_max[3:].astype(
                                "float32")).reshape(1, 3, 1) * torch.ones_like(
                                s_torch)
                s_act_torch = torch.cat([s_torch, act_torch],
                                       dim=1).reshape(-1, 4)
                pRnm_eval = nn_model(s_act_torch).detach().numpy()
                p = pRnm_eval[:, :3]
                ax4.plot3D(p[:, 0], p[:, 1], p[:, 2], '+', color="orange")

            #-----------------------------
            # Min: Get reference positions
            xyz_ref_min = res_df.loc[res_df["min"] == min_euclid,
                                ["s","x_ref", "y_ref","z_ref"]].sort_values("s")
            xyz_nn_min = res_df.loc[res_df["min"] == min_euclid,
                                    ["s","x_nn", "y_nn", "z_nn"]].sort_values("s")
            ax4.scatter(xyz_ref_min["x_ref"], xyz_ref_min["y_ref"],
                        xyz_ref_min["z_ref"],"-*", label="Ref. Min Euclid",
                       alpha=alpha, s=15, color="blue", depthshade=0)

            ax4.scatter(xyz_nn_min["x_nn"], xyz_nn_min["y_nn"],
                        xyz_nn_min["z_nn"], marker="+", label="NN Min Euclid",
                        color="blue", depthshade=0)

            ## Compute robot model for best actuation
            tendons_min = res_df.loc[res_df["min"] == min_euclid,
                                     ["s", "t1", "t2", "t3"]].sort_values("s")
            # Check if all entries are equal in each column
            if not tendons_min["t1"].max() == tendons_min["t1"].min() or \
                not tendons_min["t1"].max() == tendons_min["t1"].min() or \
                not tendons_min["t1"].max() == tendons_min["t1"].min():
                print("wtf?")

            t1 = tendons_min["t1"].max()
            t2 = tendons_min["t2"].max()
            t3 = tendons_min["t3"].max()
            act_min = np.array([0, 0, 0, t1, t2, t3])
            L = rs.CCRnp.L
            robot_vc = StaticRobotModel(segment_length__m=np.array([L / 2, L / 2]),
                                        youngs_modulus__n_per_m2=rs.CCRnp.E,
                                        pradius_disks__m=np.array(
                                            [rs.CCRnp.r1, rs.CCRnp.r1]),
                                        ro__m=rs.CCRnp.r,
                                        modelling_approach="VC",
                                        f__n=rs.CCRnp.f_ext,
                                        gxf_tol=1e-6,
                                        step_size=1e-2
                                        )
            robot_vc.use_recomputation = True
            success = robot_vc.calc_pose_from_ctrl(act_min)
            if np.alltrue(success):
                frames_min = robot_vc.disk_frames_array
                # Get all disk positions
                disks_n = frames_min.shape[1] / 4
                disks_x = []
                disks_y = []
                disks_z = []
                for n in range(int(disks_n)):
                    disk_idx = 4 * n
                    p_tmp = frames_min[0:3, disk_idx + 3]
                    disks_x.append(p_tmp[0])
                    disks_y.append(p_tmp[1])
                    disks_z.append(p_tmp[2])
                ax4.plot3D(disks_x, disks_y, disks_z, '-', alpha=alpha,
                           color="blue")
                s_torch = torch.linspace(0, float(rs.CCRnp.L), 21).reshape(
                    -1, 1, 1)
                act_torch = torch.from_numpy(act_min[3:].astype(
                    "float32")).reshape(1, 3, 1) * torch.ones_like(
                    s_torch)
                s_act_torch = torch.cat([s_torch, act_torch],
                                        dim=1).reshape(-1, 4)
                pRnm_eval = nn_model(s_act_torch).detach().numpy()
                p = pRnm_eval[:, :3]
                ax4.plot3D(p[:, 0], p[:, 1], p[:, 2], '+', color="blue")
                try:
                    disks_x_ts = [disks_x[-1], disks_x[-1]]
                    disks_y_ts = [disks_y[-1], disks_y[-1]]
                    disks_z_ts = [disks_z[-1], disks_z[0]]

                    ax4.plot3D(disks_x_ts, disks_y_ts, disks_z_ts, "--",
                               alpha=alpha, color="blue")
                    disks_x_ts = [disks_x[-1], rs.CCRnp.L / 1.5]
                    disks_y_ts = [disks_y[-1], disks_y[-1]]
                    disks_z_ts = [disks_z[0], disks_z[0]]

                    ax4.plot3D(disks_x_ts, disks_y_ts, disks_z_ts, "--",
                               alpha=alpha, color="blue")

                    disks_x_ts = [disks_x[-1], disks_x[-1]]
                    disks_y_ts = [disks_y[-1], -rs.CCRnp.L / 1.5]
                    disks_z_ts = [disks_z[0], disks_z[0]]

                    ax4.plot3D(disks_x_ts, disks_y_ts, disks_z_ts, "--",
                               alpha=alpha, color="blue")

                    disks_x_ts = [disks_x[-1], -rs.CCRnp.L / 1.5]
                    disks_y_ts = [disks_y[-1], disks_y[-1]]
                    disks_z_ts = [disks_z[-1], disks_z[-1]]

                    ax4.plot3D(disks_x_ts, disks_y_ts, disks_z_ts, "--",
                               alpha=alpha, color="blue")
                except Exception as e:
                    print("Did not plot surface due to zero actuation shape")

            #---------------------
            # Plot basis actuation

            print("tau max, evaluate_mode: ", tau_max)
            act_b1 = np.array([0, 0, 0, tau_max, 0, 0])
            act_b2 = np.array([0, 0, 0, 0, tau_max, 0])
            act_b3 = np.array([0, 0, 0, 0, 0, tau_max])
            acts = [act_b1, act_b2, act_b3]
            for act in acts:
                # Run reference
                robot_vc = StaticRobotModel(
                    segment_length__m=np.array([L / 2, L / 2]),
                    youngs_modulus__n_per_m2=rs.CCRnp.E,
                    pradius_disks__m=np.array(
                        [rs.CCRnp.r1, rs.CCRnp.r1]),
                    ro__m=rs.CCRnp.r,
                    modelling_approach="VC",
                    f__n=rs.CCRnp.f_ext,
                    gxf_tol=1e-6,
                    step_size=1e-2
                )

                robot_vc.use_recomputation = True
                _ = robot_vc.calc_pose_from_ctrl(act)
                frames_b = robot_vc.disk_frames_array
                # Get all disk positions
                disks_n = frames_b.shape[1] / 4
                disks_x = []
                disks_y = []
                disks_z = []

                for n in range(int(disks_n)):
                    disk_idx = 4 * n
                    p_tmp = frames_b[0:3, disk_idx + 3]
                    disks_x.append(p_tmp[0])
                    disks_y.append(p_tmp[1])
                    disks_z.append(p_tmp[2])

                # Plot reference
                ax4.plot3D(disks_x, disks_y, disks_z, '-', alpha=.8,
                           color="grey")

                # Construct actuation vecotors for nn
                s_torch = torch.linspace(0, float(rs.CCRnp.L), 21).reshape(
                    -1, 1, 1)
                act_torch = torch.from_numpy(act[3:].astype(
                    "float32")).reshape(1, 3, 1) * torch.ones_like(
                    s_torch)
                s_act_torch = torch.cat([s_torch, act_torch],
                                        dim=1).reshape(-1, 4)
                # Run nn
                pRnm_eval = nn_model(s_act_torch).detach().numpy()
                p = pRnm_eval[:, :3]
                # Plot nn
                ax4.plot3D(p[:, 0], p[:, 1], p[:, 2], '+', color="black",
                           alpha=alpha)

                # Plot lines for a better 3D view
                # EE projection in xy plane
                disks_x_ts = [disks_x[-1], disks_x[-1]]
                disks_y_ts = [disks_y[-1], disks_y[-1]]
                disks_z_ts = [disks_z[-1], disks_z[0]]

                ax4.plot3D(disks_x_ts, disks_y_ts, disks_z_ts, "--",
                           alpha=.3, color="grey")

                # Project in xy plane on y axis
                disks_x_ts = [disks_x[-1], rs.CCRnp.L / 1.5]
                disks_y_ts = [disks_y[-1], disks_y[-1]]
                disks_z_ts = [disks_z[0], disks_z[0]]

                ax4.plot3D(disks_x_ts, disks_y_ts, disks_z_ts, "--",
                           alpha=.3, color="grey")

                # Project in xy plane on x axis
                disks_x_ts = [disks_x[-1], disks_x[-1]]
                disks_y_ts = [disks_y[-1], -rs.CCRnp.L / 1.5]
                disks_z_ts = [disks_z[0], disks_z[0]]

                ax4.plot3D(disks_x_ts, disks_y_ts, disks_z_ts, "--",
                           alpha=.3, color="grey")

                # Project ee on xz plane
                disks_x_ts = [disks_x[-1], disks_x[-1]]
                disks_y_ts = [disks_y[-1], rs.CCRnp.L / 1.5]
                disks_z_ts = [disks_z[-1], disks_z[-1]]

                ax4.plot3D(disks_x_ts, disks_y_ts, disks_z_ts, "--",
                           alpha=.3, color="grey")

            ax4.set_xlim([-rs.CCRnp.L / 1.5, rs.CCRnp.L / 1.5])
            ax4.set_ylim([-rs.CCRnp.L / 1.5, rs.CCRnp.L / 1.5])
            ax4.set_zlim([0, rs.CCRnp.L * 1.1])
            ax4.set_xlabel('x (m)')
            ax4.set_ylabel('y (m)')
            ax4.set_zlabel('z (m)')

            ax4.legend()
            figs.append(fig3)
            fig_names.append("shapes")

        if collocation_points:
            # Load relevant data to generate collocation points
            dl = DataLoader(ds_collocation, batch_size=len(ds_collocation))
            # Make it an iterator
            dl_iter = iter(dl)
            # Get collocation points
            s_act = next(dl_iter)
            assert s_act.shape[0] == len(ds_collocation)

            s_act_df = pd.DataFrame(s_act.detach().numpy(), columns=["s", "t1",
                                                                   "t2", "t3"])
            if not gaussian_collocation:
                res = s_act_df.groupby([f't{tendons[0]}', f't{tendons[1]}']).size(
                                                        ).sort_values(ascending=False)
                res_reset = res.reset_index()
                point_size = 5
            else:
                # Split data into bins...
                res = s_act_df.groupby(
                    [f't{tendons[0]}', f't{tendons[1]}']).size(
                ).sort_values(ascending=False)
                res_reset = res.reset_index()
                point_size = .1

            fig_cp, a_cp = plt.subplots(1, 1, sharey="all",
                                              tight_layout=True,
                                        )
            a_cp_ = a_cp.scatter(res_reset[f"t{tendons[0]}"],
                                 res_reset[f"t{tendons[1]}"],
                              c=res_reset[0], s=point_size)
            #ax_hdl.append(a)
            a_cp.set_aspect("equal")
            fig_cp.colorbar(a_cp_, ax=a_cp, location="right", shrink=0.82)
            a_cp.set_title("Collocation points")
            a_cp.set_xlim([-0.1, tau_max * 1.1])
            a_cp.set_ylim([-0.1, tau_max * 1.1])
            a_cp.set_xlabel(rf"Tendon $t_{tendons[0]}$")
            a_cp.set_ylabel(rf"Tendon $t_{tendons[1]}$")
            figs.append(fig_cp)
            fig_names.append("Collocation Points")

    if close:
        for fig in figs:
            plt.close(fig)

    return figs, fig_names, res_df

