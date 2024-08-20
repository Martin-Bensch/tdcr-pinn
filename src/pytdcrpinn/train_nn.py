import gc
import sys
import typing

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from pytdcrsv.static_robot_model import StaticRobotModel
from _losses import compute_loss, compute_p_loss
import pinn_nn as pnn
import robot_specs as rs
import datasl as dsl
from robot_specs import CCRnp
import evaluate_model as em
import time
from datetime import timedelta
from numpy.random import binomial
from _helper_fn import add_individual_losses_in_epoch
import datetime


def train_model(
        hidden_layer_n: int = 3,
        layer_width: int = 250,
        boundary_s_n: int = 100,
        boundary_tau_max_n: int =100,
        collocation_tau_phi: int = 72,
        collocation_tau: int = 30,
        collocation_s: int = 100,
        tau_max: float = 0.7,
        processes: int = 4,
        percentage_training_data: int = 70,
        batch_size_collocation: int = 128,
        batch_size_iv_s0: int = 512,
        batch_size_bv_sL: int = 512,
        batch_size_bv_sL_data: int = 512,
        batch_size_bv_tau_max: int = 512,
        batch_size_bv_tau_min: int = 128,
        learning_rate: float = 0.01,
        max_epochs: int = 1_000,
        save_reference: bool = True,
        load_reference: bool = True,
        ref_name: str = None,
        phi_range: typing.Tuple = (300, 360),
        name: str = "unknown",
        importance_sampling: bool= True,
        tau_centers_n: int = 10,
        s_centers_n: int = 10,
        s_scale: float = 0.01 * rs.CCRnp.L,
        tau_scale: float = None,
        epoch_resume_training: int = -1,
        nn_opt_name: str = None
) -> (pnn.NNApproximator):
    """

    Args:
        hidden_layer_n: Number of hidden layers
        layer_width: Width of all hidden layers
        boundary_s_n: Number of positions along the backbone for the
                      boundary values
        boundary_tau_max_n: Number of tau boundary values
        collocation_tau_phi: Number of collocation points of bending plane
                             angle phi
        collocation_tau: Number of collocation points of tau
        collocation_s: Number of collocation points s
        tau_max: Maximum magnitude of tau_max, for one tendon actuation
        processes: Number of processes for computing reference
        percentage_training_data: Ratio of data sets (training, validation,
                                  testing
        batch_size_collocation: Batch size collocation points
        batch_size_iv_s0: Batch size initial values of s (reference model)
        batch_size_bv_sL: Batch size boundary values of s i.e. s = L
                          (reference model)
        batch_size_bv_sL_data: Batch size data points between s=0 and s=L
                               (reference model)
        batch_size_bv_tau_max: Batch size for tau = tau_max
        batch_size_bv_tau_min: Batch size for tau = tau_min
        learning_rate: Learning rate for training
        max_epochs: maximum training epochs
        save_reference: True for saving the computed reference
        load_reference: True for loading a previously computed reference
        ref_name: filename of the refenrece
        phi_range: range of the actuation force application (see paper)
        name: name of the approximater. Used for storing its configuration
              dictionary
        importance_sampling: True, with a probability of p=0.1 sample
                             collocation w.r.t. a gaussian distribution at
                             the regions (tau, s) that perform worst
        tau_centers_n: How many centers for tau to sample around during
                       importance_sampling=True
        s_centers_n: How many centers for s to sample around during
                     importance_sampling=True
        s_scale: spread of the gaussion distribution of s during
                importance_sampling
        tau_scale: spread of the gaussion distribution of tau during
                   importance_sampling
        epoch_resume_training: At which epoch the training resumed
        nn_opt_name: Name of the config file holding nn and optimizer
                     state

    Returns:
        Love.
    """
    # Build NN
    nn = pnn.NNApproximator(hidden_layer_n, layer_width,
                                         max_tau=tau_max,
                                         ).to(pnn.DEVICE)

    # Flag for deciding if one wants to resume training at
    # epoch epoch_resume_training
    resume_training = False if epoch_resume_training == -1 else True

    # Setup tensorboard writer
    path = f"./runs/" + name
    tb_writer = SummaryWriter(path)
    print(f"tb_writer output at: " + path)

    # Set seed for weights
    torch.manual_seed(123)

    # Configure optimizer
    optimizer_adam = torch.optim.AdamW(nn.parameters(), lr=learning_rate)

    # Load relevant data to resume training
    if resume_training:
        try:
            state_dict = torch.load("./state_dicts/" + nn_opt_name)
            nn.load_state_dict(state_dict["model_state_dict"])
            optimizer_adam.load_state_dict(state_dict["optimizer_state_dict"])
            epoch_resume_training = state_dict["epoch"]
            nn.train()
        except Exception as e:
            raise Exception(e)

    # Generate training data (run reference model)
    rr = dsl.RunReference(boundary_s_n=boundary_s_n,
                         boundary_tau_max_n=boundary_tau_max_n,
                         tau_max=tau_max,
                         processes=processes,
                         ref_name=ref_name,
                         phi_range=phi_range
                         )

    if load_reference:
        rr.load_from_disk(name=ref_name, name_config=name+"_config.json")
        nn.max_tau = rr.tau_max_generated
    else:
        rr.generate_reference()
        if save_reference:
            rr.save_to_disk(ref_name)


    # Build data sets (data from the reference model)
    #
    # Initial values s = 0
    ds_init_s0 = dsl.InitialValuesS0(rr)
    # Boundary values s = L
    ds_bv_sL = dsl.BoundaryValuesSL(rr)
    # Data points between s = 0 and s = L
    ds_bv_data = dsl.BoundaryValuesDatasl(rr)
    # Boundary values for tau = tau_max
    ds_bv_tau_max = dsl.BoundaryValuesTauMax(rr)
    # Boundary values for tau = tau_min
    ds_bv_tau_min = dsl.BoundaryValuesTauMin(rr)

    # Initial collocation points
    ds_collocation = dsl.CollocationPoints(rr,
                                      collocation_tau_phi=collocation_tau_phi,
                                      collocation_tau=collocation_tau,
                                      collocation_s=collocation_s)
    if not resume_training:
        # Split data sets
        train_smpl_i0 = int(round(len(ds_init_s0) *
                                  percentage_training_data / 100
                                  )
                            )

        val_smpl_i0 = int(0.5 * (len(ds_init_s0) - train_smpl_i0))
        test_smpl_i0 = len(ds_init_s0) - train_smpl_i0 - val_smpl_i0
        train_ds_i0, val_ds_i0, test_ds_i0 = random_split(ds_init_s0,
                                                             (train_smpl_i0,
                                                              val_smpl_i0,
                                                              test_smpl_i0)
                                                          )
        train_smpl_bv_sL = int(round(len(ds_bv_sL) *
                                     percentage_training_data / 100
                                     )
                               )

        val_smpl_bv_sL = int(0.5 * (len(ds_bv_sL) - train_smpl_bv_sL))
        test_smpl_bv_sL = len(ds_bv_sL) - train_smpl_bv_sL - val_smpl_bv_sL
        train_ds_bv_sL, val_ds_bv_sL, test_ds_bv_sL = random_split(ds_bv_sL,
                                                          (train_smpl_bv_sL,
                                                           val_smpl_bv_sL,
                                                           test_smpl_bv_sL)
                                                          )
        train_smpl_bv_data = int(round(len(ds_bv_data) *
                                       percentage_training_data / 100
                                       )
                                 )
        val_smpl_bv_data = int(0.5 * (len(ds_bv_data) - train_smpl_bv_data))
        test_smpl_bv_data = (len(ds_bv_data) - train_smpl_bv_data -
                             val_smpl_bv_data)
        (train_ds_bv_data,
         val_ds_bv_data,
         test_ds_bv_data) = random_split(ds_bv_data,
                                          (train_smpl_bv_data,
                                           val_smpl_bv_data,
                                           test_smpl_bv_data)
                                          )

        train_smpl_bv_tau_max = int(round(len(ds_bv_tau_max) *
                                    percentage_training_data / 100))

        val_smpl_bv_tau_max = int(0.5 * (
                                            len(ds_bv_tau_max) -
                                            train_smpl_bv_tau_max
                                        )
                                  )

        test_smpl_bv_tau_max = (len(ds_bv_tau_max) - train_smpl_bv_tau_max -
                                val_smpl_bv_tau_max)
        (train_ds_bv_tau_max,
         val_ds_bv_tau_max,
         test_ds_bv_tau_max) = random_split(
                                            ds_bv_tau_max,
                                          (train_smpl_bv_tau_max,
                                           val_smpl_bv_tau_max,
                                           test_smpl_bv_tau_max)
                                          )

        train_smpl_bv_tau_min = int(round(len(ds_bv_tau_min) / 3))

        val_smpl_bv_tau_min = int(0.5 * (len(ds_bv_tau_min) -
                                         train_smpl_bv_tau_min))

        test_smpl_bv_tau_min = (len(ds_bv_tau_min) - train_smpl_bv_tau_min -
                                val_smpl_bv_tau_min)
        (train_ds_bv_tau_min,
         val_ds_bv_tau_min,
         test_ds_bv_tau_min) = random_split(ds_bv_tau_min,
                                            (train_smpl_bv_tau_min,
                                            val_smpl_bv_tau_min,
                                            test_smpl_bv_tau_min)
                                            )
        # Store training, testing, validation data indices as json file.
        # And add it to RunReference object
        test_val_train_data = rr.store_training_validation_testing_indices(
                ref_name,
                train_ds_i0, val_ds_i0, test_ds_i0,
                train_ds_bv_sL, val_ds_bv_sL, test_ds_bv_sL,
                train_ds_bv_data, val_ds_bv_data, test_ds_bv_data,
                train_ds_bv_tau_max, val_ds_bv_tau_max, test_ds_bv_tau_max,
                train_ds_bv_tau_min, val_ds_bv_tau_min,
                test_ds_bv_tau_min
                                                                              )
        rr.test_val_train_data = test_val_train_data
    else:
        # Load datasets from last training
        train_ds_i0 = rr.test_val_train["training"]["i0"]
        train_ds_bv_sL = rr.test_val_train["training"]["sL"]
        train_ds_bv_data = rr.test_val_train["training"]["dataSL"]
        train_ds_bv_tau_max = rr.test_val_train["training"]["tau_max"]
        train_ds_bv_tau_min = rr.test_val_train["training"]["tau_min"]

        with torch.no_grad():
            # Track shape in tensorboard
            figure_shape = plot_shape(nn, tau_max=tau_max)
            tb_writer.add_figure(f"Shape", figure_shape,
                                 global_step=epoch_resume_training-1,
                                 close=True
                                 )
    if save_reference:
        return 0

    # Instantiate Dataloaders
    train_dl_i0 = DataLoader(train_ds_i0, batch_size=batch_size_iv_s0,
                             shuffle=True)
    train_dl_bv_sL = DataLoader(train_ds_bv_sL, batch_size=batch_size_bv_sL,
                                shuffle=True)
    train_dl_bv_data = DataLoader(train_ds_bv_data,
                                  batch_size=batch_size_bv_sL_data,
                                  shuffle=True)
    train_dl_bv_tau_max = DataLoader(train_ds_bv_tau_max,
                                     batch_size=batch_size_bv_tau_max,
                                     shuffle=True)
    train_dl_bv_tau_min = DataLoader(train_ds_bv_tau_min,
                                     shuffle=True,
                                     batch_size=batch_size_bv_tau_min,)

    train_dl_i0_iter = iter(train_dl_i0)
    train_dl_bv_sL_iter = iter(train_dl_bv_sL)
    train_dl_bv_data_iter = iter(train_dl_bv_data)
    train_dl_bv_tau_max_iter = iter(train_dl_bv_tau_max)
    train_dl_bv_tau_min_iter = iter(train_dl_bv_tau_min)

    train_dl_collocation = DataLoader(ds_collocation,
                                    batch_size=batch_size_collocation,
                                     shuffle=True)

    N_iterations_per_epoch = len(train_dl_collocation)

    # Define scheduler
    if not resume_training:
        lr_scheduler_1cyc = torch.optim.lr_scheduler.OneCycleLR(optimizer_adam,
                                       max_lr=learning_rate,
                                       pct_start=0.1,
                                       epochs=max_epochs,
                                       steps_per_epoch=N_iterations_per_epoch,
                                       div_factor=10,
                                       final_div_factor=100,
                                       cycle_momentum=True,
                                       verbose=False,
                                                        )
    patience = 20
    lr_scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                        optimizer_adam, 'min',
                                                        patience=patience,
                                                        factor=0.9,
                                                        cooldown=10
                                                        )
    plateau_metric = []
    scheduler_milestone = [int(0.3 * max_epochs)]
    training_iteration = 0
    print("+ ------------------------------- +")
    print("Start training with: \n" +
        f"\t - {len(ds_collocation)} Collocation points \n" +
        f"\t - {len(train_dl_collocation)} batches collocation "
        f"points \n" +
        f"\t - {len(train_ds_i0)} Initial values\n" +
        f"\t - {len(train_dl_i0)} batches initial values\n" +
        f"\t - {len(train_ds_bv_sL)} Boundary values s=L\n" +
        f"\t - {len(train_dl_bv_sL)} batches boundary values s=L\n" +
        f"\t - {len(train_ds_bv_data)} Boundary values sL data \n" +
        f"\t - {len(train_dl_bv_data)} batches boundary values sL data "
        f"\n" +
        f"\t - {len(train_ds_bv_tau_max)} Boundary values tau_max \n" +
        f"\t - {len(train_dl_bv_tau_max)} batches boundary values tau_max \n"
          )

    ###########################################################################
    # Training loop
    res_df_validation = None
    lr_on_plateau = False
    regenerate = True
    resample = True
    for epoch in range(max_epochs):
        if resume_training:
            epoch = epoch + epoch_resume_training
        # Use binomial distribution to decide whether to sample collocation
        # points at bad performing regions in the input space or just use a
        # grid
        grid_or_importancesmp = binomial(1, 0.1)

        if (resample and importance_sampling and res_df_validation
                is not None and grid_or_importancesmp == 1):
            ds_collocation.generate_collocation_points_importance_sampling(
                                                res_df=res_df_validation,
                                                tau_centers_n=tau_centers_n,
                                                s_centers_n=s_centers_n,
                                                s_scale=s_scale,
                                                tau_scale=tau_scale
                                                                          )
            resample = False
            regenerate = True
        elif regenerate:
            ds_collocation.generate_collocation_points_grid()
            resample = True
            regenerate = False

        # Track epoch's loss
        losses_epoch = []
        individual_losses_epoch = {}

        start = time.time()
        for s_tau in train_dl_collocation:
            # Due to differently sized batch sizes one needs these try
            # except blocks
            try:
                i0 = next(train_dl_i0_iter)
            except StopIteration:
                train_dl_i0_iter = iter(train_dl_i0)
                i0 = next(train_dl_i0_iter)
            try:
                bv_sL = next(train_dl_bv_sL_iter)
            except StopIteration:
                train_dl_bv_sL_iter = iter(train_dl_bv_sL)
                bv_sL = next(train_dl_bv_sL_iter)
            try:
                bv_data = next(train_dl_bv_data_iter)
            except StopIteration:
                train_dl_bv_data_iter = iter(train_dl_bv_data)
                bv_data = next(train_dl_bv_data_iter)
            try:
                bv_tau_max = next(train_dl_bv_tau_max_iter)
            except StopIteration:
                train_dl_bv_tau_max_iter = iter(train_dl_bv_tau_max)
                bv_tau_max = next(train_dl_bv_tau_max_iter)
            try:
                bv_tau_min = next(train_dl_bv_tau_min_iter)
            except StopIteration:
                train_dl_bv_tau_min_iter = iter(train_dl_bv_tau_min)
                bv_tau_min = next(train_dl_bv_tau_min_iter)

            try:
                losses_dict = compute_loss(nn,
                                          s_tau,
                                          i0=i0,
                                          bv_sL=bv_sL,
                                          bv_data=bv_data,
                                          bv_tau_max=bv_tau_max,
                                          bv_tau_min=bv_tau_min,
                                          )

                # Build loss
                loss = torch.tensor([0.])
                for l in losses_dict:
                    if losses_dict[l] == 0.0:
                        continue
                    loss = loss + torch.tensor([100.]) * losses_dict[l]

                # Compute loss for this batch
                loss_batch = sum([float(losses_dict[l]) for l in losses_dict])

                # Track all losses for this epoch
                losses_epoch.append(loss_batch)
                individual_losses_epoch = add_individual_losses_in_epoch(
                                                    losses_dict=losses_dict,
                        individual_losses_per_epoch=individual_losses_epoch)

                # Free some memory
                del s_tau, i0, bv_sL, bv_data, bv_tau_max, bv_tau_min, \
                    losses_dict
                gc.collect()

                # Zero out the gradients
                optimizer_adam.zero_grad(set_to_none=True)

                # Optimize
                with torch.no_grad():
                    loss.backward(retain_graph=False)
                    optimizer_adam.step()

                del loss
                # ######################## END OPTIMIsation ##################

                #  Increase learning rate in the beginning
                if epoch < scheduler_milestone[0] and not resume_training:
                    lr_scheduler_1cyc.step()
                else:
                    lr_on_plateau = True

                training_iteration += 1

            except KeyboardInterrupt:
                break

        end = time.time()
        # Monitoring the loss
        try:
            loss_epoch = sum(losses_epoch) / len(losses_epoch)
        except ZeroDivisionError:
            loss_epoch = sum(losses_epoch)

        # Metric for using the reduce learning rate on plateau scheduler
        if len(plateau_metric) < int(patience / 3) :
            plateau_metric.append(loss_epoch)
        else:
            _ = plateau_metric.pop(0)
            plateau_metric.append(loss_epoch)

        # Printing current epoch, avg. loss and elapsed time
        if epoch % 1 == 0:
            print(f"{name}: "
                  f"\n \t epoch: {epoch} "
                  f"\n \t Average loss of epoch: {loss_epoch}, "
                  f"\n \t elapsed time: {timedelta(seconds=end - start)}")

        # Save the PINNs state (weights etc.) and if one decides to resume
        # the training, also save epoch and optimizer state.
        if epoch > 99 and (epoch % 100 == 0 or epoch % 10 == 0):
            try:
                now = datetime.datetime.now()
                current_time = now.strftime("%H_%M_%S")
                torch.save(nn.state_dict(),
                           "./state_dicts/" + name + "_"
                           + "epoch_" + str(epoch) + current_time + "_"
                            + ".pth")
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': nn.state_dict(),
                            'optimizer_state_dict': optimizer_adam.state_dict()
                            },
                            "./state_dicts/" + name + "_" +
                             "_epoch_" +
                            str(epoch) + "_" + current_time + "_"
                            +str(round(loss_epoch,3)) + "nn_opt.pth")
                last_loss = loss_epoch
            except:
                print("Could not save estimator")

        # Use reduce on plateau scheduler
        if lr_on_plateau:
            lr_scheduler_plateau.step(np.mean(plateau_metric))

        if tb_writer is not None and not save_reference:
            lr = float(optimizer_adam.param_groups[0]['lr'])
            # Compute average of individual losses
            # One epoch adds len(train_dl_collocation) times each individual
            # loss
            individual_losses_epoch_avg = {}
            for k in individual_losses_epoch:
                individual_losses_epoch_avg[k] = individual_losses_epoch[k] \
                                                 / N_iterations_per_epoch

            plot_scalars(tb_writer, epoch, lr=lr, loss_tmp=loss_epoch,
                                     losses_dict=individual_losses_epoch_avg)

        # Stop batch normalization
        nn.eval()

        # -----------------------
        # Tensorboard monitoring
        # -----------------------
        with torch.no_grad():
            # Track shape in tensorboard
            figure_shape = plot_shape(nn, tau_max=tau_max,)
            tb_writer.add_figure(f"Shap VC", figure_shape,
                                 global_step=epoch,
                                 close=True,
                                 )
            del figure_shape
            ref_folder = "./reference/"
            name_config = name + "_config.json"

            # Evaluate on validation data
            figs, fig_names, res_df_validation = em.evaluate_euclid(
                                                 nn_model=nn,
                                                 data_type="validation",
                                                 folder=ref_folder,
                                                 filename=ref_name,
                                                 name_config=name_config,
                                                 phi_range=phi_range,
                                                 ds_collocation=None,
                                                collocation_points=False,
            )

            for f, fn in zip(figs, fig_names):
                tb_writer.add_figure(fn + " validation", f,
                                     global_step=epoch,
                                     close=True,
                                     )

            gc.collect()
            # Evaluate on training data
            figs, fig_names, res_df_training = em.evaluate_euclid(nn_model=nn,
                                                 data_type="training",
                                                 folder=ref_folder,
                                                 filename=ref_name,
                                                 name_config=name_config,
                                                 collocation_points=True,
                                                 phi_range=phi_range,
                                                 scatter_plot = False,
                                                 heatmap = False,
                                                 shape = False,
                                                 ds_collocation=ds_collocation,
                                                )
            for f, fn in zip(figs, fig_names):
                tb_writer.add_figure(fn + " training", f,
                                     global_step=epoch,
                                     close=True,
                                     )

            plt.close("all")

        # Switch back to training
        nn.train()

    # Return the ANN
    return nn


def plot_scalars(tb_writer, epoch, loss_tmp, lr, losses_dict):
    for k in losses_dict:
        tb_writer.add_scalar(k, losses_dict[k], epoch)

    tb_writer.add_scalar("00_loss", loss_tmp, epoch)
    tb_writer.add_scalar("01_lr", lr, epoch)

def plot_shape(nn, model="VC", tau_max=2.5):
    nn.eval()
    with torch.no_grad():
        figure = plt.figure()
        ax = plt.axes(projection='3d')
        act1 = np.array([0, 0, 0, 1, 0, 0])
        act2 = np.array([0, 0, 0, 0, 1, 0])
        act3 = np.array([0, 0, 0, 0, 0, 1])
        act7 = np.array([0, 0, 0, 0, 0, 0])
        act = [act1, act2, act3, act7]
        storage = []
        for idx, m in enumerate(act):
            m_ = float(tau_max) * m
            if model == "VC_Ref":
                disks_n = 3
            else:
                disks_n = 10
            robot_vc = StaticRobotModel(
                        segment_length__m=np.array(
                                                    [float(CCRnp.L) / 2,
                                                     float(CCRnp.L) / 2]),
                                        youngs_modulus__n_per_m2=rs.CCRnp.E,
                                        pradius_disks__m=np.array([rs.CCRnp.r1,
                                                                   rs.CCRnp.r1]
                                                                  ),
                                        ro__m=rs.CCRnp.r,
                                        modelling_approach=model,
                                        f__n=rs.CCRnp.f_ext,
                                        gxf_tol=1e-6,
                                        step_size=1e-3,
                                        number_disks=np.array([disks_n,
                                                               disks_n])
                                        )
            success = robot_vc.calc_pose_from_ctrl(act=m_)
            ee_frames_raw = robot_vc.disk_frames_array
            # Get positions
            px = []
            py = []
            pz = []
            for n in range(2 * disks_n + 1):
                disk_idx = n * 4
                T = ee_frames_raw[:, disk_idx:disk_idx + 4]
                assert T.shape[0] == 4
                px.append(T[0, 3])
                py.append(T[1, 3])
                pz.append(T[2, 3])

            s_torch = torch.linspace(0, float(CCRnp.L), 20).reshape(-1,
                                                                    1, 1)
            act_torch = torch.from_numpy(m_[3:].astype(
                "float32")).reshape(1, 3, 1) * torch.ones_like(
                s_torch)
            s_mL_torch = torch.cat([s_torch, act_torch],
                                   dim=1).reshape(-1, 4)
            pRnm_eval = pnn.f(nn, s_mL_torch).detach().numpy()
            p = pRnm_eval[:, :3]
            ax.plot3D(p[:, 0], p[:, 1], p[:, 2], '-+', label=fr"t_{idx + 1}")
            ax.plot3D(px, py, pz, label=f"{idx}")
            storage.append(pz[-1])
            if idx == 3:
                break

        ax.set_xlim([-rs.CCRnp.L / 2, rs.CCRnp.L / 2])
        ax.set_ylim([-rs.CCRnp.L / 2, rs.CCRnp.L / 2])
        ax.set_zlim([0, rs.CCRnp.L * 1.1])
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        plt.close(figure)
    nn.train()
    return figure