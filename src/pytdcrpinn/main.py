import robot_specs as rs
import os
import datetime
from copy import deepcopy
from run_training import dict_unpack_mp

if __name__ == "__main__":


    nn_opt_name = ("run_training_prec_ref__epoch_4610_06_31_08_15.536nn_opt"
                   ".pth")
    # Define parameters for this very run
    training_config = [
                    [("hidden_layer_n",3), ("layer_width",  100),
                     ("phi_range", (300, 360)),
                     ("learning_rate", 1.5 * 1e-4),
                     ("max_epochs", 1500),
                     ("name_nn", "prec_ref"),
                    # ("nn_opt_name", nn_opt_name), Enter state dictionary
                     # name like above (nn_opt_name) if you watn to resume
                     # training e.g. from epoch 4610 with the optimizer
                     # state and scheduler at that point.
                     #("epoch_resume_training", 4610)
                     ],
                       ]
    phi_range = (300, 360)
    #### Problem set-up
    # Architecture
    hidden_layer_n = 3
    layer_width = 200

    # Approximation range
    tau_max = 3.51   # Maximum actuation (virtual) actuation force. Force that
                     # act on a disk at position phi on attachment radius

    # --------------------------------------------------------------
    # PINN-training specification - Boundary- and collocation value
    # --------------------------------------------------------------
    # Number of boundary values for s = L and initial values s = 0
    boundary_s_n = 3 * 2**5
    # Number of boundary values for tau_max
    boundary_tau_max_n = 3 * 2**5

    # Number of collocation points
    #
    collocation_tau_phi = 2 ** 5
    collocation_tau = 2 ** 5
    collocation_s = 2 ** 5
    processes_gen = 10

    # NN Training specification
    batch_size_collocation = int(collocation_s * collocation_tau * \
                             collocation_tau_phi / 2**5)
    # epochs
    batch_size_iv_s0 = int(boundary_s_n/ (3 * 2**3))
    batch_size_bv_sL = int(boundary_s_n/ (3 * 2**3))
    batch_size_bv_sL_data = int(19 * boundary_s_n /(3 * 2**3))
    batch_size_bv_tau_max = int(21 * boundary_tau_max_n / (3 * 2**2))
    learning_rate = 0.0001
    max_epochs = 3_000
    retain_graph = False
    percentage_training_data = 0.7 * 100    # Amount of data to use for
                                            # training.# Rest
                                            # is split 50/50 into test and
                                            # validation

    # Sample with respect to the worst actuations during validation after
    # each epoch
    importance_sampling = True
    tau_centers_n = 10
    s_centers_n = 10
    s_scale = 0.03 * rs.CCRnp.L
    tau_scale = 0.03 * tau_max


    # Save and Load, multiprocessing
    processes = len(training_config) # Processes to use for reference data calculation

    # Naming
    # Name of reference data
    ref_name = (f"bsn{boundary_s_n}_btm{boundary_tau_max_n}_" +
                f"phirange_{phi_range[0]}_"
                f"{phi_range[1]}_"
                f"taum{tau_max}_")

    # Check if reference exists
    load_reference = os.path.isfile("./reference/" + ref_name + "_s0.pt")
    save_reference = not load_reference  # Save reference data if True
    generate_data = save_reference
    now = datetime.datetime.now()
    current_time = now.strftime("%H_%M_%S")
    name_nn = "_" + current_time + "_"

    # Pack everything into one configuration dictionary
    config_dict = {"hidden_layer_n": hidden_layer_n,
                    "layer_width": layer_width,
                    "boundary_s_n": boundary_s_n,
                    "boundary_tau_max_n": boundary_tau_max_n,
                    "collocation_tau_phi": collocation_tau_phi,
                    "collocation_tau": collocation_tau,
                    "collocation_s": collocation_s,
                    "tau_max": tau_max,
                    "processes": processes_gen,
                    "percentage_training_data": percentage_training_data,
                    "batch_size_collocation": batch_size_collocation,
                    "batch_size_iv_s0": batch_size_iv_s0,
                    "batch_size_bv_sL": batch_size_bv_sL,
                    "batch_size_bv_sL_data": batch_size_bv_sL_data,
                    "batch_size_bv_tau_max": batch_size_bv_tau_max,
                    "learning_rate": learning_rate,
                    "max_epochs": max_epochs,
                    "save_reference": save_reference,
                    "load_reference": load_reference,
                    "ref_name": ref_name,
                    "name_nn": name_nn,
                    "phi_range": (300, 360),
                     "importance_sampling": importance_sampling,
                     "tau_centers_n": tau_centers_n,
                     "s_centers_n": s_centers_n,
                     "s_scale": s_scale,
                     "tau_scale": tau_scale,
                   }

    config_dict_tmp = deepcopy(config_dict)
    config_dict_tmp["processes"] = 10

    for e in training_config[0]:
        config_dict_tmp[e[0]] = e[1]
        if "phi_range" == e[0]:
            pr1 = e[1][0]
            pr2 = e[1][1]
            ref_name = (f"bsn{boundary_s_n}_btm{boundary_tau_max_n}_" + \
                        f"phirange_{pr1}_{pr2}_taum{tau_max}_")
            config_dict_tmp[e[0]] = e[1]
            load_reference = os.path.isfile(
                "./reference/" + ref_name + "_s0.pt")  # Load
            # reference data from disk
            save_reference = not load_reference  # Save reference data if True
            generate_data = save_reference
            config_dict_tmp["save_reference"] = save_reference
            config_dict_tmp["load_reference"] = load_reference
            config_dict_tmp["ref_name"] = ref_name

    dict_unpack_mp(config_dict=config_dict_tmp)
