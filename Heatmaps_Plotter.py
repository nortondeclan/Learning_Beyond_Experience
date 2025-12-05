import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

fig_4 = False #True
fig_S4 = True

if fig_4:
    run_labels = [
        "duff_v0_vc_ic_s200_r1e-12_n1e-5",
        "duff_v0_nogen_vc_ic_s200_r1e-12_n1e-5",
        "duff_v0_V10_nl_ic_s200_r1e-12_n1e-5",
        "duff_v0_nogen_V10_nl_ic_s200_r1e-12_n1e-5",
        ]

elif fig_S4:
    run_labels = [
        "mag_v01_vc_ic_s2500_r1e-10_n1e-3",
        "mag_v01_nogen_vc_ic_s2500_r1e-10_n1e-3",
        "mag_v01_V1.5_nl_ic_s2500_r1e-10_n1e-3",
        "mag_v01_nogen_V1.5_nl_ic_s2500_r1e-10_n1e-3",
        ]

plot_diags = [
    True,
    True,
    False,
    False
    ]
run_label = run_labels[0]

rotation = 0
font_size = 20.
fp_vline = True
fp_hline = False
extra_vertical = None
equal_aspect = False

if "duff" in run_label:
    test_system = "duff"
elif "mag" in run_label:
    test_system = "mag"

if test_system == "duff":
    utility = 0.5
    vmax = 1
elif test_system == "mag":
    utility = 1./3.
    vmax = 1

def join(directory, name):
    return os.path.join(directory, name)

read_loc = join(os.getcwd(), "Averaged_Data")

folder_letters = {
    "tl" : "Test Length",
    "ic" : "Training IC Half-range, $\Delta_0^{train}$",
    "nl" : "Number of Training ICs, $N_{train}$",
    "dt" : "Convergence Distance",
    "s" : "RC Size",
    "r" : "Regularization",
    "n" : "Noise Amplitude",
    "vc" : "Test IC Half-range, $\Delta_0^{test}$",
    "sr" : "Spectral Radius, $\\rho$",
    "is" : "Input Strength Range, $\\sigma$",
    "ph" : "Pendulum Height, $d$",
    "it" : "Discarded Transient, $t_{trans}^{it}$",
    "d" : "Reservoir Density",
    "pn" : "Process/Dynamical Noise, $\\eta_p$"
    }

folder_scales = {
    "tl" : {"value" : "linear"},
    "ic" : {"value" : "linear"},
    "nl" : {"value" : "log"},
    "dt" : {"value" : "linear"},
    "s" : {"value" : "linear"},
    "r" : {"value" : "log"},
    "n" : {"value" : "log"},
    "vc" : {"value" : "linear"},
    "sr" : {"value" : "linear"},
    "is" : {"value" : "linear"},
    "ph" : {"value" : "linear"},
    "it" : {"value" : "linear"},
    "d" : {"value" : "log"},
    "pn" : {"value" : "log"}
    }

if test_system == "duff":
    a, b, c = -0.5, -1., 0.1

    fixed_pts = np.array([[-np.sqrt(-b/c), 0], [np.sqrt(-b/c), 0]])
       
elif test_system == "mag":
    
    mag_locs = 1./np.sqrt(3.) * np.array([
        [np.cos(rotation), np.sin(rotation)],
        [np.cos(rotation + 2*np.pi/3.), np.sin(rotation + 2*np.pi/3.)],
        [np.cos(rotation + 4*np.pi/3.), np.sin(rotation + 4*np.pi/3.)]
        ])
    fixed_pts = mag_locs

with mpl.rc_context({'font.size': 15}):
    comp_fig, comp_ax = plt.subplots(
        2, 2, figsize = (8, 8),
        constrained_layout = True, sharex = True, sharey = "row"
        )
    
    for ri, run_label in enumerate(run_labels):
        
        folder_1 = join(read_loc, run_label)
        f1s = os.listdir(folder_1)
        if 'accuracies.pickle' in f1s:
            f1s.remove('accuracies.pickle')
        if 'metrics.pickle' in f1s:
            f1s.remove('metrics.pickle')
            
        with open(join(folder_1, 'letters_1.pickle'), 'rb') as tmp_file:
            letters_1 = pickle.load(tmp_file)
        with open(join(folder_1, 'letters_2.pickle'), 'rb') as tmp_file:
            letters_2 = pickle.load(tmp_file)
        with open(join(folder_1, 'letters_3.pickle'), 'rb') as tmp_file:
            letters_3 = pickle.load(tmp_file)
        with open(join(folder_1, 'values_1.pickle'), 'rb') as tmp_file:
            values_1 = pickle.load(tmp_file)
        with open(join(folder_1, 'values_2.pickle'), 'rb') as tmp_file:
            values_2 = pickle.load(tmp_file)
        with open(join(folder_1, 'seeds.pickle'), 'rb') as tmp_file:
            seeds = pickle.load(tmp_file)
    
        with open(join(folder_1, 'accuracies.pickle'), 'rb') as tmp_file:
            accuracies = pickle.load(tmp_file)
        with open(join(folder_1, 'metrics.pickle'), 'rb') as tmp_file:
            metrics = pickle.load(tmp_file)
        try:
            with open(join(folder_1, 'basin_volumes.pickle'), 'rb') as tmp_file:
                basin_volumes = pickle.load(tmp_file)
        except FileExistsError:
            basin_volumes = None
    
        if folder_scales[letters_1]['value'] == "log" and np.argwhere(values_1 <= 0).shape[0] > 0:
            folder_scales[letters_1]['value'] = "symlog"
            folder_scales[letters_1]['linthresh'] = np.min(values_1[values_1 > 0])
        if folder_scales[letters_2]['value'] == "log" and np.argwhere(values_2 <= 0).shape[0] > 0:
            folder_scales[letters_2]['value'] = "symlog"
            folder_scales[letters_2]['linthresh'] = np.min(values_2[values_2 > 0])
    
        ids = ["all"]
        cmaps = ["Blues"]
        
        for idi, idx in enumerate(ids):
            
            if idx == "all":
                id_utility = utility
                bv_max = basin_volumes["all_attractors_predicted"]
                acc_extend = "min"
                bv_extend = "max"
            else:
                bv_max = None
                id_utility = 0
                acc_extend = None
                bv_extend = None
                
            if idx == "unstable":
                acc_label = "Fraction Unphysical"
                basin_label = "Fraction Unphysical"
            else:
                acc_label = "Fraction Correct, $f_c$"
                basin_label = "Basin Volume Error, $\\Delta_{BV}$"
            
            if len(values_2) > 1:
                
                ax = comp_ax.flatten()[ri]
                plot_diagonal = plot_diags[ri]
                
                y, x = np.meshgrid(values_1, values_2)
                c = np.mean(accuracies[idx], axis = 2).T#
                
                pcm = ax.pcolormesh(
                    x, y, c, cmap = cmaps[idi],
                    shading = 'nearest',
                    vmax = vmax,
                    vmin = id_utility,
                    )
                if ri % comp_ax.shape[1] == 0:
                    ax.set_ylabel(folder_letters[letters_1])
                ax.set_yscale(**folder_scales[letters_1])
                ax.set_ylim(min(values_1), max(values_1))
                if ri // comp_ax.shape[1] == comp_ax.shape[0] - 1:
                    ax.set_xlabel(folder_letters[letters_2])
                ax.set_xscale(**folder_scales[letters_2])
                ax.set_xlim(min(values_2), max(values_2))
                
                if extra_vertical is not None:
                    ax.axvline(x = extra_vertical, linestyle = "--", c = "w")
                if fp_vline:
                    ax.axvline(x = abs(fixed_pts[0, 0]), linestyle = "--", c = "k")
                if fp_hline:
                    ax.axhline(y = abs(fixed_pts[0, 0]), linestyle = "--", c = "k")
                if plot_diagonal:
                    ax.axline([0, 0], slope = 1, linestyle = "dashdot", c = "k")
                if equal_aspect:
                    ax.set_aspect("equal")
                basin_volumes = None
                if basin_volumes is not None:
                    figure, ax = plt.subplots(1, 1, figsize = (9.5, 7), constrained_layout = True)
                    
                    y, x = np.meshgrid(values_1, values_2)
                    c = np.mean(basin_volumes[idx], axis = 2).T
                    
                    pcm = ax.pcolormesh(
                        x, y, c, cmap = cmaps[idi] + "_r",
                        shading = 'nearest',
                        vmax = bv_max,
                        vmin = 0,
                        )
                    ax.set_ylabel(folder_letters[letters_1])
                    ax.set_yscale(**folder_scales[letters_1])
                    ax.set_ylim(min(values_1), max(values_1))
                    ax.set_xscale(**folder_scales[letters_2])
                    ax.set_xlabel(folder_letters[letters_2])
                    ax.set_xlim(min(values_2), max(values_2))
                    figure.colorbar(pcm, ax = ax, label = basin_label, extend = bv_extend)
                    if extra_vertical is not None:
                        ax.axvline(x = extra_vertical, linestyle = "--", c = "w")
                    if fp_vline:
                        ax.axvline(x = abs(fixed_pts[0, 0]), linestyle = "--", c = "k")
                    if fp_hline:
                        ax.axhline(y = abs(fixed_pts[0, 0]), linestyle = "--", c = "k")
                    if plot_diagonal:
                        ax.axline([0, 0], slope = 1, linestyle = "dashdot", c = "k")
                
                    if equal_aspect:
                        ax.set_aspect("equal")
                
    comp_fig.colorbar(pcm, ax = comp_ax, location = 'bottom',
                      label = acc_label, extend = acc_extend)
