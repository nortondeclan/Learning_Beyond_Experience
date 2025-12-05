import rescompy as rc
import rc_helpers as rch
import numpy as np
import rescompy.regressions as regressions
import rescompy.features as features
import basins_helpers as bh
import test_systems as tst
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Union, List
from dysts.metrics import estimate_kl_divergence

fig_7a = True
fig_7bc = False #True

# Library Parameters
lib_size = 1
visible_dimensions = [0, 1, 2]
lib_seed = 50
val_lib_seed = 101
grid_train = False
train_basin = 0
use_energies = True
standardize = True
standardizer = None
mean_reg = True

if fig_7a:
    val_grid_width = 6
    grid_val = False
    test_length = 50
    
elif fig_7bc:               #Warning: this may take hours, because of the grid resolution (and KL calcations) 
    val_grid_width = 100 #6   #100 used in Fig.s 7b and 7c, but lower resolution will be faster
    grid_val = True
    test_length = 5

# Computational Choices
reduce_fully = False
reduce_states = True
save_predictions = False
safe_save_predictions = False

# Define Experiment
return_every = 2    # Sample spacing for the reservoir computer is twice the sampling rate of the integrated Lorenz trajectories
val_length = 5000 * return_every
lib_length = 5000 * return_every
kl_length = 500
basin_check_length = lib_length
integrator_transient = 0
kl_threshold = 1.

sigma = -10.
beta = -4.
rho = 18.1

# Generate reference trajectories on each attractor to for use in calculating KL divergences
ref0 = tst.get_multistable_lorenz(
    x0 = [0, 0, -40],
    sigma = sigma,
    beta = beta,
    rho = rho,
    transient_length = 500,
    return_length = kl_length,
    return_dims = visible_dimensions,
    return_every = return_every
    )
ref1 = tst.get_multistable_lorenz(
    x0 = [0, 0, 40],
    sigma = sigma,
    beta = beta,
    rho = rho,
    transient_length = 500,
    return_length = kl_length,
    return_dims = visible_dimensions,
    return_every = return_every
    )

def get_generator(
        sigma:                  Union[float, List[float]] = sigma,
        beta:                   Union[float, List[float]] = beta,
        rho:                    Union[float, List[float]] = rho,
        transient_length:       Union[int, tuple] = 0,
        return_length:          int = 4000,
        visible_dimensions:     Union[int, List[int], np.ndarray] = np.arange(3),
        direction:              str = "forward"
        ):
        
    def generator(x0: float, x1: float, x2: float, seed: int):
        
        return tst.get_multistable_lorenz(
            sigma = sigma,
            beta = beta,
            rho = rho,
            x0 = [x0, x1, x2],
            transient_length = transient_length,
            return_length = return_length,
            h = .01,
            seed = seed,
            return_dims = visible_dimensions,
            return_every = return_every
            )
    
    generator.return_length = return_length
    generator.parameter_labels = ['x' + str(i) for i in range(3)]
    
    return generator

train_generator_args = dict(
    visible_dimensions = visible_dimensions,
    transient_length = integrator_transient,
    sigma = sigma,
    beta = beta,
    rho = rho
    )
val_generator_args = dict(
    visible_dimensions = visible_dimensions,
    transient_length = integrator_transient,
    sigma = sigma,
    beta = beta,
    rho = rho
    )

fixed_pts = np.array([None, None])

def end_processing(trajectory):
    
    kl0 = estimate_kl_divergence(ref0, trajectory[-kl_length:])
    kl1 = estimate_kl_divergence(ref1, trajectory[-kl_length:])
    
    return np.array([kl0, kl1])
    
def get_basin(kls):
    
    if kls[0] < kl_threshold and kls[1] < kl_threshold:
        print("Warning: Both attractors below kl_threshold.")
        return -1
    elif kls[0] < kl_threshold:
        return 0
    elif kls[1] < kl_threshold:
        return 1
    else:
        return np.nan

colors_map = ["#70d6ff", "#ff70a6"]
colormap = mpl.colors.LinearSegmentedColormap.from_list("custom", colors_map, N = len(colors_map))
colormap.set_bad("white")
colormap.set_under("white")
colormap.set_over("#e9ff70")
train_color = "k"
 
IC_rng = (-40, 40)
val_IC_rng = IC_rng

phase_xlims = IC_rng
phase_ylims = IC_rng
phase_zlims = IC_rng

# Define Predictor
esn_seed = 99
esn_size = 500
if standardize:
    input_strength = .5
else:
    input_strength = .5 / abs(IC_rng[0])
spectral_radius = .4
bias_strength = .5
leaking_rate = 1.
connections = 10
feature_function = features.StatesOnly()

# Set Training
transient_length = 5
if mean_reg:
    regularization = 1e-10
else:
    regularization = 1e-6
noise_amplitude = 1e-3
batch_size = 10
accessible_drives = -1

if grid_val:
    initials_1d = np.linspace(val_IC_rng[0], val_IC_rng[1], val_grid_width)
    val_initials = list(np.array([
        [0, initials_1d[i], initials_1d[j]]
        for i in range(val_grid_width)
        for j in range(val_grid_width)
        ]))
else:
    val_initials = None

# Construct the training library
train_library = bh.get_lib(
    get_generator = get_generator,
    get_basin = get_basin,
    generator_args = train_generator_args,
    lib_size = lib_size,
    lib_length = lib_length,
    basin_check_length = basin_check_length,
    seed = lib_seed,
    train_basin = train_basin,
    grid = grid_train,
    IC_rng = IC_rng,
    standardize = standardize,
    standardizer = standardizer,
    end_processing = end_processing,
    end_not_state = True
    )

print("Have Training Data")

# Construct the test library
val_library = bh.get_lib(
    get_generator = get_generator,
    get_basin = get_basin,
    generator_args = val_generator_args,
    lib_size = val_grid_width**2,
    lib_length = val_length,
    basin_check_length = basin_check_length,
    seed = val_lib_seed,
    train_basin = None,
    initial_conditions = val_initials,
    grid = False,
    IC_rng = val_IC_rng,
    standardize = standardize,
    standardizer = train_library.standardizer,
    end_processing = end_processing,
    end_not_state = True
    )

print("Have Test Data")

# Construct the predictor
predictor = bh.batch_predictor(
    reservoir = rc.ESN(
        input_dimension = len(visible_dimensions),
        seed = esn_seed,
        size = esn_size,
        connections = connections,
        spectral_radius = spectral_radius,
        input_strength = input_strength,
        bias_strength = bias_strength,
        leaking_rate = leaking_rate
    ))

if mean_reg:
    num_fit = lib_length * lib_size - lib_size * (transient_length + 1)
else:
    num_fit = 1
    
# Train the predictor
predictor.train(
    library = train_library,
    train_args = dict(
        transient_length = transient_length,
        regression = regressions.batched_ridge(
            regularization = regularization * num_fit 
            ),
        feature_function = feature_function,
        batch_size = batch_size,
        accessible_drives = accessible_drives
        ),
    return_result = False,
    noise_amp = noise_amplitude
    )

# Obtain predictions
predictions = predictor.predict(
    library = val_library,
    test_length = test_length,
    return_predictions = True,
    save_predictions = save_predictions,
    safe_save = safe_save_predictions,
    reduce_fully = reduce_fully,
    reduce_states = reduce_states,
    metric_func = None,
    end_processing = end_processing
    )

predicted_finals = [p.predicted_finals for p in predictions]
true_finals = [p.true_finals for p in predictions]
training_ics = train_library.parameters

print("Have Predictions")

total_correct = len(np.nonzero([
    get_basin(p.predicted_finals) == get_basin(p.true_finals)
    for p in predictions
    ])[0]) / len(predictions)
total_wrong = len(np.nonzero([
    get_basin(p.predicted_finals) != get_basin(p.true_finals)
    for p in predictions
    ])[0]) / len(predictions)
for basin_id in range(len(fixed_pts)):
    try:
        print(f"Basin {basin_id} Total: ",
        len(np.nonzero([
            get_basin(p.true_finals) == basin_id
            for p in predictions
            ])[0])
        )
        print(
            f"Basin {basin_id} Fraction Correct: ",
            len(np.nonzero([
                get_basin(p.true_finals) == basin_id and get_basin(p.predicted_finals) == get_basin(p.true_finals)
                for p in predictions
                ])[0]) / 
            len(np.nonzero([
                get_basin(p.true_finals) == basin_id for p in predictions
                ])[0])
            )
    except ZeroDivisionError:
        print(f"No samples from basin {basin_id}.")
print("Total Fraction Correct: ", total_correct)
print("Total Fraction Incorrect: ", total_wrong)

if grid_val:
    test_ics_2d = [entry[[1, 2]] for entry in val_initials]
    train_ics_2d = [entry[[1, 2]] for entry in train_library.parameters]
    
    bh.basin_error_heatmap(
        fixed_points = fixed_pts,
        initial_pts = test_ics_2d,
        predicted_finals = predicted_finals,
        true_finals = true_finals,
        colormap = colormap,
        get_fixed_pt = get_basin,
        overlay_marker = "o",
        overlay_color = "k",
        font_size = 16,
        transparency = True,
        equal_aspect = True,
        num_ticks = 5,
        box_xbounds = None,
        box_ybounds = None,
        box_linestyle = "--",
        box_alpha = .75,
        fp_linewidth = 3,
        xlabel = "$y_0$",
        ylabel = "$z_0$",
        fp_dim0 = 1,
        fp_dim1 = 2,
        plot_fixed_pts = False
        )

t_basins = [get_basin(p.true_finals) for p in predictions]
p_basins = [get_basin(p.predicted_finals) for p in predictions]
bve, bve_per_basin, f_unphys, bve_thresh = bh.basin_volume_error(true_basins = t_basins, pred_basins = p_basins)
print("Basin Volume Error: ", bve)
print("Basin Volume Error Generization Guarantee: ", bve_thresh)
print("Fraction Unphysical: ", f_unphys)

if not reduce_fully:
    if len(visible_dimensions) > 1:
        if train_library.standardize:
            train_trajs = [train_library.standardizer.unstandardize(
                u = datum) for datum in train_library.data]
        else:
            train_trajs = train_library.data
            
        # Plot the training trajectory and predicted trajectories to visualize the chaotic attractors
        bh.plot_state_space(
            train_trajs = train_trajs,
            train_color = train_color,
            pred_trajs = [p.prediction.reservoir_outputs for p in predictions],
            pred_color = [colormap(get_basin(p.true_finals))
                          for p in predictions],
            resync_color = [colormap(get_basin(p.true_finals))
                            for p in predictions],
            legend = False,
            legend_loc = 'best',
            n_legend_cols = 1,
            alpha = .1,
            train_alpha = 1.,
            phase_xlims = phase_xlims,
            phase_ylims = phase_ylims,
            phase_zlims = phase_zlims,
            font_size = 16.,
            num_ticks = 5,
            plot_3d = len(visible_dimensions) == 3,
            plot_dims = [0,1,2] if len(visible_dimensions) == 3 else [0, 1],
            train_linestyle = (0, (2, 2))
            )
        

# Plot the distributions of the KL divergence between true and predicted trajectories in each basin
with mpl.rc_context({"font.size" : 20}):
    n_bins = 50
    inter_attractor_error = .5 * (
        estimate_kl_divergence(ref0, ref1) +
        estimate_kl_divergence(ref1, ref0)
        )
    
    bar_xlims = (0, .29)
    bar_ylims = (1e-2, 2 * inter_attractor_error)
    bar_yticks = [1e-2, 1e-1, 1e0, 1e1]
    bar_xticks2 = [.25, .5, .75, 1.]    
    bar_xticks2 = [.05, .1, .15, .2, .25]
    bar_bins = np.logspace(np.log10(bar_ylims[0]), np.log10(bar_ylims[1]), n_bins)
    bar_scale = "linear"
    bar_error_scale = "log"
    bar_orientation = "vertical"
    bar_alpha = .5

    bar_fig, bar_axes = plt.subplots(
        1,
        1,
        figsize = (6.5, 6),
        constrained_layout = True
        )
    bar_ax = bar_axes
    for basin_id in range(len(fixed_pts)):
        p_basin_ids = np.nonzero([int(get_basin(p.true_finals) == basin_id) for p in predictions])[0]
        b_predictions = [p.prediction for p in np.array(predictions)[p_basin_ids]]
        bkl_predictions = [p for p in np.array(predictions)[p_basin_ids]]
        predicted_basins = [get_basin(p.predicted_finals) for p in np.array(predictions)[p_basin_ids]]
        true_basins = [get_basin(p.true_finals) for p in np.array(predictions)[p_basin_ids]]
        print(basin_id)
        print(len(b_predictions))

        rch.plot_attractor_error_histograms(
            predictions = bkl_predictions,
            true_basins = true_basins,
            predicted_basins = predicted_basins,
            error_threshold = kl_threshold,
            error_type = "kl",
            inter_attractor_error = inter_attractor_error,
            fig = bar_fig,
            ax = bar_axes,
            font_size = 20,
            color = colormap(basin_id),
            fig_alpha = 0,
            vert_linewidth = 3,
            bar_alpha = bar_alpha,
            bar_bins = bar_bins,
            xlims = bar_xlims,
            ylims = bar_ylims,
            orientation = bar_orientation,
            xticks = bar_xticks2,
            yticks = bar_yticks,
            bar_scale = bar_scale,
            bar_error_scale = bar_error_scale,
            make_legend = basin_id == len(fixed_pts) - 1        
            )