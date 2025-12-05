import rescompy as rc
import numpy as np
import rescompy.regressions as regressions
import rescompy.features as features
import basins_helpers as bh
import test_systems as tst
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Union, List

# Library Parameters
lib_size = 25
lib_length = 500
val_grid_width = 10
basin_check_length = 4000
visible_dimensions = [0, 1]
lib_seed = 50
val_lib_seed = 101
grid_train = False
grid_val = False

use_energies = True
process_noise = 0
val_process_noise = 0
mean_reg = True

# Computational Choices
reduce_fully = False
reduce_states = True
num_cores = 1
save_predictions = False
safe_save_predictions = False

a, b, c, d = 1., .5, 1., .5 
val_length = 2000

fixed_pts = np.array([
    [0.5 * a**2/b, 0.5 * c**2/d],
    [- 0.5 * a**2/b, 0.5 * c**2/d],
    [- 0.5 * a**2/b, - 0.5 * c**2/d],
    [0.5 * a**2/b, - 0.5 * c**2/d]
    ])
        
distance_threshold = 0.2

def get_generator(
        a,
        b,
        c,
        d,
        transient_length:       int = 0,
        return_length:          int = 4000,
        visible_dimensions:     Union[int, List[int], np.ndarray] = np.arange(2),
        direction:              str = "forward",
        return_time:            bool = True,
        process_noise:          float = 0.,
        ):
    
    def generator(x0: float, x1: float, seed: int):
        
        x0 = np.array([x0, x1])
            
        return tst.get_double_well(
            a = a,
            b = b,
            c = c,
            d = d,
            x0 = x0,
            transient_length = transient_length,
            return_length = return_length,
            seed = seed,
            return_dims = visible_dimensions,
            direction = direction,
            process_noise = process_noise,
            h = .01
            )
        
    generator.return_length = return_length
    generator.parameter_labels = ['x' + str(i) for i in range(2)]
    
    return generator

train_generator_args = dict(
    a = a,
    b = b,
    c = c,
    d = d,
    transient_length = 0,
    visible_dimensions = visible_dimensions,
    process_noise = process_noise
    )
val_generator_args = train_generator_args.copy()
val_generator_args["process_noise"] = val_process_noise

IC_rng = (-4, 4)
val_IC_rng = IC_rng

phase_xlims = val_IC_rng
phase_ylims = val_IC_rng
phase_zlims = val_IC_rng
train_color = "k"

if len(visible_dimensions) == 2:
    energy_func = tst.double_well_energy
    kinetic_func = tst._double_well_kinetic
    potential_func = tst._double_well_potential
    energy_args = {"a" : a, "b" : b, "c" : c, "d" : d}
    end_processing = None
else:
    energy_func = None
    kinetic_func = None
    potential_func = None
    energy_args = None
    
    def end_processing(x):
        
        fp_dist = np.inf
        for fp in fixed_pts:
            distances = np.sum(np.square(x[-25:] - fp), axis = 1)
            if distances.max() < fp_dist:
                finals = x[-25:][distances == distances.max()]
                if finals.shape[0] > 1:
                    finals = finals[0]
                    
        return finals

get_basin = bh.get_attractor(
    fixed_points = fixed_pts,
    use_energies = bool(len(visible_dimensions) == 2),
    energy_func = energy_func,
    energy_args = energy_args,
    energy_barrier_loc = np.zeros(2),
    distance_threshold = distance_threshold,
    visible_dimensions = visible_dimensions
    )

colors_map = ["#70d6ff", "#ff70a6", "#087e8b", "#87a330"]
colormap = mpl.colors.LinearSegmentedColormap.from_list("custom", colors_map, N = len(colors_map))
colormap.set_bad("#ff7d00")
colormap.set_under("#ff7d00")
colormap.set_over("#e9ff70")
        
# Define Predictor
esn_seed = 99
esn_size = 200
spectral_radius = .4
bias_strength = .5
leaking_rate = 1.
connections = 10
feature_function = features.StatesOnly()

# Set Training
transient_length = 5
if mean_reg:
    regularization = 1e-12
else:
    regularization = 1e-8
noise_amplitude = 1e-5
batch_size = 10
accessible_drives = -1

# Define Experiment
test_length = 5
train_basins = [[1, 3], [1, 2], [1]]

with mpl.rc_context({"font.size" : 20}):
    comp_fig, comp_ax = plt.subplots(
        3, 3, figsize = (10, 12), constrained_layout = True,
        sharex = True, sharey = True
        )
    
    for ti, train_basin in enumerate(train_basins):
        for si, standardize in enumerate([True, False]):
            
            if standardize:
                input_strength = 1
            else:
                input_strength = .25
        
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
                #standardizer = standardizer,
                end_processing = end_processing,
                )
            
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
                grid = grid_val,
                IC_rng = val_IC_rng,
                standardize = standardize,
                standardizer = train_library.standardizer,
                end_processing = end_processing
                )
            
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
            
            predictions = predictor.predict(
                library = val_library,
                test_length = test_length,
                return_predictions = True,
                save_predictions = save_predictions,
                safe_save = safe_save_predictions,
                reduce_fully = reduce_fully,
                reduce_states = reduce_states,
                end_processing = end_processing
                )
            
            test_ics = val_library.parameters
            predicted_finals = [p.predicted_finals for p in predictions]
            true_finals = [p.true_finals for p in predictions]
            training_ics = train_library.parameters
            
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
                        
                    bh.plot_state_space(
                        resync_trajs = [p.prediction.resync_inputs for p in predictions],
                        pred_trajs = [p.prediction.reservoir_outputs for p in predictions],
                        pred_color = [colormap(tbasin) if tbasin == pbasin else colormap(np.nan)
                                      for tbasin, pbasin in zip(t_basins, p_basins)],
                        resync_color = [colormap(tbasin) if tbasin == pbasin else colormap(np.nan)
                                        for tbasin, pbasin in zip(t_basins, p_basins)],
                        legend = False,
                        legend_loc = 'best',
                        n_legend_cols = 1,
                        alpha = .5,
                        fixed_points = fixed_pts,
                        phase_xlims = phase_xlims,
                        phase_ylims = phase_ylims,
                        phase_zlims = phase_zlims,
                        font_size = 20.,
                        num_ticks = 5,
                        plot_3d = False,
                        plot_dims = [0, 1],
                        fig = comp_fig,
                        ax = comp_ax[ti, si + 1],
                        xlabel = None,
                        ylabel = None,
                        zlabel = None
                        )
                    
                    if si == 0:
                        bh.plot_state_space(
                            train_trajs = train_trajs,
                            train_color = train_color,
                            legend = False,
                            legend_loc = 'best',
                            n_legend_cols = 1,
                            alpha = .5,
                            train_alpha = .25,
                            fixed_points = fixed_pts,
                            phase_xlims = phase_xlims,
                            phase_ylims = phase_ylims,
                            phase_zlims = phase_zlims,
                            font_size = 20.,
                            num_ticks = 5,
                            plot_3d = False,
                            plot_dims = [0, 1],
                            fig = comp_fig,
                            ax = comp_ax[ti, 0],
                            xlabel = None,
                            ylabel = None,
                            zlabel = None
                            )
                        
                        for b_id, fp in enumerate(fixed_pts):
                            comp_ax[ti, 0].add_patch(mpl.patches.Rectangle(
                                (0, 0), width = 4, height = 4,
                                angle = (360 / len(fixed_pts)) * b_id,
                                facecolor = colormap(get_basin(fp)), alpha = 0.25
                                ))
                    
                    comp_ax[ti, si].set_xlim(*phase_xlims)
                    comp_ax[ti, si].set_ylim(*phase_ylims)
                    comp_ax[ti, si].set_aspect('equal')
                    
                    for ax in comp_ax[:, 0]:
                        ax.set_ylabel("$y$")
                    for ax in comp_ax[-1, :]:
                        ax.set_xlabel("$x$")