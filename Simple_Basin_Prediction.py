import rescompy as rc
import rc_helpers as rch
import numpy as np
import rescompy.regressions as regressions
import rescompy.features as features
import climate_helpers as climate
import basins_helpers as bh
import test_systems as tst
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Union, List

# Choose Figure
fig_2 = False
fig_3_ab = False
fig_3_cf = False
fig_6 = True #False 
fig_7 = False

if fig_2:
    test_system = "duffing"
    val_grid_width = 6
    grid_val = False
    visible_dimensions = [0, 1]
    IC_rng = (-10, 10)
    val_IC_rng = (-10, 10)
    reduce_fully = False
elif fig_3_ab:
    test_system = "duffing"
    val_grid_width = 25
    grid_val = True
    visible_dimensions = [0]
    val_IC_rng = (-10, 10)
    IC_rng = (-10, 10) #Panels (a)-(c)
    reduce_fully = False #Panels (a) and (b)
elif fig_3_cf:
    test_system = "duffing"
    val_grid_width = 150
    grid_val = True
    visible_dimensions = [0]
    val_IC_rng = (-10, 10)
    IC_rng = (-10, 10) #Panel (a)-(c)
    #IC_rng = (-8, 8) #Panel (d)
    #IC_rng = (-6, 6) #Panel (e)
    #IC_rng = (-4, 4) #Panel (f)
    reduce_fully = True #Panels (c) - (f)
elif fig_6:
    test_system = "magnetic_pendulum"
    val_grid_width = 30
    grid_val = False
    visible_dimensions = [0, 1]
    IC_rng = (-1.5, 1.5)
    val_IC_rng = IC_rng
    reduce_fully = False
elif fig_7:
    test_system = "duffing"
    val_grid_width = 6
    grid_val = False
    visible_dimensions = [0, 1]
    val_IC_rng = (-10, 10)
    IC_rng = (-10, 10) #Panel (a)
    #IC_rng = (-8, 8) #Panel (b)
    #IC_rng = (-6, 6) #Panel (c)
    #IC_rng = (-4, 4) #Panel (d)
    reduce_fully = False

# Set training data parameters (those independent of test system)
lib_length = 500
lib_seed = 50
grid_train = False
train_basin = 0 #None to train across all basins
basin_check_length = 4000 #Number of time steps to generate to establish convergence to desired attractor before including in training data

# Set test data parameters (those independent of test system)
val_lib_seed = 101
val_length = 2000

rotation = 0
use_energies = True
standardize = True #If true, standardizes inputs using standardizer routine
standardizer = None #If None, defaults to standardization to mean zero and range one (if standardize = True)

# Computational Choices
#reduce_fully = False #If true, deletes predicted and true trajectories, and retains only initial and final conditions.
reduce_states = True #If true, deletes reservoir states after each prediction
save_predictions = False
safe_save_predictions = False

if test_system == "duffing":
    
    a, b, c = -0.5, -1., 0.1
    distance_threshold = .5
    test_length = 10
    lib_size = 10
    
    def get_generator(
            a,
            b,
            c,
            transient_length:       int = 0,
            return_length:          int = 4000,
            visible_dimensions:     Union[int, List[int], np.ndarray] = np.arange(2),
            direction:              str = "forward"
            ):
        
        def generator(x0: float, x1: float, seed: int):
            
            x0 = np.array([x0, x1])
                
            return tst.get_unforced_duffing(
                a = a,
                b = b,
                c = c,
                x0 = x0,
                transient_length = transient_length,
                return_length = return_length,
                seed = seed,
                return_dims = visible_dimensions,
                direction = direction
                )
            
        generator.return_length = return_length
        generator.parameter_labels = ['x' + str(i) for i in range(2)]
        
        return generator
    
    train_generator_args = dict(
        a = a,
        b = b,
        c = c,
        transient_length = 0,
        visible_dimensions = visible_dimensions
        )
    val_generator_args = train_generator_args.copy()
    
    if len(visible_dimensions) == 2:
        energy_func = tst.unforced_duffing_energy
        kinetic_func = tst._unforced_duffing_kinetic
        potential_func = tst._unforced_duffing_potential
        energy_args = {"a" : a, "b" : b, "c" : c}
        end_processing = None
    else:
        energy_func = None
        kinetic_func = None
        potential_func = None
        energy_args = None
        def end_processing(x):
            distances = np.sum(np.square(x[-25:]), axis = 1)
            return x[-25:][distances == distances.max()]

    fixed_pts = np.array([[-np.sqrt(-b/c), 0], [np.sqrt(-b/c), 0]])    
    get_basin = bh.get_attractor(
        fixed_points = fixed_pts,
        use_energies = bool(len(visible_dimensions) == 2),
        energy_func = energy_func,
        energy_args = energy_args,
        energy_barrier_loc = np.zeros(2),
        distance_threshold = distance_threshold,
        visible_dimensions = visible_dimensions
        )
    
    colors_map = ["#70d6ff", "#ff70a6"]
    colormap = mpl.colors.LinearSegmentedColormap.from_list("custom", colors_map, N = len(colors_map))
    colormap.set_bad("white")
    colormap.set_under("white")
    colormap.set_over("#e9ff70")
        
    # Define Predictor
    esn_seed = 99
    esn_size = 75
    input_strength = 1
    spectral_radius = .4
    bias_strength = .5
    leaking_rate = 1.
    connections = .03
    feature_function = features.StatesOnly()

    # Set Training
    transient_length = 5
    regularization = 1e-8
    noise_amplitude = 1e-5
    batch_size = 10
    accessible_drives = -1
    
    if len(visible_dimensions) < 2:
        performance_metric = None
    else:
        analytic_step = lambda x : tst.get_unforced_duffing(
            a = a,
            b = b,
            c = c,
            x0 = x.flatten(),
            transient_length = 1,
            return_length = 1,
            return_dims = visible_dimensions
            )
        
        performance_metric = lambda x: climate.get_map_error(
            predictions = x,
            analytic_map = analytic_step,
            discard = 0,
            normalize = False
            )[0]
        performance_metric.name = "Autonomous One-step Error"
       
elif test_system == "magnetic_pendulum":
    height = .2
    frequency = .5
    damping = .2
    lib_size = 100
    test_length = 100
    
    distance_threshold = .25
    
    mag_locs = 1./np.sqrt(3.) * np.array([
        [np.cos(rotation), np.sin(rotation)],
        [np.cos(rotation + 2*np.pi/3.), np.sin(rotation + 2*np.pi/3.)],
        [np.cos(rotation + 4*np.pi/3.), np.sin(rotation + 4*np.pi/3.)]
        ])
    fixed_pts = mag_locs
    
    def get_generator(
            return_length:          int = 4000,
            visible_dimensions:     Union[int, List[int], np.ndarray] = np.arange(4),
            mag_locs:               np.ndarray = np.array([[1./np.sqrt(3), 0.],
                                                           [-1./(2.*np.sqrt(3)), -.5],
                                                           [-1./(2.*np.sqrt(3)), .5]]),
            height:                 float = .2,
            frequency:              float = .5,
            damping:                float = .2,
            direction:              str = "forward"
            ):
        
        def generator(x0: float, x1: float, seed: int):
                
            return tst.get_magnetic_pendulum(initial_state = [x0, x1, 0., 0.],
                                             height = height,
                                             frequency = frequency,
                                             damping = damping,
                                             return_length = return_length,
                                             return_dims = visible_dimensions,
                                             seed = seed, mag_locs = mag_locs,
                                             direction = direction)
        
        generator.return_length = return_length
        generator.parameter_labels = ['x' + str(i) for i in range(2)]
        
        return generator
    
    train_generator_args = dict(
        height = height,
        frequency = frequency,
        damping = damping,
        visible_dimensions = visible_dimensions,
        mag_locs = mag_locs,
        )
    val_generator_args = train_generator_args.copy()
    
    if len(visible_dimensions) == 4:
        energy_func = tst.mp_energy
        kinetic_func = tst.mp_kinetic_energy
        potential_func = tst.mp_potential_energy
        energy_args = dict(
            mag_locs = mag_locs,
            height = height,
            frequency = frequency
            )
        end_processing = None
    else:
        energy_func = None
        kinetic_func = None
        potential_func = None
        energy_args = None
        def end_processing(x):
            distances = np.sum(np.square(x[-25:]), axis = 1)
            return x[-25:][distances == distances.max()]
        
    get_basin = bh.get_attractor(
        fixed_points = mag_locs,
        use_energies = bool(len(visible_dimensions) == 4 and use_energies),
        energy_func = tst.mp_energy,
        energy_args = {'mag_locs': mag_locs, 'height' : height, 'frequency' : frequency},
        energy_barrier_loc = np.zeros(4),
        distance_threshold = distance_threshold,
        visible_dimensions = visible_dimensions
        )
    
    colors_map = ["#ff006e", "#3a86ff", "#ffbe0b"] #["#4cc9f0", "#f72585"] #["#00bbf9", "#f15bb5"] #"tab:cyan", "tab:orange"] #"firebrick", "darkslategray", "tab:orange"] #, np.nan: "black"}
    colormap = mpl.colors.LinearSegmentedColormap.from_list("custom", colors_map, N = len(colors_map))
    colormap.set_bad("white")
    colormap.set_under("white")
    colormap.set_over("white")
    basin_alpha = 1
    
    input_strength = 5
    spectral_radius = .4
    esn_seed = 99
    esn_size = 2500
    bias_strength = .5
    leaking_rate = 1
    connections = .03
    feature_function = features.StatesOnly()

    # Set Training
    transient_length = 25
    regularization = 1e-6
    noise_amplitude = 1e-3
    batch_size = 10
    accessible_drives = -1
    
    if len(visible_dimensions) < 4:
        performance_metric = None
    else:
        analytic_step = lambda x : tst.get_magnetic_pendulum(
            initial_state = x.flatten(),
            height = height,
            transient_length = 1,
            return_length = 1,
            return_dims = visible_dimensions
            )
        performance_metric = lambda x: climate.get_map_error(
            predictions = x,
            analytic_map = analytic_step,
            discard = 0,
            normalize = False
            )[0]
        performance_metric.name = "Autonomous One-step Error"
        performance_metric = None

# Construct the training data library
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
    end_processing = end_processing
    )

# Construct the test data library
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
        connections = int(connections * (esn_size)),
        spectral_radius = spectral_radius,
        input_strength = input_strength,
        bias_strength = bias_strength,
        leaking_rate = leaking_rate
    ))

# Train the predictor
predictor.train(
    library = train_library,
    train_args = dict(
        transient_length = transient_length,
        regression = regressions.batched_ridge(regularization = regularization),
        feature_function = feature_function,
        batch_size = batch_size,
        accessible_drives = accessible_drives
        ),
    return_result = False,
    noise_amp = noise_amplitude
    )

# Get predictions
predictions = predictor.predict(
    library = val_library,
    test_length = test_length,
    return_predictions = True,
    save_predictions = save_predictions,
    safe_save = safe_save_predictions,
    reduce_fully = reduce_fully,
    reduce_states = reduce_states,
    metric_func = performance_metric,
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
print("Total Fraction Correct: ", total_correct)
print("Total Fraction Incorrect: ", total_wrong)
if grid_val:
    bh.basin_heatmap(
        library = val_library,
        overlay_pts = train_library.parameters,
        fixed_points = fixed_pts,
        colormap = colormap,
        font_size = 20,
        transparency = True,
        equal_aspect = True,
        num_ticks = 5,
        overlay_marker = "o",
        box_xbounds = IC_rng if IC_rng != val_IC_rng else None,
        box_ybounds = IC_rng if IC_rng != val_IC_rng else None,
        box_linestyle = "--",
        fp_linewidth = 3,
        )
    bh.basin_heatmap(
        initial_pts = test_ics,
        final_pts = predicted_finals,
        overlay_pts = train_library.parameters,
        fixed_points = fixed_pts,
        get_fixed_pt = get_basin,
        colormap = colormap,
        overlay_marker = "o",
        overlay_color = "k",
        font_size = 20,
        transparency = True,
        equal_aspect = True,
        num_ticks = 5,
        box_xbounds = IC_rng if IC_rng != val_IC_rng else None,
        box_ybounds = IC_rng if IC_rng != val_IC_rng else None,
        box_linestyle = "--",
        fp_linewidth = 3,
        )
    bh.basin_error_heatmap(
        fixed_points = fixed_pts,
        initial_pts = test_ics,
        predicted_finals = predicted_finals,
        true_finals = true_finals,
        colormap = colormap,
        get_fixed_pt = get_basin,
        overlay_pts = train_library.parameters,
        overlay_marker = "o",
        overlay_color = "k",
        font_size = 20,
        transparency = True,
        equal_aspect = True,
        num_ticks = 5,
        box_xbounds = IC_rng if IC_rng != val_IC_rng else None,
        box_ybounds = IC_rng if IC_rng != val_IC_rng else None,
        box_linestyle = "--",
        box_alpha = .75,
        fp_linewidth = 3,
        )

if performance_metric is not None:
    one_steps = np.zeros(len(fixed_pts))
    for basin in range(len(fixed_pts)):
        true_basins = np.nonzero([int(get_basin(p.true_finals) == basin) for p in predictions])[0]
        print(f"Closed, Basin {basin}: ", np.mean([p.performance_metric for p in np.array(predictions)[true_basins]]))

t_basins = [get_basin(p.true_finals) for p in predictions]
p_basins = [get_basin(p.predicted_finals) for p in predictions]
bve, bve_per_basin, f_unphys, bve_thresh = bh.basin_volume_error(true_basins = t_basins, pred_basins = p_basins)
print("Basin Volume Error: ", bve)
print("Basin Volume Error Generization Guarantee: ", bve_thresh)
print("Fraction Unphysical: ", f_unphys)

if not reduce_fully:
    if test_system == "duffing" and len(visible_dimensions) > 1:
        if train_library.standardize:
            train_trajs = [train_library.standardizer.unstandardize(
                u = datum) for datum in train_library.data]
        else:
            train_trajs = train_library.data
        bh.plot_state_space(
            train_trajs = train_trajs,
            train_color = 'tab:grey',
            legend = False,
            legend_loc = 'best',
            n_legend_cols = 1,
            alpha = .5,
            fixed_points = fixed_pts,
            phase_xlims = (-10, 10),
            phase_ylims = (-20, 20),
            font_size = 20.,
            num_ticks = 5
            )
        bh.plot_state_space(
            resync_trajs = [p.prediction.resync_inputs for p in predictions],
            pred_trajs = [p.prediction.reservoir_outputs for p in predictions],
            pred_color = [colors_map[0] if get_basin(p.true_finals) == 0 else colors_map[1]
                          for p in predictions],
            resync_color = [colors_map[0] if get_basin(p.true_finals) == 0 else colors_map[1]
                          for p in predictions],
            legend = False,
            legend_loc = 'best',
            n_legend_cols = 1,
            alpha = .5,
            fixed_points = fixed_pts,
            phase_xlims = (-10, 10),
            phase_ylims = (-20, 20),
            font_size = 20.,
            num_ticks = 5
            )
    else:
        if test_system == "duffing":
            ylims = (-10.5, 10.5)
        else:
            ylims = (-1.6, 1.6)
        
        correct_count = 0
        for p in predictions[::len(predictions)//50]:
            if get_basin(p.true_finals) != train_basin and \
                get_basin(p.predicted_finals) != train_basin:
                    if correct_count == 1:
                        print(p.initial_pts)
                        rch.plot_predict_result(
                            prediction = p.prediction,
                            frame_legend = False,
                            legend_ax = 0,
                            n_legend_cols = 3,
                            figsize = (9, 3),
                            incl_tvalid = False,
                            ylabel = "$x$",
                            font_size = 20,
                            prediction_color = "#4361ee",
                            truth_color = "#6c757d",
                            linewidth = 5,
                            xlims = (-30, val_length + 30), #(-3 * test_length, val_length + 3 * test_length),
                            ylims = ylims,
                            fig_alpha = 0,
                            line_alpha = .75,
                            vert_linewidth = 2.5,
                            t0 = test_length - 1
                            )
                    correct_count += 1
                
        incorrect_count = 0
        for p in predictions:
            if get_basin(p.true_finals) == train_basin and \
                get_basin(p.predicted_finals) != train_basin:
                    if incorrect_count == 2:
                        print(p.initial_pts)
                        rch.plot_predict_result(
                            prediction = p.prediction,
                            frame_legend = False,
                            legend_ax = 0,
                            n_legend_cols = 3,
                            figsize = (9, 3),
                            incl_tvalid = False,
                            ylabel = "$x$",
                            font_size = 20,
                            prediction_color = "#4361ee",
                            truth_color = "#6c757d",
                            linewidth = 5,
                            xlims = (-30, val_length + 30), #(-3 * test_length, val_length + 3 * test_length),
                            ylims = ylims,
                            fig_alpha = 0,
                            line_alpha = .75,
                            vert_linewidth = 2.5,
                            t0 = test_length - 1
                            )
                    incorrect_count += 1
        
    if test_system == "magnetic_pendulum":
        with mpl.rc_context({"font.size" : 20}):
            num_each = 2,
            avg_type = None
            normalization = False
            error_type = "euclidean"
            decimation = None     
            log_scale = False
            incorrect_color = "k"
            plot_distance_threshold = True
            err_xticks = [test_length] + [500, 1000, 1500, 2000]
            err_yticks = [0.5, 1.]
            err_linewidth = 2
            if log_scale:
                n_bins = 100
                b_xlims = (0, .4)
                e_ylims = (1e-5, 1e1)
                bins = np.logspace(np.log10(e_ylims[0]), np.log10(e_ylims[1]), n_bins)
            else:
                n_bins = 50
                b_xlims = (0, .7)
                e_ylims = (0, 1.25)
                bins = np.linspace(e_ylims[0], e_ylims[1], n_bins)
            
            bar_n_bins = 100
            bar_xlims = (0, .5)
            bar_ylims = (1e-3, 1e1)
            bar_yticks = [1e-3, 1e-1, 1e1]
            bar_xticks1 = [.04, .08, .12]
            bar_xticks2 = [.2, .4]
            bar_bins = np.logspace(np.log10(bar_ylims[0]), np.log10(bar_ylims[1]), n_bins)
            bar_scale = "linear"
            bar_error_scale = "log"
            bar_orientation = "vertical"
            
            if error_type != "euclidean" or normalization:
                plot_distance_threshold = False
            bar_alpha = .5
            err_alpha = 1
            
            e_fig, e_axes = plt.subplots(
                3, 1, figsize = (9, 9),
                constrained_layout = True,
                sharex = True,
                sharey = True
                )
            bar_fig, bar_axes = plt.subplots(
                1, 1,
                figsize = (9, 3),
                constrained_layout = True,
                )
            bar_ax = bar_axes
            for basin_id in range(len(fixed_pts)):
                e_axs = e_axes[basin_id]
                p_basin_ids = np.nonzero([int(get_basin(p.true_finals) == basin_id) for p in predictions])[0]
                b_predictions = [p.prediction for p in np.array(predictions)[p_basin_ids]]
                predicted_basins = [get_basin(p.predicted_finals) for p in np.array(predictions)[p_basin_ids]]
                print(basin_id)
                print(len(b_predictions))
                rch.plot_predict_result_errors(
                    predictions = b_predictions,
                    predicted_basins = predicted_basins,
                    error_threshold = distance_threshold,
                    normalize = normalization,
                    error_type = error_type,
                    frame_legend = False,
                    decimation = decimation,
                    legend_ax = 0,
                    n_legend_cols = 3,
                    figsize = (9, 3),
                    xlabel = "Time (Time Steps, $\\Delta t$)" if basin_id == len(fixed_pts) - 1 else None,
                    fig = e_fig,
                    axes = e_axs,
                    bar_axes = None,
                    average = avg_type,
                    spread = None,
                    incl_tvalid = False,
                    ylabel = "Error, $\\varepsilon$",
                    font_size = 20,
                    linewidth = err_linewidth,
                    xlims = (test_length - 10, val_length + 10),
                    ylims = e_ylims,
                    fig_alpha = 0,
                    line_alpha = err_alpha,
                    vert_linewidth = 2.5,
                    bar_alpha = bar_alpha,
                    bar_bins = bins,
                    t0 = test_length - 1,
                    incorrect_color = incorrect_color,
                    colormap = colormap,
                    xticks = err_xticks,
                    yticks = err_yticks,
                    num_each = num_each,
                    num_basins = len(fixed_pts)
                    )
                
                if log_scale:
                    e_axs.set_yscale("log")
                if plot_distance_threshold:
                    if isinstance(e_axs, np.ndarray):
                        for ax in e_axs:
                            ax.axhline(distance_threshold, c = "k", linestyle = "--")
                    else:
                        e_axs.axhline(distance_threshold, c = "k", linestyle = "--")
                rch.plot_errors_histograms(
                    predictions = b_predictions,
                    predicted_basins = predicted_basins,
                    error_threshold = distance_threshold,
                    normalize = normalization,
                    error_type = error_type,
                    fig = bar_fig,
                    ax = bar_axes,
                    font_size = 20,
                    color = colormap(basin_id),
                    fig_alpha = 0,
                    vert_linewidth = 2.5,
                    bar_alpha = bar_alpha,
                    bar_bins = bar_bins,
                    xlims = bar_xlims,
                    ylims = bar_ylims,
                    orientation = bar_orientation,
                    xticks = bar_xticks2,
                    yticks = bar_yticks,
                    bar_scale = bar_scale,
                    bar_error_scale = bar_error_scale,
                    t0 = -25,
                    t1 = None                    
                    )