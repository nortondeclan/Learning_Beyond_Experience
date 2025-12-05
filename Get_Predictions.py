import rescompy as rc
import rc_helpers as rch
import numpy as np
import rescompy.regressions as regressions
import rescompy.features as features
import os
import climate_helpers as climate
import basins_helpers as bh
import test_systems as tst
import argparse
from typing import Union, List
import pickle

join = os.path.join

skip_repeats = True

parser = argparse.ArgumentParser()
parser.add_argument('--test_system', required = True, choices = ['mag', 'duff', 'fduff'])
parser.add_argument('--partial_state', type = int, required = False, default = False)
parser.add_argument('--reduce_fully', type = int, required = False, default = True)
parser.add_argument('--test_length', type = int, required = True)
parser.add_argument('--IC_lim', type = float, default = 1)
parser.add_argument('--lib_size', type = int, default = 100)
parser.add_argument('--val_grid_width', type = int, default = 50)
parser.add_argument('--distance_threshold', type = float, default = None)
parser.add_argument('--esn_size', type = int, default = 500)
parser.add_argument('--regularization', type = float, default = 1e-7)
parser.add_argument('--noise_amplitude', type = float, default = 1e-5)
parser.add_argument('--train_seed', type = int, default = 1000)
parser.add_argument('--folder_one', required = True)
parser.add_argument('--folder_two', default = None, required = False)
parser.add_argument('--extra_name', default = None, required = False)
parser.add_argument('--val_IC_lim', type = float, default = None, required = False)
parser.add_argument('--train_basin', type = int, default = None, required = False)
parser.add_argument('--process_noise', type = float, default = 0., required = False)
parser.add_argument('--val_process_noise', type = float, default = 0., required = False)
parser.add_argument('--rc_time_step', type = int, default = 1, required = False)
parser.add_argument('--leakage', type = float, default = 1., required = False)
parser.add_argument('--connections', type = int, default = 10, required = False)
parser.add_argument('--integrator_transient', type = int, default = 0, required = False)
parser.add_argument('--grid_val', type = int, required = False, default = True)
parser.add_argument('--mean_reg', type = int, required = False, default = True)
c = parser.parse_args()

test_system = c.test_system
partial_state = bool(c.partial_state)
reduce_fully = bool(c.reduce_fully)
test_length = c.test_length
IC_lim = c.IC_lim
lib_size = c.lib_size
val_grid_width = c.val_grid_width
distance_threshold = c.distance_threshold
esn_size = c.esn_size
regularization = c.regularization
noise_amplitude = c.noise_amplitude
train_seed = c.train_seed
folder_one = c.folder_one
folder_two = c.folder_two
extra_name = c.extra_name
val_IC_lim = c.val_IC_lim
if c.train_basin < 0:
    train_basin = None
else:
    train_basin = c.train_basin
rc_time_step = c.rc_time_step
process_noise = c.process_noise
val_process_noise = c.val_process_noise
leakage = c.leakage
connections = c.connections
integrator_transient = c.integrator_transient
grid_val = bool(c.grid_val)
mean_reg = bool(c.mean_reg)

print("Partial_State: ", partial_state)

if hasattr(c, 'extra_name'):
    is_extra_name = True
else:
    is_extra_name = False
    
folder_letters = {
    "tl" : str(test_length),
    "ic" : str(IC_lim),
    "nl" : str(lib_size),
    "dt" : str(distance_threshold),
    "s" : str(esn_size),
    "r" : str(regularization),
    "n" : str(noise_amplitude),
    "vc" : str(val_IC_lim),
    "rt" : str(rc_time_step),
    "pn" : str(process_noise),
    "vp" : str(val_process_noise),
    "lk" : str(leakage),
    "c" : str(connections),
    "it" : str(integrator_transient)
    }

# Library Parameters
basin_check_length = 4000    
grid_train = False
rotation = 0
use_energies = True
open_loop = False
closed_loop = True
standardize = True
IC_rng = (-IC_lim, IC_lim)
if val_IC_lim is None:
    val_IC_rng = IC_rng
else:
    val_IC_rng = (-val_IC_lim, val_IC_lim)

# Computational Choices
reduce_states = True
num_cores = 1
save_predictions = True
safe_save_predictions = False

if test_system == "duff":
    a, b, c = -0.5, -1., 0.1
    lib_length = 500
    val_length = 2000
    
    fixed_pts = np.array([[-np.sqrt(-b/c), 0], [np.sqrt(-b/c), 0]])
    
    if partial_state:
        visible_dimensions = [0]
    else:
        visible_dimensions = [0, 1]
    
    def get_generator(
            a,
            b,
            c,
            transient_length:       int = 0,
            return_length:          int = 4000,
            visible_dimensions:     Union[int, List[int], np.ndarray] = np.arange(2),
            direction:              str = "forward",
            process_noise:          float = 0.,
            return_every:           int = 1
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
                direction = direction,
                return_every = return_every,
                process_noise = process_noise
                )
            
        generator.return_length = return_length
        generator.parameter_labels = ['x' + str(i) for i in range(2)]
        
        return generator
    
    train_generator_args = dict(
        a = a,
        b = b,
        c = c,
        visible_dimensions = visible_dimensions,
        process_noise = process_noise,
        transient_length = integrator_transient,
        return_every = rc_time_step
        )
    val_generator_args = train_generator_args.copy()
    val_generator_args["process_noise"] = val_process_noise

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
    
    # Define Predictor
    esn_seed = 99
    input_strength = 1
    spectral_radius = .4
    bias_strength = .5
    leaking_rate = leakage
    feature_function = features.StatesOnly()

    # Set Training
    transient_length = 5
    batch_size = 10
    accessible_drives = -1
    
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
       
elif test_system == "fduff":
    a, b, c, f0, fv, w = -0.5, -1., 0.1, 1., 0., .1 # Asymmetric
    # a, b, c, f0, fv, w = -0.5, -1., 0.1, 0., 0., 2. #.5 # Periodic
    # a, b, c, f0, fv, w = -0.5, -1., 0.1, 0., 0., .1 # Unforced
    lib_length = 500
    val_length = 2000
    
    if f0 == 0 and fv == 0:
        fixed_pts = np.array([[-np.sqrt(-b/c), 0], [np.sqrt(-b/c), 0]])
    if f0 == 1:
        fixed_pts = np.array([
            [-2.42362, 0],
            #[-1.15347, 0], #Unstable
            [3.57709, 0]
            ])
    
    if partial_state:
        visible_dimensions = [0]
    else:
        visible_dimensions = [0, 1]
    
    def get_generator(
            a,
            b,
            c,
            f0,
            fv,
            w,
            transient_length:       int = 0,
            return_length:          int = 4000,
            visible_dimensions:     Union[int, List[int], np.ndarray] = np.arange(2),
            direction:              str = "forward",
            return_time:            bool = True,
            process_noise:          float = 0.,
            return_every:           int = 1
            ):
        
        def generator(x0: float, x1: float, seed: int):
            
            x0 = np.array([x0, x1])
                
            return tst.get_forced_duffing(
                a = a,
                b = b,
                c = c,
                f0 = f0,
                fv = fv,
                w = w,
                x0 = x0,
                transient_length = transient_length,
                return_length = return_length,
                seed = seed,
                return_dims = visible_dimensions,
                direction = direction,
                process_noise = process_noise,
                return_every = return_every
                )
            
        generator.return_length = return_length
        generator.parameter_labels = ['x' + str(i) for i in range(2)]
        
        return generator
    
    train_generator_args = dict(
        a = a,
        b = b,
        c = c,
        f0 = f0,
        fv = fv,
        w = w,
        transient_length = integrator_transient,
        visible_dimensions = visible_dimensions,
        process_noise = process_noise,
        return_every = rc_time_step
        )
    val_generator_args = train_generator_args.copy()
    val_generator_args["process_noise"] = val_process_noise
    
    if len(visible_dimensions) == 2:
        energy_func = tst.unforced_duffing_energy
        kinetic_func = tst._unforced_duffing_kinetic
        potential_func = tst._unforced_duffing_potential
        energy_args = {"a" : a, "b" : b, "c" : c}
        
        def end_processing(x):
            
            fp_dist = np.inf
            for fp in fixed_pts:
                distances = np.sum(np.square(x[-25:] - fp), axis = 1)
                if distances.max() < fp_dist:
                    finals = x[-25:][distances == distances.max()]
                    if finals.shape[0] > 1:
                        finals = finals[0]
                        
            return finals
        
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
      
    # Define Predictor
    esn_seed = 99
    input_strength = 1
    spectral_radius = .4
    bias_strength = .5
    leaking_rate = leakage
    feature_function = features.StatesOnly()

    # Set Training
    transient_length = 5
    batch_size = 10
    accessible_drives = -1
    
    performance_metric = None

elif test_system == "mag":
    height = .2
    frequency = .5
    damping = .2
    lib_length = 500
    val_length = 2000
    if partial_state:
        visible_dimensions = [0, 1]
    else:
        visible_dimensions = [0, 1, 2, 3]
    
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
            direction:              str = "forward",
            transient_length:       int = 0,
            process_noise:          float = 0.,
            return_every:           int = 1
            ):
        
        def generator(x0: float, x1: float, seed: int):
                
            return tst.get_magnetic_pendulum(
                initial_state = [x0, x1, 0., 0.],
                height = height,
                frequency = frequency,
                damping = damping,
                return_length = return_length,
                return_dims = visible_dimensions,
                seed = seed,
                mag_locs = mag_locs,
                direction = direction,
                transient_length = transient_length,
                #process_noise = process_noise,
                return_every = return_every
                )
        
        generator.return_length = return_length
        generator.parameter_labels = ['x' + str(i) for i in range(2)]
        
        return generator
    
    train_generator_args = dict(
        height = height,
        frequency = frequency,
        damping = damping,
        visible_dimensions = visible_dimensions,
        mag_locs = mag_locs,
        transient_length = integrator_transient,
        process_noise = process_noise,
        return_every = rc_time_step
        )
    val_generator_args = train_generator_args.copy()
    val_generator_args["process_noise"] = val_process_noise
    
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
            
            fp_dist = np.inf
            for fp in fixed_pts:
                distances = np.sum(np.square(x[-25:] - fp), axis = 1)
                if distances.max() < fp_dist:
                    finals = x[-25:][distances == distances.max()]
                    if finals.shape[0] > 1:
                        finals = finals[0]
                        
            return finals
        
    get_basin = bh.get_attractor(
        fixed_points = mag_locs,
        use_energies = bool(len(visible_dimensions) == 4 and use_energies),
        energy_func = tst.mp_energy,
        energy_args = {'mag_locs': mag_locs, 'height' : height, 'frequency' : frequency},
        energy_barrier_loc = np.zeros(4),
        distance_threshold = distance_threshold,
        visible_dimensions = visible_dimensions
        )
    
    # Define Predictor
    esn_seed = 99
    input_strength = 5.
    spectral_radius = .4
    bias_strength = .5
    leaking_rate = leakage
    feature_function = features.StatesOnly()

    # Set Training
    transient_length = 25
    batch_size = 10
    accessible_drives = -1
    
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

save_loc = os.path.join(os.getcwd(), "Basin_Data")
folder = test_system + "_"
if partial_state:
    folder += "v"
    for v in visible_dimensions:
        folder += str(v)
if is_extra_name:
    folder += extra_name
save_loc = os.path.join(save_loc, folder)
save_loc = os.path.join(save_loc, folder_one + folder_letters[folder_one])
if folder_two in folder_letters.keys():
    save_loc = os.path.join(save_loc, folder_two + folder_letters[folder_two])

if not os.path.isdir(save_loc):
    os.makedirs(save_loc)
if str(train_seed) + 'closed.pickle' in os.listdir(save_loc) and skip_repeats:
    print("Data Already Exists in Folder: ", save_loc)
else:
    print("Training Starting, Folder: ", folder)
    
    # Construct the training library
    train_library = bh.get_lib(
        get_generator = get_generator,
        get_basin = get_basin,
        generator_args = train_generator_args,
        lib_size = lib_size,
        lib_length = lib_length,
        basin_check_length = basin_check_length,
        seed = train_seed,
        train_basin = train_basin,
        grid = grid_train,
        IC_rng = IC_rng,
        standardize = standardize,
        end_processing = end_processing
        )

    # Construct the test library
    val_library = bh.get_lib(
        get_generator = get_generator,
        get_basin = get_basin,
        generator_args = val_generator_args,
        lib_size = val_grid_width**2,
        lib_length = val_length,
        basin_check_length = basin_check_length,
        seed = 1000,
        train_basin = None,
        grid = grid_val,
        IC_rng = val_IC_rng,
        standardize = standardize,
        standardizer = train_library.standardizer,
        end_processing = end_processing
        )
    
    with open(join(save_loc, f'{train_seed}_train_ics.pickle'), 'wb') as tmp_file:
        pickle.dump(train_library.parameters, tmp_file)
    with open(join(save_loc, f'{train_seed}_val_ics.pickle'), 'wb') as tmp_file:
        pickle.dump(val_library.parameters, tmp_file)
    
    if mean_reg:
        num_fit = lib_length * lib_size - lib_size * (transient_length + 1)
    else:
        num_fit = 1
    break_check = True
    try_seed = 0
    while break_check:
        try:
            # Construct the predictor
            predictor = bh.batch_predictor(
                reservoir = rc.ESN(
                    input_dimension = len(visible_dimensions),
                    seed = esn_seed + (try_seed * np.random.default_rng(esn_seed * train_seed).integers(1000, 2000)),
                    size = esn_size,
                    connections = connections,
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
                    regression = regressions.batched_ridge(regularization = num_fit * regularization),
                    feature_function = feature_function,
                    batch_size = batch_size,
                    accessible_drives = accessible_drives
                    ),
                return_result = False,
                noise_amp = noise_amplitude
                )
            break_check = False
        except np.linalg.LinAlgError:
            try_seed += 1
    
    print("Final try_seed: ", try_seed)
    
    if open_loop:
        print("Predicting Open Loop")
        mapper_func = rch.drive_mapper
        predictor.predict(
            library = val_library,
            test_length = test_length,
            return_predictions = False,
            open_loop = True,
            save_predictions = save_predictions,
            save_loc = save_loc,
            safe_save = safe_save_predictions,
            file_name = str(train_seed) + 'open',
            reduce_fully = reduce_fully,
            reduce_states = reduce_states,
            metric_func = performance_metric,
            mapper_func = mapper_func,
            end_processing = end_processing
            )
        print("Done Open Loop")
    
    if closed_loop:
        print("Predicting Closed Loop")
        predictor.predict(
            library = val_library,
            test_length = test_length,
            return_predictions = False,
            save_predictions = save_predictions,
            save_loc = save_loc,
            safe_save = safe_save_predictions,
            file_name = str(train_seed) + 'closed',
            reduce_fully = reduce_fully,
            reduce_states = reduce_states,
            metric_func = performance_metric,
            end_processing = end_processing
            )
        print("Done Closed Loop")