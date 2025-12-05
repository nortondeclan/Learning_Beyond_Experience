import numpy as np
import os
import basins_helpers as bh
import climate_helpers as climate
import test_systems as tst
import pickle
from typing import Union, List
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument(
    '-r', '--run_label', type = str, required = True,
    )
run_label = parser.parse_args().run_label

rotation = 0

if "duff" in run_label:
    test_system = "duff"
    if "v0" in run_label:
        partial_state = True
    else:
        partial_state = False
elif "mag" in run_label:
    test_system = "mag"
    if "v01" in run_label:
        partial_state = True
    else:
        partial_state = False

if test_system == "duff":
    utility = 0.5
    distance_threshold = 0.5
elif test_system == "mag":
    utility = 1./3.
    distance_threshold = 0.25
elif test_system == "mlor":
    utility = 0.5
    distance_threshold = 1.0
    partial_state = True

def join(directory, name):
    return os.path.join(directory, name)

read_loc = join(os.getcwd(), "Basin_Data")
save_loc = join(os.getcwd(), "Averaged_Data")

folder_1_save = join(save_loc, run_label)

if os.path.isdir(folder_1_save):
    shutil.rmtree(folder_1_save)
os.makedirs(folder_1_save)

folder_1 = join(read_loc, run_label)
f1s = os.listdir(folder_1)
if 'accuracies.pickle' in f1s:
    f1s.remove('accuracies.pickle')
if 'metrics.pickle' in f1s:
    f1s.remove('metrics.pickle')
letters_1 = ''.join(c for c in f1s[0][:2] if not c.isdigit())
lvalues_1 = np.array([c.replace(letters_1, "") for c in f1s])
argsort = np.argsort([float(entry) for entry in lvalues_1])
values_1 = np.sort(np.array([float(c.replace(letters_1, "")) for c in f1s]))
lvalues_1 = list(lvalues_1[argsort])

folder_2 = join(folder_1, f1s[0])
f2s = os.listdir(folder_2)
letters_2 = ''.join(c for c in f2s[0][:2] if not c.isdigit())
lvalues_2 = np.array([c.replace(letters_2, "") for c in f2s])
argsort = np.argsort([float(entry) for entry in lvalues_2])
values_2 = np.sort(np.array([float(c.replace(letters_2, "")) for c in f2s]))
lvalues_2 = list(lvalues_2[argsort])

folder_3 = join(folder_2, f2s[0])
f3s = os.listdir(folder_3)
f3sb = []
for entry in f3s:
    if 'train_ics' not in entry and 'val_ics' not in entry:
        f3sb.append(entry)
        f3s = f3sb
letters_3 = ''.join(c for c in f3s[0] if not c.isdigit())
seeds = np.sort(np.array([int(c.replace(letters_3, "")) for c in f3s]))
loop = letters_3.replace('.pickle', '')

if test_system == "duff":
    a, b, c = -0.5, -1., 0.1
    
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
    else:
        energy_func = None
        kinetic_func = None
        potential_func = None
        energy_args = None

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
       
elif test_system == "mag":
    height = .2
    frequency = .5
    damping = .2
    
    print("Partial State: ", partial_state)
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
    else:
        energy_func = None
        kinetic_func = None
        potential_func = None
        energy_args = None
        
    get_basin = bh.get_attractor(
        fixed_points = mag_locs,
        use_energies = bool(len(visible_dimensions) == 4),
        energy_func = tst.mp_energy,
        energy_args = {'mag_locs': mag_locs, 'height' : height, 'frequency' : frequency},
        energy_barrier_loc = np.zeros(4),
        distance_threshold = distance_threshold,
        visible_dimensions = visible_dimensions
        )

vlength_threshold = 1.
basin_ids = list(np.arange(len(fixed_pts))) +  ["unstable", "all"]
accuracies = {
    basin_id :
    np.zeros((len(values_1), len(values_2), len(seeds)))
    for basin_id in basin_ids
    }
metrics = {
    basin_id :
    np.zeros((len(values_1), len(values_2), len(seeds)))
    for basin_id in basin_ids
    }
metrics["stable"] = np.zeros((len(values_1), len(values_2), len(seeds)))
basin_volumes = {
    basin_id :
    np.zeros((len(values_1), len(values_2), len(seeds)))
    for basin_id in basin_ids
    }

try:
    with open(join(folder_1, 'accuracies.pickle'), 'rb') as tmp_file:
        accuracies = pickle.load(tmp_file)
    with open(join(folder_1, 'metrics.pickle'), 'rb') as tmp_file:
        metrics = pickle.load(tmp_file)
    with open(join(folder_1, 'basin_volumes.pickle'), 'rb') as tmp_file:
        basin_volumes = pickle.load(tmp_file)
        
except:
    for f1i, f1 in enumerate(lvalues_1):
        for f2i, f2 in enumerate(lvalues_2):
            for si, seed in enumerate(seeds):
                pred_dir = join(join(join(
                                folder_1,
                                letters_1 + f1),
                                letters_2 + f2),
                                str(seed) + loop + ".pickle")
                with open(pred_dir, "rb") as tmp_file:
                    basin_predictions = pickle.load(tmp_file)
                sorted_predictions = bh.sort_by_basin(basin_predictions, get_basin = get_basin)

                bve = bh.basin_volume_error(basin_predictions, get_basin = get_basin)
                for basin_id in basin_ids:
                    if basin_id == "all":
                        fraction_correct, fraction_unstable = bh.get_prediction_stats(
                                basin_predictions = basin_predictions,
                                get_basin = get_basin
                                )
                        accuracies["all"][f1i, f2i, si] = fraction_correct
                        accuracies["unstable"][f1i, f2i, si] = fraction_unstable
                        basin_volumes["all"][f1i, f2i, si] = bve[0]
                        if basin_predictions[0].performance_metric is not None:
                            metrics["all"][f1i, f2i, si] = np.mean([
                                p.performance_metric for p in basin_predictions
                                ])
                    elif basin_id == "unstable":
                        predicted_nearest = np.array(
                            [get_basin(pred.predicted_finals) for pred in basin_predictions]
                            )
                        stable_ids = np.argwhere(np.isnan(predicted_nearest))
                        unstable_ids = np.argwhere(np.isnan(predicted_nearest))
                        basin_volumes["unstable"][f1i, f2i, si] = bve[2]
                        if basin_predictions[0].performance_metric is not None:
                            if len(stable_ids) > 0:
                                metrics["stable"][f1i, f2i, si] = np.mean([
                                    p.performance_metric for p in np.array(basin_predictions)[stable_ids][0]
                                    ])
                            else:
                                metrics["stable"][f1i, f2i, si] = np.nan
                            if len(unstable_ids) > 0:
                                metrics["unstable"][f1i, f2i, si] = np.mean([
                                    p.performance_metric for p in np.array(basin_predictions)[unstable_ids][0]
                                    ])
                            else:
                                metrics["unstable"][f1i, f2i, si] = np.nan
                    else:
                        basin_volumes[basin_id][f1i, f2i, si] = bve[1][basin_id]
                        try:
                            fraction_correct, _ = bh.get_prediction_stats(
                                    basin_predictions = sorted_predictions[basin_id],
                                    get_basin = get_basin
                                    )
                            accuracies[basin_id][f1i, f2i, si] = fraction_correct
                            if basin_predictions[0].performance_metric is not None:
                                metrics[basin_id][f1i, f2i, si] = np.mean([
                                    p.performance_metric for p in sorted_predictions[basin_id]
                                    ])
                        except KeyError:
                            accuracies[basin_id][f1i, f2i, si] = 0
                            if basin_predictions[0].performance_metric is not None:
                                metrics[basin_id][f1i, f2i, si] = np.nan
    
    basin_volumes["all_attractors_predicted"] = bve[3]

    with open(join(folder_1_save, 'accuracies.pickle'), 'wb') as tmp_file:
        pickle.dump(accuracies, tmp_file)
    with open(join(folder_1_save, 'metrics.pickle'), 'wb') as tmp_file:
        pickle.dump(metrics, tmp_file)
    with open(join(folder_1_save, 'basin_volumes.pickle'), 'wb') as tmp_file:
        pickle.dump(basin_volumes, tmp_file)
    with open(join(folder_1_save, 'letters_1.pickle'), 'wb') as tmp_file:
        pickle.dump(letters_1, tmp_file)
    with open(join(folder_1_save, 'letters_2.pickle'), 'wb') as tmp_file:
        pickle.dump(letters_2, tmp_file)
    with open(join(folder_1_save, 'letters_3.pickle'), 'wb') as tmp_file:
        pickle.dump(letters_3, tmp_file)
    with open(join(folder_1_save, 'values_1.pickle'), 'wb') as tmp_file:
        pickle.dump(values_1, tmp_file)
    with open(join(folder_1_save, 'values_2.pickle'), 'wb') as tmp_file:
        pickle.dump(values_2, tmp_file)
    with open(join(folder_1_save, 'seeds.pickle'), 'wb') as tmp_file:
        pickle.dump(seeds, tmp_file)