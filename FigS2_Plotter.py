import numpy as np
import os
import test_systems as tst
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import basins_helpers as bh

run_label = "duff_v0_V10_size_pn_r1e-12_n1e-5"

train_basin = 0
partial_state = True
font_size = 20.

join = os.path.join

read_loc = os.path.join(os.getcwd(), "Basin_Data")

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

a, b, c = -0.5, -1., 0.1
distance_threshold = 0.5

if partial_state:
    visible_dimensions = [0]
else:
    visible_dimensions = [0, 1]

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

colors_map = ["#70d6ff", "#ff70a6"]
colormap = mpl.colors.LinearSegmentedColormap.from_list("custom", colors_map, N = len(colors_map))
colormap.set_bad("white")
colormap.set_under("white")
colormap.set_over("#e9ff70")
basin_alpha = .75

plot1 = lvalues_1 #np.array(['200']) 
plot2 = lvalues_2

if plot1 is not None and plot2 is not None:
    with mpl.rc_context({'font.size': font_size}):  
        if plot1 is not None and plot2 is not None:
            bfig, bax = plt.subplots(nrows = len(plot1), ncols = len(plot2),
                                     figsize = (4 * len(plot2) , 4 * len(plot1)),
                                     constrained_layout = True,
                                     sharex = True, sharey = True)
            if len(bax.shape) == 1:
                bax = bax.reshape((1, -1))

for f1i, f1 in enumerate(lvalues_1):
    for f2i, f2 in enumerate(lvalues_2):
        for si, seed in enumerate(seeds):
            data_dir = join(join(
                            folder_1,
                            letters_1 + f1),
                            letters_2 + f2)
            pred_dir = join(data_dir, str(seed) + loop + ".pickle")
            with open(pred_dir, "rb") as tmp_file:
                basin_predictions = pickle.load(tmp_file)
                
                if (f1 in plot1 and f2 in plot2) and seed == 1:
                    
                    with open(join(data_dir, str(seed) + '_val_ics.pickle'), 'rb') as tmp_file:
                        tval_ics = list(np.array(pickle.load(tmp_file)))
                    with open(join(data_dir, str(seed) + '_train_ics.pickle'), 'rb') as tmp_file:
                        train_ics = list(np.array(pickle.load(tmp_file)))
                    
                    print("found " + f2)
                    
                    with mpl.rc_context({'font.size': font_size}):
                        
                        bh.basin_error_heatmap(
                            fixed_points = fixed_pts,
                            initial_pts = tval_ics,
                            predicted_finals = [p.predicted_finals for p in basin_predictions],
                            true_finals = [p.true_finals for p in basin_predictions],
                            colormap = colormap,
                            get_fixed_pt = get_basin,
                            overlay_pts = train_ics,
                            overlay_marker = "o",
                            overlay_color = "k",
                            font_size = 20,
                            transparency = False,
                            equal_aspect = True,
                            num_ticks = 5,
                            box_linestyle = "--",
                            box_alpha = .75,
                            basin_alpha = basin_alpha,
                            fp_linewidth = 3,
                            ax = bax[f1i, f2i],
                            xlabel = "$x_0$" if f1i == len(plot1) - 1 else None,
                            ylabel = "$y_0$" if f2i == 0 else None,
                            )
                        bfig.patch.set_alpha(0)