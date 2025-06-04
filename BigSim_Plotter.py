import numpy as np
import os
import basins_helpers as bh
import test_systems as tst
import matplotlib as mpl
import pickle

v_width = 300
rc_size = 2500
test_length = 100 #50 #25

run_directory = f'MP_Big_Sim_tl{test_length}_v{v_width}_s{rc_size}'
run_label = "mag_v01_tv_ics_" + f"tl{test_length}_" + "s{" + f"{rc_size}" + "}_v{" + f"{v_width}" + "}_nl100_r1e-6_n1e-3"
test_dir = f"MP_TLConv_NoStand_pTrue_v{v_width}_tl{test_length}"

# Library Parameters
visible_dimensions = [0, 1]
lib_seed = 50
val_lib_seed = 101
grid_val = True
train_basin = 0
test_system = "magnetic_pendulum"
#test_system = "duffing"
rotation = 0
use_energies = True

if test_system == "duffing":
    a, b, c = -0.5, -1., 0.1
    
    distance_threshold = .5
    
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
    
    basin_alpha = 0.75
       
elif test_system == "magnetic_pendulum":
    height = .2
    frequency = .5
    damping = .2
    
    mag_locs = 1./np.sqrt(3.) * np.array([
        [np.cos(rotation), np.sin(rotation)],
        [np.cos(rotation + 2*np.pi/3.), np.sin(rotation + 2*np.pi/3.)],
        [np.cos(rotation + 4*np.pi/3.), np.sin(rotation + 4*np.pi/3.)]
        ])
    fixed_pts = mag_locs
    
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
        
    distance_threshold = .25
        
    get_basin = bh.get_attractor(
        fixed_points = mag_locs,
        use_energies = bool(len(visible_dimensions) == 4 and use_energies),
        energy_func = tst.mp_energy,
        energy_args = {'mag_locs': mag_locs, 'height' : height, 'frequency' : frequency},
        energy_barrier_loc = np.zeros(4),
        distance_threshold = distance_threshold,
        visible_dimensions = visible_dimensions
        )
    
    simple_get_basin = bh.get_attractor(
        fixed_points = mag_locs,
        use_energies = bool(len(visible_dimensions) == 4 and use_energies),
        energy_func = tst.mp_energy,
        energy_args = {'mag_locs': mag_locs, 'height' : height, 'frequency' : frequency},
        energy_barrier_loc = np.zeros(4),
        visible_dimensions = visible_dimensions
        )
    
    colors_map = ["#ff006e", "#3a86ff", "#ffbe0b"]
    colormap = mpl.colors.LinearSegmentedColormap.from_list("custom", colors_map, N = len(colors_map))
    colormap.set_under("#00b48a")
    colormap.set_bad("#00b48a")
    colormap.set_over("white")
    basin_alpha = 1


def join(directory, name):
    return os.path.join(directory, name)

read_loc = join(join(os.getcwd(), "Big_Sim_Data"), run_directory)
test_read_loc = join(join(os.getcwd(), "Big_Sim_Data"), test_dir)

try:
    with open(join(test_read_loc, 'val_ics.pickle'), 'rb') as tmp_file:
        tval_ics = list(np.array(pickle.load(tmp_file)))
        
    with open(join(test_read_loc, 'val_testends.pickle'), 'rb') as tmp_file:
        tval_testends = list(np.array(pickle.load(tmp_file)))
    have_tlen_convs = True
except:
    have_tlen_convs = False


folder_1 = join(read_loc, run_label)
f1s = os.listdir(folder_1)
if 'train_ics.pickle' in os.listdir(read_loc):
    with open(join(read_loc, 'train_ics.pickle'), 'rb') as tmp_file:
        training_ics = list(np.array(pickle.load(tmp_file)))

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
letters_3 = ''.join(c for c in f3s[0] if not c.isdigit())
seeds = np.sort(np.array([int(c.replace(letters_3, "")) for c in f3s]))
loop = letters_3.replace('.pickle', '')

for f1i, f1 in enumerate(lvalues_1):
    for f2i, f2 in enumerate(lvalues_2):
        for si, seed in enumerate(seeds):
            pred_dir = join(join(join(
                            folder_1,
                            letters_1 + f1),
                            letters_2 + f2),
                            str(seed) + loop + ".pickle")
            with open(pred_dir, "rb") as tmp_file:
                predictions = pickle.load(tmp_file)
                
            test_ics = [p.initial_pts for p in predictions]
            predicted_finals = [p.predicted_finals for p in predictions]
            true_finals = [p.true_finals for p in predictions]

#if len(visible_dimensions) == 1:
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
    # Plot the true basins
    bh.basin_heatmap(
        initial_pts = test_ics,
        final_pts = true_finals,
        overlay_pts = training_ics,
        overlay_size = 10., 
        fixed_points = fixed_pts,
        get_fixed_pt = get_basin,
        colormap = colormap,
        font_size = 20,
        transparency = True,
        equal_aspect = True,
        num_ticks = 5, #Set to zero to remove ticks for Fig. 8
        overlay_marker = "o",
        box_linestyle = "--",
        basin_alpha = basin_alpha,
        fp_linewidth = 3,
        label_axes = True #Set to False to remove labels for Fig. 8
        )
    
    if have_tlen_convs:
        try:
            simple_correct = len(np.nonzero([
                simple_get_basin(tval_testends[ind]) == get_basin(true_finals[ind])
                for ind in range(len(predictions))
                ])[0]) / len(predictions)
            ic_correct = len(np.nonzero([
                simple_get_basin(tval_ics[ind]) == get_basin(true_finals[ind])
                for ind in range(len(predictions))
                ])[0]) / len(predictions)
            
            # Plot predicted basins
            bh.basin_heatmap(
                initial_pts = tval_ics,
                final_pts = tval_testends,
                fixed_points = fixed_pts,
                get_fixed_pt = simple_get_basin,
                skip_heatmap = False,
                colormap = colormap,
                font_size = 20,
                transparency = True,
                equal_aspect = True,
                num_ticks = 0,
                overlay_marker = "o",
                box_linestyle = "--",
                basin_alpha = basin_alpha,
                fp_linewidth = 3,
                label_axes = False,
                xlims = (-1.5, 1.5),
                ylims = (-1.5, 1.5)
                )  
            for overlay_pts in [tval_testends]: #, tval_ics]:
                for overlay_colors in [
                        [colormap(simple_get_basin(test_end)) for test_end in tval_testends],
                        [colormap(get_basin(true_final)) for true_final in true_finals],
                        [colormap(get_basin(pred_final)) for pred_final in predicted_finals]
                        ]:
                    bh.basin_heatmap(
                        initial_pts = tval_ics,
                        final_pts = tval_testends,
                        overlay_pts = overlay_pts,
                        overlay_size = .1,
                        fixed_points = fixed_pts,
                        get_fixed_pt = simple_get_basin,
                        overlay_colors = overlay_colors,
                        skip_heatmap = True,
                        colormap = colormap,
                        font_size = 20,
                        transparency = True,
                        equal_aspect = True,
                        num_ticks = 0,
                        overlay_marker = "o",
                        box_linestyle = "--",
                        basin_alpha = basin_alpha,
                        fp_linewidth = 3,
                        label_axes = False,
                        xlims = (-1.5, 1.5),
                        ylims = (-1.5, 1.5)
                        )
        except:
            pass
    
    # Plot the RC-predicted basins
    bh.basin_heatmap(
        initial_pts = test_ics,
        final_pts = predicted_finals,
        overlay_pts = training_ics,
        fixed_points = fixed_pts,
        get_fixed_pt = get_basin,
        colormap = colormap,
        overlay_marker = "o",
        overlay_color = "k",
        overlay_size = 10., 
        font_size = 20,
        transparency = True,
        equal_aspect = True,
        num_ticks = 5, #Set to zero to remove ticks for Fig. 8
        box_linestyle = "--",
        basin_alpha = basin_alpha,
        fp_linewidth = 3,
        label_axes = True, #Set to False to remove labels for Fig. 8
        xlims = (-1.5, 1.5),
        ylims = (-1.5, 1.5)        
        )
    
    # Color incorrectly predicted basins white
    bh.basin_error_heatmap(
        fixed_points = fixed_pts,
        initial_pts = test_ics,
        predicted_finals = predicted_finals,
        true_finals = true_finals,
        colormap = colormap,
        get_fixed_pt = get_basin,
        overlay_pts = training_ics,
        overlay_marker = "o",
        overlay_color = "k",
        overlay_size = 10., 
        font_size = 20,
        transparency = True,
        equal_aspect = True,
        num_ticks = 5,
        box_linestyle = "--",
        box_alpha = .75,
        basin_alpha = basin_alpha,
        fp_linewidth = 3
        )

t_basins = [get_basin(p.true_finals) for p in predictions]
p_basins = [get_basin(p.predicted_finals) for p in predictions]
bve, bve_per_basin, f_unphys, bve_thresh = bh.basin_volume_error(true_basins = t_basins, pred_basins = p_basins)
print("Basin Volume Error: ", bve)
print("Basin Volume Error Generization Guarantee: ", bve_thresh)
print("Fraction Unphysical: ", f_unphys)