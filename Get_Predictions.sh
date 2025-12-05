#!/bin/bash
#SBATCH -t 900:00

while getopts T:P:R:b:t:i:l:v:V:d:s:r:n:S:F:G:E:N:p:c:L:D:e:g:m: flag
do
	case "${flag}" in
		T) test_system=${OPTARG};;
		P) partial_state=${OPTARG};;
		R) reduce_fully=${OPTARG};;
		b) train_basin=${OPTARG};;
		t) test_length=${OPTARG};;
		i) IC_lim=${OPTARG};;
		l) lib_size=${OPTARG};;
		v) val_width=${OPTARG};;
		V) val_IC_lim=${OPTARG};;
		d) dist_thresh=${OPTARG};;
		s) rc_size=${OPTARG};;
		r) reg=${OPTARG};;
		n) noise=${OPTARG};;
		S) seed=${OPTARG};;
		F) fone=${OPTARG};;
		G) ftwo=${OPTARG};;
		E) extra_name=${OPTARG};;
		N) process_noise=${OPTARG};;
		p) val_process_noise=${OPTARG};;
		c) connections=${OPTARG};;
		L) leakage=${OPTARG};;
		D) integrator_transient=${OPTARG};;
		e) rc_time_step=${OPTARG};;
		g) grid_val=${OPTARG};;
		m) mean_reg=${OPTARG};;
	esac
done

source /home/user/miniconda3/etc/profile.d/conda.sh #Activate conda from appropriate directory location
conda activate rc_env
python3 -u Get_Predictions.py --test_system $test_system --partial_state $partial_state --reduce_fully $reduce_fully --train_basin $train_basin --test_length $test_length --IC_lim $IC_lim --lib_size $lib_size --val_grid_width $val_width --distance_threshold $dist_thresh --esn_size $rc_size --regularization $reg --noise_amplitude $noise --train_seed $seed --folder_one $fone --folder_two $ftwo --extra_name $extra_name --val_IC_lim $val_IC_lim --process_noise $process_noise --val_process_noise $val_process_noise --rc_time_step $rc_time_step --leakage $leakage --connections $connections --integrator_transient $integrator_transient --grid_val $grid_val --mean_reg $mean_reg
