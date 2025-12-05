#!/bin/bash
#SBATCH -t 600:00

while getopts r: flag
do
	case "${flag}" in
		r) run_label=${OPTARG};;
	esac
done

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate rc_env
echo Averaging ${run_label}
python3 Basin_Data_Averaging.py --r ${run_label}
echo Averaged ${run_label}
