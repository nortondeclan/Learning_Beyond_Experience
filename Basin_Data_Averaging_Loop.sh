#!/bin/bash
#SBATCH -t 120:00
#SBATCH --mail-type=ALL

cd Basin_Data 
for run_label in */; do
	cd ..
	sbatch Basin_Data_Averaging.sh -r ${run_label}
	cd Basin_Data
done
