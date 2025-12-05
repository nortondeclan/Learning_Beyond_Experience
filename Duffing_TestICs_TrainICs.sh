#!/bin/bash
#SBATCH --mail-type=ALL

for ic_lim in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
do
	for val_ic_lim in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}
	do
		for seed in {1,2,3,4,5,6,7,8,9,10}
		do
			sbatch --job-name "vc$val_ic_lim ic$ic_lim" Get_Predictions.sh -T "duff" -P 1 -R 1 -b -1 -t 10 -i $ic_lim -l 10 -v 50 -V $val_ic_lim -d 1 -s 200 -r '1e-12' -n '1e-5' -S $seed -F "vc" -G "ic" -E "_nogen_vc_ic_s200_r1e-12_n1e-5" -N 0 -p 0 -c 10 -L 1.0 -D 0 -e 1 -g 1 -m 1
			echo "Running Size IC $ic_lim, NL $nl, Seed $seed"
		done
	done
done
