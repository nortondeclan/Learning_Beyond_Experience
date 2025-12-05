#!/bin/bash
#SBATCH --mail-type=ALL

for ic_lim in {.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0,1.1,1.2,1.3,1.4,1.5}
do
	for val_ic_lim in {.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0}
	do
		for seed in {1,2,3,4,5,6,7,8,9,10}
		do
			sbatch --job-name "vc$val_ic_lim ic$ic_lim" Get_Predictions.sh -T "mag" -P 1 -R 1 -b 0 -t 100 -i $ic_lim -l 100 -v 50 -V $val_ic_lim -d 0.5 -s 2500 -r '1e-10' -n '1e-3' -S $seed -F "vc" -G "ic" -E "_vc_ic_s2500_r1e-10_n1e-3" -N 0 -p 0 -c 10 -L 1.0 -D 0 -e 1 -g 1 -m 1
			echo "Running Size IC $ic_lim, NL $nl, Seed $seed"
		done
	done
done
