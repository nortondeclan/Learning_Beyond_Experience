#!/bin/bash
#SBATCH --mail-type=ALL

for ic_lim in {1.5,}
do
	for val_ic_lim in {1.5,}
	do
		for seed in {50,}
		do
			for size in {1000,1500,2000,2500}
			do
				for v_grid_width in {100,200,300}
				do
					sbatch --job-name "ic$ic_lim vic$val_ic_lim sd$seed" BigSim.sh -T "mag" -P 1 -R 1 -b 0 -t 25 -i $ic_lim -l 100 -v $v_grid_width -V $val_ic_lim -d 0.5 -s $size -r '1e-6' -n '1e-3' -S $seed -F "vc" -G "ic" -E "_tv_ics_tl25_s{$size}_v{$v_grid_width}_nl100_r1e-6_n1e-3"
				done
			done
		done
	done
done
