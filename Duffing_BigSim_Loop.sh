#!/bin/bash
#SBATCH --mail-type=ALL

for ic_lim in {4,7,10}
do
	for val_ic_lim in {10,}
	do
		for seed in {50,}
		do
			for size in {200,}
			do
				for test_length in {10,}
				do
					sbatch --job-name "ic$ic_lim s$size" BigSim.sh -T "duff" -P 1 -R 1 -b 0 -t $test_length -i $ic_lim -l 10 -v 150 -V $val_ic_lim -d 0.5 -s $size -r '1e-12' -n '1e-5' -S $seed -F "vc" -G "ic" -E "_bigsim_duff150_ic{$ic_lim}_s{$size}_c10" -N 0 -p 0 -c 10 -L 1.0 -D 0 -e 1 -g 1 -m 1
					echo "Running ic$ic_lim s$size"
				done
			done
		done
	done
done
