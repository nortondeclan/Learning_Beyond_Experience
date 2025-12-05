#!/bin/bash
#SBATCH --mail-type=ALL

for ic_lim in {1.5,}
do
	for val_ic_lim in {1.5,}
	do
		for seed in {50,}
		do
			for size in {2500,}
			do
				for test_length in {25,50,100}
				do
					sbatch --job-name "tl$test_length" BigSim.sh -T "mag" -P 1 -R 1 -b 0 -t $test_length -i $ic_lim -l 100 -v 300 -V $val_ic_lim -d 0.5 -s $size -r '1e-10' -n '1e-3' -S $seed -F "vc" -G "ic" -E "_bigsim300_tl{$test_length}_s{$size}_c10" -N 0 -p 0 -c 10 -L 1.0 -D 0 -e 1 -g 1 -m 1
					echo "Running Test Length $test_length"
				done
			done
		done
	done
done
