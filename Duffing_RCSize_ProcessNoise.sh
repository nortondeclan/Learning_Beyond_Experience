#!/bin/bash
#SBATCH --mail-type=ALL

for train_pn in {0.1,0.5,1.0,2.0}
do
	for size in {200,}
	do
		for seed in {1,}
		do
			sbatch --job-name "s$size pn$train_pn" Get_Predictions.sh -T "duff" -P 1 -R 1 -b 0 -t 10 -i 10 -l 10 -v 150 -V 10 -d .5 -s $size -r '1e-12' -n '1e-5' -S $seed -F "s" -G "pn" -E "_V10_size_pn_r1e-12_n1e-5" -N $train_pn -p 0 -c 10 -L 1.0 -D 0 -e 1 -g 1 -m 1
			echo "Running Size PN $train_pn, NL $nl, Seed $seed"
		done
	done
done
