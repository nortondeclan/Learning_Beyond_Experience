#!/bin/bash
#SBATCH --mail-type=ALL

for ic_lim in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
do
	for nl in {1,2,3,4,5,8,10,15,20,30,40,50,60,70,80,90,100,200,300,400,500}
	do
		for seed in {1,2,3,4,5,6,7,8,9,10}
		do
			sbatch --job-name "nl$nl ic$ic_lim" Get_Predictions.sh -T "duff" -P 1 -R 1 -b 0 -t 10 -i $ic_lim -l $nl -v 50 -V 10 -d 1.0 -s 200 -r '1e-12' -n '1e-5' -S $seed -F "nl" -G "ic" -E "_V10_nl_ic_s200_r1e-12_n1e-5" -N 0 -p 0 -c 10 -L 1.0 -D 0 -e 1 -g 1 -m 1
			echo "Running Size IC $ic_lim, NL $nl, Seed $seed"
		done
	done
done
