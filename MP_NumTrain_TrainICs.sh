#!/bin/bash
#SBATCH --mail-type=ALL

for ic_lim in {.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0,1.1,1.2,1.3,1.4,1.5}
do
	for nl in {1,2,3,4,5,8,10,15,20,30,40,50,60,70,80,90,100,125,150,175,200,250,300,400,500}
	do
		for seed in {1,2,3,4,5,6,7,8,9,10}
		do
			sbatch --job-name "nl$nl ic$ic_lim" Get_Predictions.sh -T "mag" -P 1 -R 1 -b -1 -t 100 -i $ic_lim -l $nl -v 50 -V 1.5 -d 0.5 -s 2500 -r '1e-10' -n '1e-3' -S $seed -F "nl" -G "ic" -E "_nogen_V1.5_nl_ic_s2500_r1e-10_n1e-3" -N 0 -p 0 -c 10 -L 1.0 -D 0 -e 1 -g 1 -m 1
			echo "Running Size IC $ic_lim, NL $nl, Seed $seed"
		done
	done
done
