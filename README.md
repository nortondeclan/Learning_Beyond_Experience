# Learning beyond experience: Generalizing to unseen state space with reservoir computing
Code to predict the basins of attraction of multistable dynamical systems using reservoir computing, to accompany [*D. A. Norton, Y. Zhang, and M. Girvan, “Learning beyond experience: Generalizing to unseen state space with reservoir computing,” Chaos 35 103146 (2025)*](https://doi.org/10.1063/5.0283421). 

Open-access preprint available at [*https://arxiv.org/abs/2506.05292*](https://arxiv.org/abs/2506.05292).

Built on the *rescompy* python module for reservoir computing, [*D. Canaday, D. Kalra, A. Wikner, D. A. Norton, B. Hunt, and A. Pomerance, “rescompy 1.0.0: Fundamental Methods for Reservoir Computing in
Python,” GitHub (2024)*](https://github.com/PotomacResearch/rescompy).

[*dysts*](https://github.com/GilpinLab/dysts) package also used to estimate Kullback-Leibler divergences.

![Duffing_Full_Nw10_S75_NL10_TLen500_R1e-8_N1e-5_Example_Horizontal](https://github.com/user-attachments/assets/e04c87e1-4a81-414d-8e99-342d8edd623d)

**RC identification of an unseen attractor in the Duffing system.** **(a)** We train an RC with $N_r=75$ nodes on $N_{train}=10$ fully-observed trajectories (gray lines) from one of the Duffing system’s basins of attraction. **(b)** Then, we forecast the system’s evolution from $36$ initial conditions (dots) and corresponding short test signals (thick lines, $N_{test}=10$ observations). Predictions illustrated by thin lines. We color each trajectory according the true basin of its initial conditions. The RC recovers system behavior in both the seen (blue) and unseen (pink) basins. Only one of the sample predictions (a pink trajectory) goes to the incorrect fixed point.
