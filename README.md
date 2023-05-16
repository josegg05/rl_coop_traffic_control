## rl_coop_traffic_control
RL Distributed Control Scheme for Cooperative Intersection Traffic Control

# Prerequisites
- Install SUMO:
sudo apt-get install sumo sumo-tools sumo-doc

- Install and activate conda environment:
conda env create -f environment.yml
conda activate rl_coop_tc


# Run test
sh src/scripts/run_rl_tests.sh

# NOTES:
- Windows: change "forkserver" for "spawn" in src/environment/vec_env.py > line 62.


