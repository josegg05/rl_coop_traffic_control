# rl_coop_traffic_control
RL Distributed Control Scheme for Cooperative Intersection Traffic Control

## Prerequisites
### Install SUMO:
```bash
sudo apt-get install sumo sumo-tools sumo-doc
```

### Install and activate conda environment:
```bash
conda env create -f environment.yml
conda activate rl_coop_tc
```


## Run test
```bash
sh src/scripts/run_rl_tests.sh
```

## Create tests ".csv". Especially useful when you have more than one model tested on each test log folder.
```bash
python test_log_files/create_tables_m_samples.py
```

## NOTES:
- Windows: change "forkserver" to "spawn" in "src/environment/vec_env.py > line 62".


