
# Most change for every experiment
test_type="opt"

# Not changed for speed tests. For other erase variable "$speed_type"
model_folder="models/$test_type"
test_log_folder="test_log_files/$test_type"
cycle_time=60
mult_factor=0.7

# test random model
model_name="rw_veh_difference_exp3_mf7"
python src/scripts/run_rl_test.py \
            --env_path "src/data/sumo_optimal/optimal.sumocfg" \
            --model_name "$model_name" \
            --num_cpu 1 \
            --n_scenarios 10 \
            --steps_per_epoch 120 \
            --ep_len 7200  \
            --graph_path "src/data/sensor_graph/adj_mx_opt.pkl" \
            --net_path "src/data/sumo_optimal/optimal.net.xml" \
            --pred_model_path "src/data/predictor_model/best_GNN_ep37_mae0.4023.pt" \
            --test_log_folder "$test_log_folder" \
            --seed 2 \
            --mult_factor $mult_factor \
            --cycle_time $cycle_time \
            --time_to_teleport -1 \
            --size 64 \
            --n_layers 2 \
            --gnn_size 16 \
            --gnn_n_layers 2 \


# test best error model
model_name="rw_veh_difference_exp3_mf7"
python src/scripts/run_rl_test.py \
            --env_path "src/data/sumo_optimal/optimal.sumocfg" \
            --model_name "$model_name" \
            --num_cpu 1 \
            --n_scenarios 10 \
            --steps_per_epoch 120 \
            --ep_len 7200  \
            --graph_path "src/data/sensor_graph/adj_mx_opt.pkl" \
            --net_path "src/data/sumo_optimal/optimal.net.xml" \
            --pred_model_path "src/data/predictor_model/best_GNN_ep37_mae0.4023.pt" \
            --test_log_folder "$test_log_folder" \
            --seed 2 \
            --mult_factor $mult_factor \
            --cycle_time $cycle_time \
            --time_to_teleport -1 \
            --size 64 \
            --n_layers 2 \
            --gnn_size 16 \
            --gnn_n_layers 2 \
            --ac_in_model_folder "$model_folder" \
            --ac_in_model_type "best" \

# test best waiting time model
model_name="rw_veh_difference_exp3_mf7"
python src/scripts/run_rl_test.py \
            --env_path "src/data/sumo_optimal/optimal.sumocfg" \
            --model_name "$model_name" \
            --num_cpu 1 \
            --n_scenarios 10 \
            --steps_per_epoch 120 \
            --ep_len 7200  \
            --graph_path "src/data/sensor_graph/adj_mx_opt.pkl" \
            --net_path "src/data/sumo_optimal/optimal.net.xml" \
            --pred_model_path "src/data/predictor_model/best_GNN_ep37_mae0.4023.pt" \
            --test_log_folder "$test_log_folder" \
            --seed 2 \
            --mult_factor $mult_factor \
            --cycle_time $cycle_time \
            --time_to_teleport -1 \
            --size 64 \
            --n_layers 2 \
            --gnn_size 16 \
            --gnn_n_layers 2 \
            --ac_in_model_folder "$model_folder" \
            --ac_in_model_type "best_wt"