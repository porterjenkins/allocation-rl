# Optimal Product Allocation - PSD-sim
Simulation and optimization techniques to solve the optimal allocation problem


## Config File

The project uses a config file (called config.json) to do basic setup. Currently, the config file must be placed in the `config` directory and should have the following parameters:


```
"n_products": 15,
"max_t": 25,
"prices": [2.50, 4.99, 3.00, 1.20],
"n_temporal_features": 7,
"env_init_day": 0,
"adj_mtx_fname": "data/store-1-adj-mtx.json",
"train_data": "data/store-1-train.csv",
"test_data": "data/store-1-test.csv",
"model_type": "hierarchical",
"prior_fname": "envs/prior.json",
"precision_mtx": true,
"cost": 50.0,
"log_linear": false,
"episode_len": 60
```
