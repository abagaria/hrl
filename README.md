# hrl
Robust deep skill chaining

# Robustly Learning Composable Options in Deep Reinforcement Learning

## Installation & Dependencies
 Users will need a mujoco license to run experiments.

In addition, users will need to install d4rl version 1.1: https://github.com/rail-berkeley/d4rl

 

## Running The Code
To train a DSC model
```python
python -m hrl --environment antmaze-umaze-v0 [--options]
```

Some noteworthy arguments are:

 - `environment`: Can take **umaze**, **medium**, or **4-room** as input
 - `seed`: Sets seed for replication of results
 - `lr_c, lr_a`: Denotes learning rates of critic and actor networks for TD3

To run model-free DSC simply remove the `--use_model` flag. 

the `hyperparams/default.csv` params is currently set to the same as as model_based for antmaze
