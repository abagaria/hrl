# hrl
Robust deep skill chaining

# Robustly Learning Composable Options in Deep Reinforcement Learning

## Installation & Dependencies

Our code builds on top of the simple_rl library. We recommend using Anaconda and creating an environment from the **environment.yml** file (which includes all dependencies and their versions). Users will need a mujoco license to run experiments.

In addition, users will need to install d4rl version 1.1: https://github.com/rail-berkeley/d4rl

 

## Running The Code

With an iPython terminal running in this directory, the following command is an example of how run our DSC++ experiments:

    python -u simple_rl/agents/func_approx/dsc/experiments/online_mbrl_skill_chaining.py --experiment_name='experiment-name' --device="cuda" --environment="4-room"  --gestation_period=10 --episodes=2001 --steps=1000 --warmup_episodes=50 --use_value_function --use_global_value_function --use_model --use_diverse_starts --logging_frequency=2000 --seed=0 --lr_c=1e-5 --lr_a=1e-5 --evaluation_frequency=10 --buffer_length=50

Some noteworthy arguments are:

 - `environment`: Can take **umaze**, **medium**, or **4-room** as input
 - `seed`: Sets seed for replication of results
 - `lr_c, lr_a`: Denotes learning rates of critic and actor networks for TD3

To run model-free DSC simply remove the `--use_model` flag. 
