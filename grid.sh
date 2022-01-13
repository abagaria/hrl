#$-cwd
#$-l long
#$-l gpus=1
#$-l vf=20G
#$-l vlong
#$-pe smp 4
#$-e monte_dsg_rnd_sticky_actions.err
#$-o monte_dsg_rnd_sticky_actions.out
#$-t 1-5
#$-tc 5
. /home/abagaria/.bashrc
. /home/abagaria/miniconda3/bin/activate dsgenv
python -um hrl.agent.dsg.train --experiment_name=dsg_test_gc_policy_while_running_rnd_sticky_actions_grid --environment_name=MontezumaRevengeNoFrameskip-v4 --gpu_id=0 --seed=0 --gestation_period=10 --num_training_steps=2000000 --max_frames_per_episode=8000 --replay_original_goal_on_pos --distance_metric=ucb --use_pos_for_init
