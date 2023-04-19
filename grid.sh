#$-cwd
#$-l gpus=1
#$-l vf=20G
#$-l vlong
#$-t 1-5
#$-tc 5
#$-e gridlogs/gridtest.err
#$-o gridlogs/gridtest.out
. /home/npermpre/.bashrc
. /home/npermpre/miniconda3/bin/activate dsgenv
python -um hrl.agent.dsg.train --experiment_name=dsg_gridtest --environment_name=MontezumaRevengeNoFrameskip-v4 --gpu_id=0 --seed=$SGE_TASK_ID --gestation_period=10 --num_training_steps=2000000 --max_frames_per_episode=8000 --replay_original_goal_on_pos --distance_metric=ucb --use_pos_for_init
