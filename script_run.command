#!/bin/bash
#$-cwd
#$-l gpus=1
#$-l vf=20G
export PATH=~/miniconda3/bin:$PATH
conda activate dsg
export PYTHONPATH="${PYTHONPATH}:/home/npermpre/miniconda3/envs/dsg/lib/python3.9/site-packages"
python3 -um hrl.agent.dsg.train --experiment_name=noOptionFilitering_keyDoorRoomConds_CNNInit_ImaginaryLUT_noChainLastOption_NoTrajSegment_NewEventCloseness_InventoryStr --environment_name=MontezumaRevengeNoFrameskip-v4 --gpu_id=0 --seed=18 --gestation_period=10 --num_training_steps=20000000 --max_frames_per_episode=4000 --replay_original_goal_on_pos --distance_metric=empirical --goal_selection_criterion=boltzmann_unconnected --enable_rnd_logging --initiation_classifier_type="cnn" --use_full_negative_trajectory --n_expansion_episodes=20 --n_warmup_iterations=2 --goal_selection_epsilon=0.35 --min_n_points_for_expansion=10 --reject_jumping_states --use_empirical_distances --expansion_fraction_threshold=0.5 --create_sparse_graph
