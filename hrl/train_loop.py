import os
import csv
import pickle
import logging
import traceback
from pathlib import Path

import numpy as np

from hrl.envs.vector_env import EpisodicSyncVectorEnv
from hrl import utils
from hrl.utils import filter_token
from hrl.utils import StopExecution


def train_agent_batch_with_eval(
    agent,
    env,
    num_episodes,
    test_env=None,
    num_test_episodes=None,
    goal_conditioned=False,
    goal_state=None,
    logging_freq=None,
    testing_freq=None,
    plotting_freq=None,
    saving_freq=None,
    saving_dir=None,
    state_to_goal_fn=None,
    reward_fn=None,
):
    """Train an agent in a batch environment.

    Args:
        agent: Agent to train.
        env: Environment to train the agent against.
        num_episodes (int): number of episodes to train the agent
        test_env: envrionment to test the agent against, this should have a different random seed than env
        params (dict): training parameters
        max_episode_len (int): max steps in one episode 
        state_to_goal_fn: a function to extract the goal from an observation state
    Returns:
        List of evaluation episode stats dict.
    """

    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)

    assert isinstance(env, EpisodicSyncVectorEnv)

    # main training loops
    try:
        # for each episode
        for episode in range(num_episodes):
            # episode rollout
            trajectory, episode_r, episode_len, episode_start_poss, episode_reached_pos = episode_rollout(
                testing=False,
                episode_idx=episode,
                env=env,
                agent=agent,
                goal_conditoned=goal_conditioned,
                goal_state=goal_state,
                logger=logger,
                logging_freq=logging_freq,
                state_to_goal_fn=state_to_goal_fn,
            )
            assert len(episode_r) == len(episode_len) == len(episode_start_poss) == len(episode_reached_pos)

            # logging the testing stats
            mode = 'w' if episode == 0 else 'a'
            training_rewards_file = os.path.join(saving_dir, 'training_rewards.csv')
            with open(training_rewards_file, mode) as f:
                csv_writer = csv.writer(f)
                if mode == 'w':  # write header
                    csv_writer.writerow(['episode_idx'] + [f'rewards_{i}' for i in range(len(episode_r))])
                csv_writer.writerow(np.append([episode], episode_r))
            
            training_lens_file = os.path.join(saving_dir, 'training_lens.csv')
            with open(training_lens_file, mode) as f:
                csv_writer = csv.writer(f)
                if mode == 'w':  # write header
                    csv_writer.writerow(['episode_idx'] + [f'episode_len_{i}' for i in range(len(episode_len))])
                csv_writer.writerow(np.append([episode], episode_len))
            
            training_start_pos_file = os.path.join(saving_dir, 'training_start_pos.csv')
            with open(training_start_pos_file, mode) as f:
                csv_writer = csv.writer(f)
                if mode == 'w':  # write header
                    csv_writer.writerow(['episode_idx'] + [f'episode_start_pos_{i}' for i in range(len(episode_start_poss))])
                csv_writer.writerow([episode] + episode_start_poss)
            
            # experience replay for each episode
            logger.info(f'Episode {episode} ended. Doing experience replay')
            if goal_conditioned:
                assert goal_state is not None
                assert state_to_goal_fn is not None
                highsight_experience_replay(trajectory, agent.batch_observe, reward_fn, intended_goals=[goal_state] * env.num_envs, reached_goals=episode_reached_pos)
            else:
                normal_experience_replay(trajectory, agent.batch_observe)
            
            # testing the agent
            if testing_freq is not None and episode % testing_freq == 0:
                assert num_test_episodes is not None
                test_env = env if test_env is None else test_env
                test_agent_batch(
                    agent=agent,
                    test_env=test_env,
                    num_episodes=num_test_episodes,
                    cur_episode_idx=episode,
                    goal_conditioned=goal_conditioned,
                    goal_state=goal_state,
                    saving_dir=saving_dir,
                )
            
            # saving the agent
            if saving_freq is not None and episode % saving_freq == 0:
                agent_save_path = Path(saving_dir).joinpath('latest_agent.pkl')
                with open(agent_save_path, 'wb') as f:
                    pickle.dump(agent, f)
            
            # plotting the value function and reward func
            if plotting_freq is not None and episode % plotting_freq == 0:
                # value function
                value_func_dir = Path(saving_dir).joinpath('value_function_plots')
                value_func_dir.mkdir(exist_ok=True)
                utils.make_chunked_value_function_plot(solver=agent, 
                                                        episode=episode, 
                                                        goal=goal_state, 
                                                        saving_dir=value_func_dir, 
                                                        replay_buffer=None)
                # reward
                reward_dir = Path(saving_dir).joinpath('reward_plots')
                reward_dir.mkdir(exist_ok=True)
                utils.make_reward_plot(solver=agent,
                                                episode=episode,
                                                saving_dir=reward_dir,
                                                replay_buffer=None)

                # for debugging purposes, plot the positions where done = True
                done_pos_dir = Path(saving_dir).joinpath('done_position_plots')
                done_pos_dir.mkdir(exist_ok=True)
                utils.make_done_position_plot(solver=agent,
                                                episode=episode,
                                                saving_dir=done_pos_dir,
                                                replay_buffer=None)
                logger.info('made value function plot and reward plot')
    
    except Exception as e:
        logger.info('ooops, sth went wrong :( ')
        env.close()
        traceback.print_exception(type(e), e, e.__traceback__)
        raise e


def episode_rollout(
    testing,
    episode_idx,
    env,
    agent,
    goal_conditoned,
    goal_state,
    logger,
    logging_freq,
    state_to_goal_fn=None,
):
    """
    rollout one episode
    Args:
        testing: whether rolling out for training or testing
    """
    num_envs = env.num_envs

    # o_0, r_0
    obss = env.reset(mask=None, testing=testing)
    obss = list(map(lambda obs: obs.astype(np.float32), obss))  # convert np.float64 to np.float32, for torch forward pass
    if not testing:
        starting_poss = [state_to_goal_fn(obs) for obs in obss]
        episode_reached_pos = starting_poss
        print(f"start position is {starting_poss}")

    # create trajectory accumulator
    if not testing:
        trajectory = []

    # run until all parallel envs finish the current episode
    while not env.all_envs_done:
        # a_t
        assert len(obss) == num_envs
        filtered_obss = filter_token(obss)  # (num_running_env, state_dim)
        if goal_conditoned:
            enhanced_obss = list(map(lambda obs: utils.augment_state(obs, goal_state), filtered_obss))
        else:
            enhanced_obss = filtered_obss
        actions = agent.batch_act(enhanced_obss, evaluation_mode=testing)
        assert len(actions) == num_envs - env.n_done_envs  # assert actions.shape == (num_running_env, action_dim)

        # o_{t+1}, r_{t+1}
        next_obss, rs, dones, infos = env.step(actions)

        # record stats
        if not testing:
            episode_reached_pos = [episode_reached_pos[i] if obs is StopExecution else state_to_goal_fn(obs) for i, obs in enumerate(next_obss)]

        # logging
        if logging_freq and np.max(env.episode_lens) % logging_freq == 0:
            assert logger is not None
            logger.info(
                "at episode {}, step {}, with reward {}".format(
                    episode_idx,
                    env.episode_lens,
                    env.episode_rs,
                )
            )

        # add to experience buffer
        if not testing:
            # make sure everything is the same size
            filtered_next_obss = filter_token(next_obss)  # remove the None
            filtered_actions = filter_token(actions)  # remove the None
            filtered_rs = filter_token(rs)  # remove the None
            filtered_dones = filter_token(dones)  # remove the None
            
            assert len(filtered_obss) == len(filtered_actions) == len(filtered_rs) == len(filtered_next_obss) == len(filtered_dones)
            trajectory.append((filtered_obss, filtered_actions, filtered_rs, filtered_next_obss, filtered_dones, env.episode_dones))
        
        # update obss
        obss = [StopExecution if env.episode_dones[i]==True else obs for i, obs in enumerate(next_obss)]  # mask done states
    
    if testing:
        return env.episode_rs, env.episode_successes
    else:
        return trajectory, env.episode_rs, env.episode_lens, starting_poss, episode_reached_pos


def experience_replay(t, observe_fn, reward_fn=None, target_goals=None):
    """
    experience replay targeting a specific goal
    Args:
        t: trajectory, which is a list of transitions
        observe_fn: the agent.observe function
        reward_fn: the reward function of the environment. Need this because in HER,
                    we need to override the rewards & dones in case the target_goal
                    is the reached_goal
                    reward_fn(state, goal) -> reward, done
        target_goal: should be a np.array of length num_envs, and each element of length goal_size
    """
    if target_goals is None:
        # for normal ER
        for obss, actions, rs, next_obss, dones, terminal in t:
            observe_fn(obss, actions, rs, next_obss, dones)
    else:
        # for HER
        prev_terminal = [False] * len(t[0][0])
        for obss, actions, rs, next_obss, dones, terminal in t:
            # get target_goals for env_idx that have not terminated
            # check prev_terminal because len(obss) is dependent on whether previous step terminated
            step_target_goals = target_goals[np.logical_not(prev_terminal)]
            assert len(obss) == len(next_obss) == len(step_target_goals)
            prev_terminal = terminal

            # override reward for target_goals
            rs, dones = reward_fn(next_obss, step_target_goals)

            # augment the obss with target_goal
            goal_augmented_obss = [utils.augment_state(obs, g) for obs, g in zip(obss, step_target_goals)]
            # augment the next_obss with target goal
            goal_augmented_next_obss = [utils.augment_state(obs, g) for obs, g in zip(next_obss, step_target_goals)]
            observe_fn(goal_augmented_obss, actions, rs, goal_augmented_next_obss, dones)


def normal_experience_replay(trajectories, agent_observe_fn):
    """
    normal experience replay
    """
    experience_replay(t=trajectories, observe_fn=agent_observe_fn, target_goals=None)


def highsight_experience_replay(trajectories, agent_observe_fn, reward_fn, intended_goals, reached_goals):
    """
    highsight experience replay
    Args:
        intended_goals: a list of shape (num_envs, goal_size)
        reached_goals: a list of shape (num_envs, reached_goals)
    """
    # hindsight
    experience_replay(t=trajectories, observe_fn=agent_observe_fn, reward_fn=reward_fn, target_goals=np.array(intended_goals))
    experience_replay(t=trajectories, observe_fn=agent_observe_fn, reward_fn=reward_fn, target_goals=np.array(reached_goals))


def test_agent_batch(
    agent,
    test_env,
    num_episodes,
    cur_episode_idx,
    goal_conditioned,
    goal_state,
    saving_dir,
):
    """
    test the agent for num_episodes episodes
    """

    logger = logging.getLogger("testing")
    logger.setLevel(logging.INFO)

    assert isinstance(test_env, EpisodicSyncVectorEnv)

    # main training loops
    try:
        success_rates = np.array([])
        rewards = np.array([])
        # for each episode
        for episode in range(num_episodes):
            # episode rollout
            episode_r, episode_success = episode_rollout(
                testing=True,
                episode_idx=episode,
                env=test_env,
                agent=agent,
                goal_conditoned=goal_conditioned,
                goal_state=goal_state,
                logger=None,
                logging_freq=None
            )
            
            # logging the success rate per episode
            success_rates = np.concatenate([success_rates, episode_success], axis=0)
            rewards = np.concatenate([rewards, episode_r], axis=0)
            logger.info(
                "testing episode {} with success rate: {} and reward {}".format(
                    episode,
                    episode_success,
                    episode_r,
                )
            )

        # save the success metrics per testing run
        mode = 'w' if cur_episode_idx == 0 else 'a'
        success_file = os.path.join(saving_dir, 'testing_success_rates.csv')
        with open(success_file, mode) as f:
            csv_writer = csv.writer(f)
            if mode == 'w':  # write header
                csv_writer.writerow(['episode_idx'] + [f'success_rate_{i}' for i in range(len(success_rates))])
            csv_writer.writerow(np.append([cur_episode_idx], success_rates))
        logger.info(f"saved to file {success_file}")

        rewards_file = os.path.join(saving_dir, 'testing_rewards.csv')
        with open(rewards_file, mode) as f:
            csv_writer = csv.writer(f)
            if mode == 'w':  # write header
                csv_writer.writerow(['episode_idx'] + [f'rewards_{i}' for i in range(len(rewards))])
            csv_writer.writerow(np.append([cur_episode_idx], rewards))
        logger.info(f"saved to file {rewards_file}")
    
    except Exception as e:
        logger.info('ooops, sth went wrong during testing :( ')
        test_env.close()
        traceback.print_exception(type(e), e, e.__traceback__)
        raise e
