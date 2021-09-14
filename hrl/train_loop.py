import os
import csv
import logging
import traceback
from pathlib import Path

import numpy as np

from hrl.envs.vector_env import SyncVectorEnv
from hrl import utils
from hrl.utils import StopExecution


def train_agent_batch_with_eval(
    agent,
    env,
    num_episodes,
    test_env=None,
    num_test_episodes=None,
    goal_conditioned=False,
    goal_state=None,
    max_episode_len=None,
    logging_freq=None,
    testing_freq=None,
    plotting_freq=None,
    saving_dir=None,
    state_to_goal_fn=None,
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

    assert isinstance(env, SyncVectorEnv)

    # main training loops
    try:
        # for each episode
        for episode in range(num_episodes):
            # episode rollout
            trajectory, episode_r, episode_len, episode_start_poss = episode_rollout(
                testing=False,
                episode_idx=episode,
                env=env,
                agent=agent,
                max_episode_len=max_episode_len,
                goal_conditoned=goal_conditioned,
                goal_state=goal_state,
                logger=logger,
                logging_freq=logging_freq,
            )
            assert len(episode_r) == len(episode_len) == len(episode_start_poss)

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
                highsight_experience_replay(trajectory, agent.batch_observe, goal_state, state_to_goal_fn=state_to_goal_fn)
            else:
                experience_replay(trajectory, agent.batch_observe)
            
            # testing the agent
            if testing_freq is not None and episode % testing_freq == 0:
                assert num_test_episodes is not None
                test_env = env if test_env is None else test_env
                test_agent_batch(
                    agent=agent,
                    test_env=test_env,
                    num_episodes=num_test_episodes,
                    cur_episode_idx=episode,
                    max_episode_len=max_episode_len,
                    goal_conditioned=goal_conditioned,
                    goal_state=goal_state,
                    saving_dir=saving_dir,
                )
            
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
    max_episode_len,
    logger,
    logging_freq,
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
        starting_poss = [obs[:2] for obs in obss]
        print(f"start position is {starting_poss}")

    if not testing:
        trajectory = []
    episode_r = np.zeros(num_envs, dtype=np.float32)  # reward for the current episode
    episode_len = np.zeros(num_envs, dtype="i")  # the idx of step each env is on for their cur episode
    episode_done = np.zeros(num_envs, dtype="i")  # whether the corresponding env is done with the cur episode
    if testing:
        episode_success = np.zeros(num_envs, dtype="i")  # whether each corresponding succeeded

    # run until all parallel envs finish the current episode
    while not np.all(episode_done):
        # a_t
        if goal_conditoned:
            enhanced_obss = list(map(lambda obs: utils.augment_state(obs, goal_state), obss))
        else:
            enhanced_obss = obss
        actions = agent.batch_act(enhanced_obss, evaluation_mode=testing)

        # mask actions for each env, done_envs has action None
        # but first need to ensure that actions have the correct length
        try:
            assert len(actions) == num_envs
        except AssertionError:
            action_iter = iter(actions)
            actions = [StopExecution if episode_done[idx_env] else next(action_iter) for idx_env in range(num_envs)]
            assert len(actions) == num_envs

        # o_{t+1}, r_{t+1}
        next_obss, rs, dones, infos = env.step(actions)
        zeroed_rs = list(map(lambda r: 0 if r is StopExecution else r, rs))  # change None to 0
        terminal = list(map(lambda done: 1 if done is StopExecution else done, dones))  # change None to 1
        infos = list(map(lambda info: {} if info is StopExecution else info, infos))  # change None to {}

        # record stats
        if testing:
            episode_success = np.logical_or(episode_success, terminal)
        episode_not_done = np.logical_not(episode_done)
        episode_r += zeroed_rs * episode_not_done
        episode_len += 1 * episode_not_done

        # Compute mask for done and reset
        if max_episode_len is None:
            resets = np.zeros(num_envs, dtype=bool)
        else:
            resets = episode_len == max_episode_len
        resets = np.logical_or(
            resets, [info.get("needs_reset", False) for info in infos]
        )
        # Make mask. 0 if done/reset, 1 if pass
        end = np.logical_or(resets, terminal)
        episode_done = np.logical_or(episode_done, end)

        # logging
        if logging_freq and np.max(episode_len) % logging_freq == 0:
            assert logger is not None
            logger.info(
                "at episode {}, step {}, with reward {}".format(
                    episode_idx,
                    episode_len,
                    episode_r,
                )
            )

        # add to experience buffer
        if not testing:
            # make sure everything is the same size
            try:
                next_obss = list(filter(lambda obs: obs is not StopExecution, next_obss))  # remove the None
                assert len(obss) == len(next_obss)
            except AssertionError:  # some envs hit Nones
                # obss should only take into account the indices on which next_obss is not None
                not_none_index = [i for i, obs in enumerate(next_obss) if next_obss[i] is not StopExecution]
                obss = [obss[idx] for idx in not_none_index]
                next_obss = [next_obss[idx] for idx in not_none_index]
                assert len(obss) == len(next_obss)
            finally:
                next_obss = list(map(lambda obs: obs.astype(np.float32), next_obss))  # convert np.float64 to np.float32, for torch forward pass
                actions = list(filter(lambda a: a is not StopExecution, actions))  # remove the None
                rs = list(filter(lambda r: r is not StopExecution, rs))  # remove the None
                dones = list(filter(lambda done: done is not StopExecution, dones))  # remove the None
                assert len(obss) == len(actions) == len(rs) == len(next_obss) == len(dones)
            trajectory.append((obss, actions, rs, next_obss, dones, resets))
        
        # update obss
        obss = next_obss
    
    if testing:
        return episode_r, episode_success
    else:
        return trajectory, episode_r, episode_len, starting_poss


def experience_replay_for_goal(t, observe_fn, target_goals=None):
    """
    experience replay targeting a specific goal
    the `target_goal` here should be a list of lenght num_envs, and each element of length goal_size
    """
    if target_goals is None:
        # for normal ER
        for obss, actions, rs, next_obss, dones, resets in t:
            observe_fn(obss, actions, rs, next_obss, dones)
    else:
        # for HER
        for obss, actions, rs, next_obss, dones, resets in t:
            # augment the obss with target_goal
            goal_augmented_obss = [utils.augment_state(obs, g) for obs, g in zip(obss, target_goals)]
            # augment the next_obss with target goal
            goal_augmented_next_obss = [utils.augment_state(obs, g) for obs, g in zip(next_obss, target_goals)]
            observe_fn(goal_augmented_obss, actions, rs, goal_augmented_next_obss, dones)


def experience_replay(trajectories, agent_observe_fn):
    """
    normal experience replay
    """
    experience_replay_for_goal(t=trajectories, observe_fn=agent_observe_fn, target_goals=None)


def highsight_experience_replay(trajectories, agent_observe_fn, goal_state, state_to_goal_fn=lambda x: x):
    """
    highsight experience replay
    """
    # recover the reached goals 
    last_obss = trajectories[-1][0]  # the last observation, -1 index last, 0 index obss in a transition
    reached_goals = list(map(state_to_goal_fn, last_obss))

    # make goal_state into a list
    goal_state = [goal_state] * len(reached_goals)

    # hindsight
    experience_replay_for_goal(t=trajectories, observe_fn=agent_observe_fn, target_goals=goal_state)
    experience_replay_for_goal(t=trajectories, observe_fn=agent_observe_fn, target_goals=reached_goals)


def test_agent_batch(
    agent,
    test_env,
    num_episodes,
    cur_episode_idx,
    max_episode_len,
    goal_conditioned,
    goal_state,
    saving_dir,
):
    """
    test the agent for num_episodes episodes
    """

    logger = logging.getLogger("testing")
    logger.setLevel(logging.INFO)

    assert isinstance(test_env, SyncVectorEnv)

    # main training loops
    try:
        success_rates = []
        rewards = []
        # for each episode
        for episode in range(num_episodes):
            # episode rollout
            episode_r, episode_success = episode_rollout(
                testing=True,
                episode_idx=episode,
                env=test_env,
                agent=agent,
                max_episode_len=max_episode_len,
                goal_conditoned=goal_conditioned,
                goal_state=goal_state,
                logger=None,
                logging_freq=None
            )
            
            # logging the success rate per episode
            success_rates.append(episode_success)
            rewards.append(episode_r)
            logger.info(
                "testing episode {} with success rate: {} and reward {}".format(
                    episode,
                    np.mean(episode_success),
                    np.mean(episode_r),
                )
            )

        # save the success metrics per testing run
        average_success_rates = np.mean(success_rates)
        average_rewards = np.mean(rewards)
        results_file = os.path.join(saving_dir, 'metrics.csv')
        mode = 'w' if cur_episode_idx == 0 else 'a'
        with open(results_file, mode) as f:
            csv_writer = csv.writer(f)
            if mode == 'w':  # write header
                csv_writer.writerow(['episode_idx', 'success_rate', 'reward'])
            csv_writer.writerow([cur_episode_idx, average_success_rates, average_rewards])
        logger.info(f"saved metrics to file {results_file}")
    
    except Exception as e:
        logger.info('ooops, sth went wrong during testing :( ')
        test_env.close()
        traceback.print_exception(type(e), e, e.__traceback__)
        raise e
