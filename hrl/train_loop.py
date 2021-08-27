import os
import csv
import logging
import traceback

import numpy as np

from hrl.envs.vector_env import SyncVectorEnv
from hrl import utils


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
            trajectory = episode_rollout(
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
            
            # plotting the value function
            if plotting_freq is not None and episode % plotting_freq == 0:
                utils.make_chunked_value_function_plot(solver=agent, 
                                                        episode=episode, 
                                                        goal=np.repeat(goal_state, env.num_envs, 0).reshape((env.num_envs, -1)), 
                                                        saving_dir=saving_dir, 
                                                        replay_buffer=trajectory)
                logger.info('making value function plot')
    
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
        print(f"start position is {[obs[:2] for obs in obss]}")

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
            obss = list(map(lambda obs: utils.augment_state(obs, goal_state), obss))
        actions = agent.batch_act(obss)

        # mask actions for each env, done_envs has action None
        for i, a in enumerate(actions):
            if episode_done[i]:
                actions[i] = None

        # o_{t+1}, r_{t+1}
        obss, rs, dones, infos = env.step(actions)
        obss = list(filter(lambda obs: obs is not None, obss))  # remove the None
        obss = list(map(lambda obs: obs.astype(np.float32), obss))  # convert np.float64 to np.float32, for torch forward pass
        rs = list(map(lambda r: 0 if r is None else r, rs))  # change None to 0
        dones = list(map(lambda done: 1 if done is None else 0, dones))  # change None to 1
        infos = list(map(lambda info: {} if info is None else info, infos))  # change None to {}

        # record stats
        if testing:
            episode_success = np.logical_or(episode_success, dones)
        episode_not_done = np.logical_not(episode_done)
        episode_r += rs * episode_not_done
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
        end = np.logical_or(resets, dones)
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
            trajectory.append((obss, actions, rs, dones, resets))
    
    if testing:
        return episode_r, episode_success
    else:
        return trajectory


def experience_replay(trajectories, agent_observe_fn):
    """
    normal experience replay
    """
    for obss, actions, rs, dones, resets in trajectories:
        agent_observe_fn(obss, rs, dones, resets)


def highsight_experience_replay(trajectories, agent_observe_fn, goal_state, state_to_goal_fn=lambda x: x):
    """
    highsight experience replay
    """
    last_obss = trajectories[-1][0]  # the last observation
    reached_goals = list(map(state_to_goal_fn, last_obss))
    for obss, actions, rs, dones, resets, in trajectories:
        goal_augmented_obss = list(map(lambda obs: utils.augment_state(obs, goal_state), obss))
        reached_goal_augmented_obss = []
        for obs, reached_goal in zip(obss, reached_goals):
            reached_goal_augmented_obss.append(utils.augment_state(obs, reached_goal))
        agent_observe_fn(goal_augmented_obss, rs, dones, resets)
        agent_observe_fn(reached_goal_augmented_obss, rs, dones, resets)


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
