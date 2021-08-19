import logging

import numpy as np

from hrl.envs.vector_env import SyncVectorEnv


def train_agent_batch(
    agent,
    env,
    num_episodes,
    goal_conditioned=False,
    goal_state = None,
    max_episode_len=None,
    logging_freq=None,
):
    """Train an agent in a batch environment.

    Args:
        agent: Agent to train.
        env: Environment to train the agent against.
        num_episodes (int): number of episodes to train the agent
        params (dict): training parameters
        max_episode_len (int): max steps in one episode 
        the average returns of the current agent.
    Returns:
        List of evaluation episode stats dict.
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    assert isinstance(env, SyncVectorEnv)
    num_envs = env.num_envs

    # main training loops
    try:
        # for each episode
        for episode in range(num_episodes):
            # o_0, r_0
            obss = env.reset()

            trajectory = []
            episode_r = np.zeros(num_envs, dtype=np.float64)  # reward for the current episode
            episode_len = np.zeros(num_envs, dtype="i")  # the idx of step each env is on for their cur episode
            episode_done = np.zeros(num_envs, dtype="i")  # whether the corresponding env is done with the cur episode

            # run until all parallel envs finish the current episode
            while not np.all(episode_done):
                # a_t
                actions = agent.batch_act(obss)

                # mask actions for each env, done_envs has action None
                episode_not_done = np.logical_not(episode_done)
                for i, a in enumerate(actions):
                    if episode_done[i]:
                        actions[i] = None

                # o_{t+1}, r_{t+1}
                obss, rs, dones, infos = env.step(actions)
                episode_r += rs * episode_not_done
                episode_len += 1 * episode_not_done

                # logging
                if logging_freq and np.max(episode_len) % logging_freq == 0:
                    logger.info(
                        "at episode {}, step {}, with reward {}".format(
                            episode,
                            episode_len,
                            episode_r,
                        )
                    )

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

                # add to experience buffer
                trajectory.append((obss, actions, rs, dones, resets))
            
            # experience replay
            logger.info(f'Episode {episode} ended. Doing experience replay')
            if goal_conditioned:
                assert goal_state is not None
                highsight_experience_replay(trajectory, agent.batch_observe, goal_state)
            else:
                experience_replay(trajectory, agent.batch_observe)
    
    except Exception as e:
        logger.info('ooops, sth went wrong :( ')
        env.close()
        raise e


def experience_replay(trajectories, agent_observe_fn):
    """
    normal experience replay
    """
    for obss, actions, rs, dones, resets in trajectories:
        agent_observe_fn(obss, rs, dones, resets)


def highsight_experience_replay(trajectories, agent_observe_fn, goal_state):
    """
    highsight experience replay
    """
    def augment_state(obss, goal):
        """
        make the state goal-conditioned by concating state with goal
        NOTE: obss is a batch of obs, while goal is a single goal
        """
        assert len(obss.shape) == 2
        assert len(goal.shape) == 1
        batched_goal = np.repeat(goal.reshape(1, -1), repeats=obss.shape[0], axis=0)
        aug = np.concatenate([obss, batched_goal], axis=-1)
        return aug

    reached_goal = trajectories[-1][0]  # the last observation
    for obss, actions, rs, dones, resets, in trajectories:
        goal_augmented_obss = augment_state(obss, goal_state)
        reached_goal_augmented_obss = augment_state(obss, reached_goal)
        agent_observe_fn(goal_augmented_obss, rs, dones, resets)
        agent_observe_fn(reached_goal_augmented_obss, rs, dones, resets)
