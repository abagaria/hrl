import logging

import numpy as np
import pfrl


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

    assert isinstance(env, pfrl.envs.MultiprocessVectorEnv)
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

            # run until all parallel envs reach the max_episode
            while np.any(episode_len < max_episode_len):
                # a_t
                actions = agent.batch_act(obss)
                # o_{t+1}, r_{t+1}
                obss, rs, dones, infos = env.step(actions)
                episode_r += rs
                episode_len += 1

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
                not_end = np.logical_not(end)

                # add to experience buffer
                trajectory.append((obss, actions, rs, dones, resets))

                # reset the env to start a new episode
                # mask the episode that didn't end, so they don't get reset
                obss = env.reset(mask=not_end)

                # TODO: 
                # if episode_len== num_episode, that particular env should stop executing
            
            # experience replay
            logger.info(f'Episode {episode} ended. Doing experience replay')
            if goal_conditioned:
                assert goal_state is not None
                highsight_experience_replay(trajectory, agent.batch_observe, goal_state)
            else:
                experience_replay(trajectory, agent.batch_observe)
    
    except Exception:
        logger.info('ooops, sth went wrong :( ')
        env.close()


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
