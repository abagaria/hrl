import numpy as np
from copy import deepcopy
from hrl.utils import create_log_dir
from hrl.agent.td3.TD3AgentClass import TD3
from hrl.agent.td3.utils import make_chunked_value_function_plot
from hrl.mdp.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP


EPISODES_PER_GOAL = 100
EXPERIMENT_NAME = "vanilla-td3-curriculum-umaze"


def curriculum_goal(episode):
    waypoints = [(2, 0), (4, 0), (6, 0), (8, 0),
                 (8, 2), (8, 4), (8, 6), (8, 8),
                 (6, 8), (4, 8), (2, 8), (0, 8)]
    idx = min(episode // EPISODES_PER_GOAL, len(waypoints) - 1)
    return waypoints[idx]


def experience_replay(agent, mdp, trajectory, goal):
    for state, action, _, next_state in trajectory:
        reward, done = mdp.sparse_gc_reward_function(next_state, goal, info={})
        agent.step(state.features(), action, reward, next_state.features(), done)


def rollout(agent, mdp, goal, steps):
    score = 0.
    mdp.reset()
    trajectory = []

    for step in range(steps):
        state = deepcopy(mdp.cur_state)
        action = agent.act(state.features())

        _, next_state = mdp.execute_agent_action(action)
        reward, done = mdp.sparse_gc_reward_function(next_state, goal, info={})

        score = score + reward
        trajectory.append((state, action, reward, next_state))

        if done:
            break

    return score, trajectory


def training_loop(num_episodes, num_steps):
    mdp = D4RLAntMazeMDP("umaze", goal_state=np.array((0, 8)), seed=0)

    agent = TD3(state_dim=mdp.state_space_size(),
                action_dim=mdp.action_space_size(),
                max_action=1.,
                use_output_normalization=False)

    per_episode_scores = []

    for episode in range(num_episodes):
        mdp.reset()
        goal = curriculum_goal(episode)
        score, trajectory = rollout(agent, mdp, goal, num_steps)
        experience_replay(agent, mdp, trajectory, goal)

        per_episode_scores.append(score)
        print(f"Episode: {episode} | Goal: {goal} | Score: {score}")
        if episode > 0 and episode % EPISODES_PER_GOAL == 0:
             make_chunked_value_function_plot(agent, episode, 0, EXPERIMENT_NAME)

    return per_episode_scores


if __name__ == "__main__":
    create_log_dir("value_function_plots")
    create_log_dir(f"value_function_plots/{EXPERIMENT_NAME}")
    pes = training_loop(num_episodes=EPISODES_PER_GOAL*12, num_steps=1000)
