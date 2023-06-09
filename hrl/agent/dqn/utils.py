import numpy as np
from .dqn import DQN
from PIL import Image
from pfrl.wrappers import atari_wrappers


def save_frame(state, image_name, experiment_name, seed):
    assert isinstance(state, atari_wrappers.LazyFrames)
    frame = state._frames[-1].squeeze()
    image = Image.fromarray(frame)
    image.save(f"plots/{experiment_name}/{seed}/{image_name}.png")


def save_frames_with_highest_reward(agent, experiment_name, seed):
    assert isinstance(agent, DQN)

    states = [experience["next_state"] for experience in agent.rbuf.memory]
    rewards = [experience["reward"] for experience in agent.rbuf.memory]

    max_reward = rewards.max()
    max_reward_label = f"{np.round(max_reward, 1)}"

    for i, (state, reward) in enumerate(zip(states, rewards)):
        if np.isclose(reward, max_reward):
            save_frame(state, f"max_reward_{max_reward_label}_{i}", experiment_name, seed)


def save_terminal_frames(agent, experiment_name, seed):
    states = [experience["next_state"] for experience in agent.rbuf.memory]
    dones = [experience["is_state_terminal"] for experience in agent.rbuf.memory]

    for i, (state, done) in enumerate(zip(states, dones)):
        if done: save_frame(state, f"done_{i}", experiment_name, seed)


def save_frames_with_higest_qvalue(agent, experiment_name, seed):
    pass
