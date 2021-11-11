import pfrl
import random
import argparse
import cv2
import pdb
import collections
import numpy as np
import matplotlib.pyplot as plt
from pfrl.wrappers import atari_wrappers
import hrl.utils

from tqdm import tqdm
from hrl.tasks.monte.MRRAMMDPClass import MontezumaRAMMDP
from hrl.agent.rainbow.rainbow import Rainbow

def write_to_disk(traj):
    idx = 0
    for result in traj:
        img = result[0]
        idx += 1
        cv2.imwrite(f'debug-images/{idx}.png', np.squeeze(img._frames[-1]))
    print(f'wrote {len(traj)} images')

def make_chunked_gc_value_function_plot(pfrl_agent, states, goal, goal_pos, episode, seed, experiment_name, chunk_size=1000):
    assert isinstance(pfrl_agent, Rainbow)

    def get_augmented_states(s, g):
        assert isinstance(g, (np.ndarray, atari_wrappers.LazyFrames)), type(g)
        g = g._frames[-1] if isinstance(g, atari_wrappers.LazyFrames) else g
        augmented_states = [atari_wrappers.LazyFrames(ss.image._frames+[g], stack_axis=0) for ss in s]
        return augmented_states

    def get_chunks(x, n):
        """ Break x into chunks of size n. """
        for i in range(0, len(x), n):
            yield x[i: i+n]

    augmented_states = get_augmented_states(states, goal)
    augmented_state_chunks = get_chunks(augmented_states, chunk_size)
    values = np.zeros((len(augmented_states),))
    current_idx = 0

    for state_chunk in tqdm(augmented_state_chunks, desc="Making VF plot"):
        chunk_values = pfrl_agent.value_function(state_chunk).cpu().numpy()
        current_chunk_size = len(state_chunk)
        values[current_idx:current_idx + current_chunk_size] = chunk_values
        current_idx += current_chunk_size

    x = [state.get_player_x(state.ram) for state in states]
    y = [state.get_player_y(state.ram) for state in states]

    plt.scatter(x, y, c=values)
    plt.colorbar()
    hrl.utils.create_log_dir(f'plots/{experiment_name}')
    hrl.utils.create_log_dir(f'plots/{experiment_name}/{seed}')
    file_name = f"rainbow_value_function_seed_{seed}_episode_{episode}"
    plt.savefig(f"plots/{experiment_name}/{seed}/{file_name}-{goal_pos}.png")
    plt.close()

    return values.max()

def test(starts, goals, agent, goal_img, episode):
    for start, goal in zip(starts, goals):
        mdp.reset()
        mdp.set_player_position(*goal)
        goal_img = mdp.curr_state.image
        mdp.reset()
        mdp.set_player_position(*start)
        
        agent.gc_rollout(mdp,
                        goal_img,
                        end,
                        current_episode_number,
                        max_episodic_reward,
                        test=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--n_atoms", type=int, default=51)
    parser.add_argument("--v_max", type=float, default=+10.)
    parser.add_argument("--v_min", type=float, default=-10.)
    parser.add_argument("--noisy_net_sigma", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=6.25e-5)
    parser.add_argument("--n_steps", type=int, default=3)
    parser.add_argument("--replay_start_size", type=int, default=500)
    parser.add_argument("--replay_buffer_size", type=int, default=10**6)
    parser.add_argument("--num_training_steps", type=int, default=int(13e6))

    args = parser.parse_args()

    """ After arg parsing """

    pfrl.utils.set_random_seed(args.seed)
    mdp = MontezumaRAMMDP(render=False, seed=args.seed)
    rainbow_agent = Rainbow(n_actions=mdp.env.action_space.n,
                            n_atoms=args.n_atoms,
                            v_min=args.v_min,
                            v_max=args.v_max,
                            noisy_net_sigma=args.noisy_net_sigma,
                            lr=args.lr,
                            n_steps=args.n_steps,
                            betasteps=args.num_training_steps / 4,
                            replay_start_size=args.replay_start_size,
                            replay_buffer_size=args.replay_buffer_size,
                            gpu=args.gpu_id,
                            goal_conditioned=True,
                            use_her=False,
                    )

    starts = [(38,148), (114,148)]
    goals = [(114,148), (38,148)]



    current_step_number = 0
    max_episodic_reward = 0
    current_episode_number = 0
    
    ram_buffer = collections.deque(maxlen=1500)

    ctr = 0

    while current_step_number < args.num_training_steps:
        # sample start,goal pair
        idx = random.randint(0,1)
        start = starts[idx]
        goal = goals[idx]

        mdp.reset()
        mdp.set_player_position(*goal)
        goal_img = mdp.curr_state.image

        mdp.reset()
        mdp.set_player_position(*start)
        # end sampling procedure
        
        episodic_reward, episodic_duration, max_episodic_reward, trajectory, ram_trajectory = rainbow_agent.gc_rollout(mdp,
                                                                                            goal_img,
                                                                                            goal,
                                                                                            current_episode_number,
                                                                                            max_episodic_reward)

        ram_buffer.extend(ram_trajectory)
        if episodic_reward > 0:
            ctr += 1

        if episodic_reward > 0 and ctr >= 100:
            pdb.set_trace()
            make_chunked_gc_value_function_plot(rainbow_agent, list(ram_buffer), goal_img, goal, current_episode_number, args.seed, "test")
        # write_to_disk(trajectory)


    # mdp.saveImage("img1")
    # mdp.set_player_position(start[0], start[1])
    # mdp.saveImage("img2")
    # print("success!")

# python -m hrl.experiments.monte_test