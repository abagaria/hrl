import pfrl
import argparse
import cv2
import pdb
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from hrl.tasks.monte.MRRAMMDPClass import MontezumaRAMMDP
from hrl.agent.rainbow.rainbow import Rainbow

def write_to_disk(traj):
    idx = 0
    for result in traj:
        img = result[0]
        idx += 1
        cv2.imwrite(f'debug-images/{idx}.png', img[:,:,-1])
    print(f'wrote {len(traj)} images')

def make_chunked_gc_value_function_plot(pfrl_agent, states, goal, episode, seed, experiment_name, chunk_size=1000):
    assert isinstance(pfrl_agent, Rainbow)

    def get_augmented_state(state, goal):
        if np.shape(goal) == (84, 84):
            goal = np.expand_dims(goal, 2)
        res = np.concatenate((goal, state), axis=2)
        return np.reshape(res, (5, 84, 84))

    def get_chunks(x, n):
        """ Break x into chunks of size n. """
        for i in range(0, len(x), n):
            yield x[i: i+n]

    states = augmenent_states(states, goal)
    state_chunks = get_chunks(states, chunk_size)
    values = np.zeros((len(states),))
    current_idx = 0

    for state_chunk in tqdm(state_chunks, desc="Making VF plot"):
        chunk_values = pfrl_agent.value_function(state_chunk).cpu().numpy()
        current_chunk_size = len(state_chunk)
        values[current_idx:current_idx + current_chunk_size] = chunk_values
        current_idx += current_chunk_size

    x = [info['player_x'] for info in infos]
    y = [info['player_y'] for info in infos]

    plt.scatter(x, y, c=values)
    plt.colorbar()
    file_name = f"rainbow_value_function_seed_{seed}_episode_{episode}"
    plt.savefig(f"plots/{experiment_name}/{seed}/{file_name}.png")
    plt.close()

    return values.max()

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
    parser.add_argument("--replay_start_size", type=int, default=80_000)
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

    start = (77,235)
    end = (130,192)

    stend_tuple = (start, end)

    mdp.set_player_position(end[0], end[1])
    goal_img = mdp.curr_state.image[:,:,-1]

    current_step_number = 0
    max_episodic_reward = 0
    current_episode_number = 0
    
    while current_step_number < args.num_training_steps:
        mdp.reset()
        mdp.set_player_position(*start)
        
        episodic_reward, episodic_duration, max_episodic_reward, trajectory = rainbow_agent.gc_rollout(mdp,
                                                                                            goal_img,
                                                                                            end,
                                                                                            current_episode_number,
                                                                                            max_episodic_reward)
        # print((rainbow_agent.my_dict))
        # pdb.set_trace()
        if episodic_reward >= -100:
            # write_to_disk(trajectory)
            pdb.set_trace()
        # write_to_disk(trajectory)


    # mdp.saveImage("img1")
    # mdp.set_player_position(start[0], start[1])
    # mdp.saveImage("img2")
    # print("success!")

# python -m hrl.experiments.monte_test