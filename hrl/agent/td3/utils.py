import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def save(td3_agent, filename):
    torch.save(td3_agent.critic.state_dict(), filename + "_critic")
    torch.save(td3_agent.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    torch.save(td3_agent.actor.state_dict(), filename + "_actor")
    torch.save(td3_agent.actor_optimizer.state_dict(), filename + "_actor_optimizer")


def load(td3_agent, filename):
    td3_agent.critic.load_state_dict(torch.load(filename + "_critic"))
    td3_agent.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
    td3_agent.critic_target = copy.deepcopy(td3_agent.critic)

    td3_agent.actor.load_state_dict(torch.load(filename + "_actor"))
    td3_agent.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
    td3_agent.actor_target = copy.deepcopy(td3_agent.actor)


def make_chunked_value_function_plot(solver, episode, seed, experiment_name, chunk_size=1000, replay_buffer=None):
    replay_buffer = replay_buffer if replay_buffer is not None else solver.replay_buffer
    states = np.array([exp[0] for exp in replay_buffer])
    actions = np.array([exp[1] for exp in replay_buffer])

    # Chunk up the inputs so as to conserve GPU memory
    num_chunks = int(np.ceil(states.shape[0] / chunk_size))

    if num_chunks == 0:
        return 0.

    state_chunks = np.array_split(states, num_chunks, axis=0)
    action_chunks = np.array_split(actions, num_chunks, axis=0)
    qvalues = np.zeros((states.shape[0],))
    current_idx = 0

    for state_chunk, action_chunk in tqdm(zip(state_chunks, action_chunks), desc="Making VF plot"):
        state_chunk = torch.from_numpy(state_chunk).float().to(solver.device)
        action_chunk = torch.from_numpy(action_chunk).float().to(solver.device)
        chunk_qvalues = solver.get_qvalues(state_chunk, action_chunk).cpu().numpy().squeeze(1)
        current_chunk_size = len(state_chunk)
        qvalues[current_idx:current_idx + current_chunk_size] = chunk_qvalues
        current_idx += current_chunk_size

    plt.scatter(states[:, 0], states[:, 1], c=qvalues)
    plt.colorbar()
    file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}"
    plt.savefig(f"value_function_plots/{experiment_name}/{file_name}.png")
    plt.close()

    return qvalues.max()
