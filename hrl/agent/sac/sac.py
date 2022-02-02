import ipdb
import pfrl
import torch
import numpy as np
from torch import Tensor, nn
from torch import distributions
from pfrl import replay_buffers
from pfrl.nn.lmbda import Lambda
from pfrl.agents import SoftActorCritic


class SAC:
    def __init__(self,
                obs_size,
                action_size,
                batch_size,
                replay_start_size,
                replay_buffer_size,
                policy_output_scale,
                action_space_low,
                action_space_high,
                gpu) -> None:

        def squashed_diagonal_gaussian_head(x):
            assert x.shape[-1] == action_size * 2
            mean, log_scale = torch.chunk(x, 2, dim=1)
            log_scale = torch.clamp(log_scale, -20.0, 2.0)
            var = torch.exp(log_scale * 2)
            base_distribution = distributions.Independent(
                distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
            )
            # cache_size=1 is required for numerical stability
            return distributions.transformed_distribution.TransformedDistribution(
                base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
            )

        self.policy = nn.Sequential(
                nn.Linear(obs_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_size * 2),
                Lambda(squashed_diagonal_gaussian_head),
        )
        
        torch.nn.init.xavier_uniform_(self.policy[0].weight)
        torch.nn.init.xavier_uniform_(self.policy[2].weight)
        torch.nn.init.xavier_uniform_(self.policy[4].weight, gain=policy_output_scale)
        policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

        def make_q_func_with_optimizer():
            q_func = nn.Sequential(
                pfrl.nn.ConcatObsAndAction(),
                nn.Linear(obs_size + action_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
            torch.nn.init.xavier_uniform_(q_func[1].weight)
            torch.nn.init.xavier_uniform_(q_func[3].weight)
            torch.nn.init.xavier_uniform_(q_func[5].weight)
            q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=3e-4)
            return q_func, q_func_optimizer

        self.q_func1, q_func1_optimizer = make_q_func_with_optimizer()
        self.q_func2, q_func2_optimizer = make_q_func_with_optimizer()

        self.replay_buffer = replay_buffers.ReplayBuffer(replay_buffer_size)

        def burnin_action_func():
            """Select random actions until model is updated one or more times."""
            return np.random.uniform(action_space_low, action_space_high).astype(np.float32)

        self.agent = SoftActorCritic(
                        self.policy,
                        self.q_func1,
                        self.q_func2,
                        policy_optimizer,
                        q_func1_optimizer,
                        q_func2_optimizer,
                        self.replay_buffer,
                        gamma=0.99,
                        replay_start_size=replay_start_size,
                        gpu=gpu,
                        minibatch_size=batch_size,
                        burnin_action_func=burnin_action_func,
                        entropy_target=-action_size,
                        temperature_optimizer_lr=3e-4,
                        batch_states=self.batch_states,
                        phi=self.phi
        )

        self.T = 0
        self.device = torch.device(f"cuda:{gpu}" if gpu > -1 else "cpu")

    @staticmethod
    def phi(x):
        """ Observation pre-processing for neural networks. """
        return np.asarray(x, dtype=np.float32)

    def batch_states(self, states, device):
        assert isinstance(states[0], np.ndarray), type(states[0])
        features = np.array([self.agent.phi(s) for s in states])
        return torch.as_tensor(features).to(device)

    def act(self, state):
        """ Action selection method at the current state. """
        return self.agent.act(state)

    def step(self, state, action, reward, next_state, done, reset):
        """ Learning update based on a given transition from the environment. """
        self._overwrite_pfrl_state(state, action)
        self.agent.observe(next_state, reward, done, reset)

    def _overwrite_pfrl_state(self, state, action):
        """ Hack the pfrl state so that we can call act() consecutively during an episode before calling step(). """
        self.agent.batch_last_obs = [state]
        self.agent.batch_last_action = [action]

    @torch.no_grad()
    def value_function(self, states):
        """ Query the value function for the set of input states. """
        actions = self.agent.batch_select_greedy_action(states)
        return self.action_value_function(states, actions)

    @torch.no_grad()
    def action_value_function(self, states, actions):
        batch_states = states if isinstance(states, torch.Tensor) else \
            self.agent.batch_states(
                states, self.device, self.phi
        )
        batch_actions = torch.as_tensor(actions).float().to(self.device)
        q1 = self.agent.q_func1((batch_states, batch_actions))
        q2 = self.agent.q_func2((batch_states, batch_actions))
        return torch.min(q1, q2)

    def experience_replay(self, trajectory):
        """ Add trajectory to the replay buffer and perform agent learning updates. """

        for transition in trajectory:
            self.step(*transition)

    def rollout(self, env, state, episode):
        """ Single episode of interaction with the env followed by experience replay. """
        
        done = False
        reset = False
        reached = False

        episode_length = 0
        episode_reward = 0.
        episode_trajectory = []

        while not done and not reset and not reached:
            action = self.act(state)
            next_state, reward, done, info = env.step(action)
            
            reset = info.get("needs_reset", False)

            episode_trajectory.append((state,
                                       action,
                                       np.sign(reward), 
                                       next_state, 
                                       done or reached, 
                                       reset))

            self.T += 1
            episode_length += 1
            episode_reward += reward

            state = next_state
        
        self.experience_replay(episode_trajectory)
        self.log_progress(env, episode, episode_reward, episode_length)
        
        return episode_reward, episode_length

    def log_progress(self, env, episode, episode_reward, episode_length):
        start = env.get_position(env.init_state).round(decimals=2)
        print(f"Episode: {episode}, Starting at {start}, Reward: {episode_reward}, Length: {episode_length}")
