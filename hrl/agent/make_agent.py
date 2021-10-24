from distutils.version import LooseVersion

import numpy as np
import torch
from torch import nn, distributions
import pfrl
from pfrl.nn.lmbda import Lambda

from hrl.agent.td3.TD3AgentClass import TD3
from hrl.agent.soft_actor_critic import SoftActorCritic


def make_td3_agent(observation_space, action_space, params):
	"""
	make a custom-made td3 agent
	"""
	obs_size = observation_space.low.size
	if params['goal_conditioned']:
		assert 'goal_state_size' in params
		obs_size += params['goal_state_size']
	return TD3(
		state_dim=obs_size,
		action_dim=action_space.low.size,
		max_action=1,
		device=params['device'],
		name='TD3-agent',
		lr_c=params['lr'],
		lr_a=params['lr'],
		use_output_normalization=False,  # model-free setting
	)


def make_sac_agent(observation_space, action_space, params):
	"""
	return a Soft-Actor-Critic agent, according to params specified
	"""
	if LooseVersion(torch.__version__) < LooseVersion("1.5.0"):
		raise Exception("This script requires a PyTorch version >= 1.5.0")
	
	obs_size = observation_space.low.size
	if params['goal_conditioned']:
		assert 'goal_state_size' in params
		obs_size += params['goal_state_size']
	action_size = action_space.low.size

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

	policy = nn.Sequential(
		nn.Linear(obs_size, 256),
		nn.ReLU(),
		nn.Linear(256, 256),
		nn.ReLU(),
		nn.Linear(256, action_size * 2),
		Lambda(squashed_diagonal_gaussian_head),
	)
	torch.nn.init.xavier_uniform_(policy[0].weight)
	torch.nn.init.xavier_uniform_(policy[2].weight)
	torch.nn.init.xavier_uniform_(policy[4].weight, gain=params['policy_output_scale'])
	policy_optimizer = torch.optim.Adam(policy.parameters(), lr=params['lr'])

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
		q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=params['lr'])
		return q_func, q_func_optimizer

	q_func1, q_func1_optimizer = make_q_func_with_optimizer()
	q_func2, q_func2_optimizer = make_q_func_with_optimizer()

	rbuf = pfrl.replay_buffers.ReplayBuffer(params['buffer_length'])

	def burnin_action_func():
		"""Select random actions until model is updated one or more times."""
		return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

	# Hyperparameters in http://arxiv.org/abs/1802.09477
	gpu = 0 if params['device'] == torch.device('cuda') else -1
	agent = SoftActorCritic(
		policy,
		q_func1,
		q_func2,
		policy_optimizer,
		q_func1_optimizer,
		q_func2_optimizer,
		rbuf,
		gamma=0.99,
		replay_start_size=params['replay_start_size'],
		gpu=gpu,
		minibatch_size=params['batch_size'],
		burnin_action_func=burnin_action_func,
		entropy_target=-action_size,
		temperature_optimizer_lr=3e-4,
	)
	return agent
