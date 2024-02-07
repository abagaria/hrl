from hrl.agent.td3.TD3AgentClass import TD3
from hrl import utils


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
