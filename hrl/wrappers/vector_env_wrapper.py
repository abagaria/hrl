class VectorEnvWrapper:
	"""
	this wrapper is designed to expose the API of the base_env of the MultiprocessVectorEnv
	"""
	def __init__(self, env):
		self.env = env.env

	# implicitly forward all other methods and attributes to self.env
	def __getattr__(self, name):
		# if name.startswith("_"):
		# 	raise AttributeError(
		# 		"attempted to get missing private attribute '{}'".format(name)
		# 	)
		return getattr(self.env, name)

	def __repr__(self):
		return "<{}, {}>".format(self.__class__.__name__, self.env)
