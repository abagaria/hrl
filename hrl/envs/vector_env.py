import signal

import pfrl


def worker(remote, env_fn):
	# Ignore CTRL+C in the worker process
	signal.signal(signal.SIGINT, signal.SIG_IGN)
	env = env_fn()
	try:
		while True:
			cmd, data = remote.recv()
			if cmd == "step":
				ob, reward, done, info = env.step(data)
				remote.send((ob, reward, done, info))
			elif cmd == "reset":
				ob = env.reset()
				remote.send(ob)
			elif cmd == "close":
				remote.close()
				break
			elif cmd == "get_spaces":
				remote.send((env.action_space, env.observation_space))
			elif cmd == "spec":
				remote.send(env.spec)
			elif cmd == "seed":
				remote.send(env.seed(data))
			elif cmd == "done":  # the environment should be closed
				remote.send(None)
			else:
				raise NotImplementedError
	finally:
		env.close()


class SyncVectorEnv(pfrl.envs.MultiprocessVectorEnv):
	"""
	This VectorEnv supports the different parallel envs sync at the end of an episode run
	only function different from pfrl.envs.MultiprocessVectorEnv is self.step()
	"""
	def step(self, actions):
		self._assert_not_closed()
		for remote, action in zip(self.remotes, actions):
			if action is not None:  # action is not None
				remote.send(("step", action))
			else:
				remote.send(("done", None))
		results = [remote.recv() for remote in self.remotes]
		results = list(filter(lambda x: x is not None, results))
		self.last_obs, rews, dones, infos = zip(*results)
		return self.last_obs, rews, dones, infos
