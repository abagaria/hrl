import signal
import warnings
from multiprocessing import Pipe, Process

import pfrl
import numpy as np


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
				remote.send((None, None, None, None))
			else:
				raise NotImplementedError
	finally:
		env.close()


class SyncVectorEnv(pfrl.envs.MultiprocessVectorEnv):
	"""
	This VectorEnv supports the different parallel envs sync at the end of an episode run
	only function different from pfrl.envs.MultiprocessVectorEnv is self.step()
	"""
	def __init__(self, env_fns):
		"""
		the __init__ method is the same as the one if pfrl.envs.MultiprocessVectorEnv
		this is re-implemented because the worker() method is redefined above
		"""
		if np.__version__ == "1.16.0":
			warnings.warn(
				"""
				NumPy 1.16.0 can cause severe memory leak in pfrl.envs.MultiprocessVectorEnv.
				We recommend using other versions of NumPy.
				See https://github.com/numpy/numpy/issues/12793 for details.
				"""
			)  # NOQA

		nenvs = len(env_fns)
		self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
		self.ps = [
			Process(target=worker, args=(work_remote, env_fn))
			for (work_remote, env_fn) in zip(self.work_remotes, env_fns)
		]
		for p in self.ps:
			p.start()
		self.last_obs = [None] * self.num_envs
		self.remotes[0].send(("get_spaces", None))
		self.action_space, self.observation_space = self.remotes[0].recv()
		self.closed = False

	def step(self, actions):
		self._assert_not_closed()
		for remote, action in zip(self.remotes, actions):
			if action is not None:  # action is not None
				remote.send(("step", action))
			else:
				remote.send(("done", None))
		results = [remote.recv() for remote in self.remotes]
		self.last_obs, rews, dones, infos = zip(*results)
		return self.last_obs, rews, dones, infos
