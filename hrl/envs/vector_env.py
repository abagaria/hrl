import warnings
from multiprocessing import Pipe, Process

import pfrl
import numpy as np

from hrl.utils import StopExecution
from hrl.utils import filter_token


def worker(remote, env_fn):
    # Ignore CTRL+C in the worker process
    # signal.signal(signal.SIGINT, signal.SIG_IGN)
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            # operations on env
            if cmd == "step":
                ob, reward, done, info = env.step(data)
                remote.send((ob, reward, done, info))
            elif cmd == "reset":
                ob = env.reset(**data)
                remote.send(ob)
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "done":  # the environment should be closed
                remote.send((StopExecution, StopExecution, StopExecution, StopExecution))
            # get infos
            elif cmd == "get_spaces":
                remote.send((env.action_space, env.observation_space))
            elif cmd == "spec":
                remote.send(env.spec)
            else:
                raise NotImplementedError
    finally:
        env.close()


class EpisodicSyncVectorEnv(pfrl.envs.MultiprocessVectorEnv):
    """
    This VectorEnv supports the different parallel envs sync at the end of an episode run
    This is used for when each of the parallel env is run by episodes, instead of total number
    of step (for running total number of step, just use pfrl.envs.MultiprocessVectorEnv)

    when a certain environment has finished running the current epsiode, env.step()
    will return a StopExecution toekn, which will be used in the training loop to
    indicate that the environment has finished. 
    It is up to the training loop what to do with the StopExecution token.
    """
    def __init__(self, env_fns, max_episode_len=None):
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

        self.nenvs = len(env_fns)
        self.max_episode_len = max_episode_len
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])
        self.ps = [
            Process(target=worker, args=(work_remote, env_fn))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)
        ]
        for p in self.ps:
            p.start()
        self.last_obs = [None] * self.num_envs
        # get info from indivisual env
        self.remotes[0].send(("get_spaces", None))
        self.action_space, self.observation_space = self.remotes[0].recv()
        dummy_env = env_fns[0]()
        self.reward_func = dummy_env.reward_func
        self.get_position = dummy_env.get_position
        self.closed = False

        # keep track of episodes
        self.episode_dones = np.zeros(self.nenvs, dtype="i")  # whether the corresponding env is done with the cur episode
        self.episode_lens = np.zeros(self.nenvs, dtype="i")  # the idx of step each env is on for their cur episode 
        self.episode_rs = np.zeros(self.nenvs, dtype=np.float32)  # reward for the current episode
        self.episode_successes = np.zeros(self.nenvs, dtype="i")
    
    @property
    def all_envs_done(self):
        """
        whether all envs has finished running the current episode
        """
        return np.all(self.episode_dones)
    
    @property
    def n_done_envs(self):
        """
        the number of parallel envs that have finished executing
        """
        return np.sum(self.episode_dones)

    def step(self, actions):
        """
        the results returned by this function may be lists that contain StopExecution
        """
        self._assert_not_closed()
        # note that actions might not be a list of length self.nenvs
        # first need to mask actions: done_envs has action StopExecution
        action_iter = iter(actions)
        actions = [StopExecution if self.episode_dones[idx_env] else next(action_iter) for idx_env in range(self.nenvs)]
        assert len(actions) == self.nenvs

        # send each action to remote workers and receive results
        for remote, action in zip(self.remotes, actions):
            if action is not StopExecution:
                remote.send(("step", action))
            else:
                remote.send(("done", None))
        results = [remote.recv() for remote in self.remotes]
        self.last_obs, rews, dones, infos = zip(*results)

        # update the counters
        episode_not_done = np.logical_not(self.episode_dones)
        # 1. r
        zeroed_rs = filter_token(rews, replace_with=np.array(0))  # change StopExecution to 0
        self.episode_rs += zeroed_rs * episode_not_done
        # 2. len
        self.episode_lens += 1 * episode_not_done
        # 3. dones
        terminal = filter_token(dones, replace_with=True)  # change StopExecution to True
        if self.max_episode_len is None:
            needs_resets = np.zeros(self.nenvs, dtype=bool)
        else:
            needs_resets = self.episode_lens == self.max_episode_len
        self.episode_dones = np.logical_or(self.episode_dones, terminal)
        self.episode_dones = np.logical_or(self.episode_dones, needs_resets)
        # 4. success
        self.episode_successes = np.logical_or(self.episode_successes, terminal)

        return self.last_obs, rews, dones, infos

    def reset(self, mask=None, **kwargs):
        self._assert_not_closed()
        if mask is None:
            mask = np.zeros(self.num_envs)
        for m, remote in zip(mask, self.remotes):
            if not m:
                remote.send(("reset", kwargs))

        obs = [
            remote.recv() if not m else o
            for m, remote, o in zip(mask, self.remotes, self.last_obs)
        ]
        self.last_obs = obs

        # reset the counters
        self.episode_dones = np.zeros(self.nenvs, dtype="i")
        self.episode_lens = np.zeros(self.nenvs, dtype="i")
        self.episode_rs = np.zeros(self.nenvs, dtype=np.float32) 
        self.episode_successes = np.zeros(self.nenvs, dtype="i")

        return obs
