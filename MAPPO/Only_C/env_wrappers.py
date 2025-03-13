"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
import torch
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs):
        self.num_envs = num_envs
        # self.observation_space = observation_space
        # self.share_observation_space = share_observation_space
        # self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs_n, obs_mask_n, \
                share_obs, \
                    done_n, creward_n, rreward_n, cact_n, ract_n, \
                        activate_agent_ci, activate_to_cact, activate_agent_ri, activate_to_ract \
                            = env.step(data)
            remote.send((
                obs_n, obs_mask_n, \
                    share_obs, \
                        done_n, creward_n, rreward_n, cact_n, ract_n, \
                            activate_agent_ci, activate_to_cact, activate_agent_ri, activate_to_ract
                    ))
        elif cmd == 'reset':
            env.reset()
        elif cmd == 'render':
            env.render()
        elif cmd == 'is_finished':
            is_f = (env.agents_active == [])
            remote.send(is_f)
        elif cmd == 'reset_process':
            env.reset()
            obs_n, obs_mask_n, \
                share_obs, \
                    done_n, creward_n, rreward_n, cact_n, ract_n, \
                        activate_agent_ci, activate_to_cact, activate_agent_ri, activate_to_ract \
                            = env.step(data)
            remote.send((
                obs_n, obs_mask_n, \
                    share_obs, \
                        done_n, creward_n, rreward_n, cact_n, ract_n, \
                            activate_agent_ci, activate_to_cact, activate_agent_ri, activate_to_ract
                    ))
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((0))
        else:
            raise NotImplementedError

class SubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, agent_num):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(
                target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn))
                )
                for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
            ]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        temp = self.remotes[0].recv()
        self.default_action = (np.zeros(agent_num), np.zeros(agent_num))
        ShareVecEnv.__init__(self, len(env_fns))

    def step_async(self, actions):
        cactions = actions[0]
        ractions = actions[1]
        for (remote, caction, raction) in zip(self.remotes, cactions, ractions):
            remote.send(('step', (caction, raction)))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs_n, obs_mask_n, \
            share_obs, \
                done_n, creward_n, rreward_n, cact_n, ract_n, \
                    activate_agent_ci, activate_to_cact, \
                        activate_agent_ri, activate_to_ract = zip(*results)
        return np.stack(obs_n), np.stack(obs_mask_n), \
                np.stack(share_obs), \
                    np.stack(done_n), np.stack(creward_n), np.stack(rreward_n), np.stack(cact_n), np.stack(ract_n), \
                        list(activate_agent_ci), list(activate_to_cact), \
                            list(activate_agent_ri), list(activate_to_ract)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))

    def is_finished(self):
        for remote in self.remotes:
            remote.send(('is_finished', None))
        is_f = [remote.recv() for remote in self.remotes]
        index = []
        for i in range(len(is_f)):
            if is_f[i] == True:
                index.append(i)
        return index

    def reset_process(self, index):
        for i in index:
            self.remotes[i].send(('reset_process', self.default_action))
        results = []
        for i in index: 
            results.append(self.remotes[i].recv())
        obs_n, obs_mask_n, \
            share_obs, \
                done_n, creward_n, rreward_n, cact_n, ract_n, \
                    activate_agent_ci, activate_to_cact, \
                        activate_agent_ri, activate_to_ract = zip(*results)
        return np.stack(obs_n), np.stack(obs_mask_n), \
                np.stack(share_obs), \
                    np.stack(done_n), np.stack(creward_n), np.stack(rreward_n), np.stack(cact_n), np.stack(ract_n), \
                        list(activate_agent_ci), list(activate_to_cact), \
                            list(activate_agent_ri), list(activate_to_ract)
    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, i):
        self.remotes[i].send(('render', None))

# single env
class DummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns, agent_num):
        self.envs = [fn() for fn in env_fns]
        # self.env = self.envs[0]
        self.default_action = (np.zeros(agent_num), np.zeros(agent_num))
        ShareVecEnv.__init__(self, len(env_fns))
        self.cactions = None
        self.ractions = None

    def step_async(self, actions):
        self.cactions = actions[0]
        self.ractions = actions[1]

    def step_wait(self):
        results = [env.step((ca, ra)) for (ca, ra, env) in zip(self.cactions, self.ractions, self.envs)] # type: ignore
        obs_n, obs_mask_n, \
            share_obs, \
                done_n, creward_n, rreward_n, cact_n, ract_n, \
                    activate_agent_ci, activate_to_cact, \
                        activate_agent_ri, activate_to_ract = zip(*results)
        self.cactions = None
        self.ractions = None
        return np.stack(obs_n), np.stack(obs_mask_n), \
                np.stack(share_obs), \
                    np.stack(done_n), np.stack(creward_n), np.stack(rreward_n), np.stack(cact_n), np.stack(ract_n), \
                        list(activate_agent_ci), list(activate_to_cact), \
                            list(activate_agent_ri), list(activate_to_ract)

    def is_finished(self):
        index = []
        for i, env in enumerate(self.envs):
            if env.agents_active == []:
                index.append(i)
        return index

    def reset_process(self, index):
        for i in index:
            self.envs[i].reset()
            
        results = [self.envs[i].step(self.default_action) for i in index] # type: ignore
        obs_n, obs_mask_n, \
            share_obs, \
                done_n, creward_n, rreward_n, cact_n, ract_n, \
                    activate_agent_ci, activate_to_cact, \
                        activate_agent_ri, activate_to_ract = zip(*results)
        return np.stack(obs_n), np.stack(obs_mask_n), \
                np.stack(share_obs), \
                    np.stack(done_n), np.stack(creward_n), np.stack(rreward_n), np.stack(cact_n), np.stack(ract_n), \
                        list(activate_agent_ci), list(activate_to_cact), \
                            list(activate_agent_ri), list(activate_to_ract)
    
    def reset(self):
        for env in self.envs:
            env.reset()

    def close(self):
        for env in self.envs:
            env.close()

    def render(self):
        for env in self.envs:
            env.render()
