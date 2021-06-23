# Copyright (C) 2021 Bosch LLC CR, North America. All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import numpy as np
import os
from gym import spaces
from stable_baselines3.common.vec_env import VecEnv

class GymIgnitionVecEnv(VecEnv):
    def __init__(self, impl, clip_obs=10.0):
        self.wrapper = impl
        self.wrapper.init()
        
        self.num_obs = self.wrapper.getObsDim()
        self.num_acts = self.wrapper.getActionDim()
        self.num_extras = self.wrapper.getExtraInfoDim()

        self._observation_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self._action_space = spaces.Box(np.ones(self.num_acts) * -1., np.ones(self.num_acts) * 1., dtype=np.float32)

        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._extraInfo = np.zeros([self.num_envs, len(self._extraInfoNames)], dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros((self.num_envs), dtype=np.bool)

        self.clip_obs = clip_obs

    # TODO: set the random seed
    # def seed(self, seed=None):
    #     self.wrapper.setSeed(seed)

    def step(self, action, visualize=False):
        if not visualize:
            self.wrapper.step(action, self._observation, self._reward, self._done)
        # else:
        #   only run the first environment here to visualize the policy

        return self._reward.copy(), self._done.copy()

    def observe(self):
        self.wrapper.observe(self._observation)

        return self._observation.copy()

    def get_extras(self):
        self.wrapper.getExtraInfo(self._extraInfo)

        return self._extraInfo.copy()

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=np.bool)

        self.wrapper.reset(self._observation)

    def render(self, mode='human'):
        raise RuntimeError('This method is not implemented')

    def close(self):
        self.wrapper.close()

    def step_async(self):
        raise RuntimeError('This method is not implemented')

    def step_wait(self):
        raise RuntimeError('This method is not implemented')

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.
        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        raise RuntimeError('This method is not implemented')

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.
        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise RuntimeError('This method is not implemented')

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.
        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        raise RuntimeError('This method is not implemented')

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def num_obs(self):
        return self.num_obs

    @property
    def num_acts(self):
        return self.num_acts

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space