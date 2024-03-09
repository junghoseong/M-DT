import gym
import numpy as np
import datetime
import os
import time
import pytz
from filelock import FileLock
from typing import Tuple, Dict
import pymap3d as pm

class AirCombatEnvironment(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=[69], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=[4], dtype=np.float32)
        self._info = {}
        self.win = 1

    def __del__(self):
        print("JSBSim destroy")

    def reset(self, logging:bool = False, log_name:str = None, seed:int = None):
        if seed is not None:
            np.random.seed(seed=seed)
        cur_obs = self.get_observation()
        return np.array(cur_obs)

    def get_observation(self):
        obs = np.zeros(self.observation_space.shape[0])
        return np.array(obs)

    def get_reward(self):
        reward = 0
        return reward

    def get_done(self):
        done = True
        return done

    def get_win(self):
        return self.win

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        cur_obs = self.get_observation()
        reward = self.get_reward()
        done = self.get_done()
        return np.array(cur_obs), reward, done, self._info