import sys
import os

from collections import namedtuple
import json, pickle
import numpy as np
from typing import List

class HalfCheetahDirEnv():
    def __init__(self, tasks: List[dict], include_goal: bool = False):
        if tasks is None:
            tasks = [{'direction': 1}, {'direction': -1}]
        self.tasks = tasks
        self._max_episode_steps = 200

class HalfCheetahVelEnv():
    def __init__(self, tasks: List[dict], include_goal: bool = False, one_hot_goal: bool = False, n_tasks: int = None):
        if tasks is None:
            tasks = [{'direction': 1}, {'direction': -1}]
        self.tasks = tasks
        self._max_episode_steps = 200

class AntDirEnv():
    def __init__(self, tasks: List[dict], n_tasks: int = None, include_goal: bool = False):
        if tasks is None:
            tasks = [{'direction': 1}, {'direction': -1}]
        self.tasks = tasks
        self._max_episode_steps = 200
