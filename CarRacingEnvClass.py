from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import tensorflow as tf
import numpy as nu
import pandas as pd
import datetime as dt
import os
import multiprocessing as mp
# tensorflows
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
tf.compat.v1.enable_v2_behavior()
import gym


class CarRacingEnv(py_environment.PyEnvironment):
    def __init__(self, gamma, dtype=nu.float32):
        env_name = "CarRacing-v0"
        # self.__env = suite_gym.load(env_name)
        self.__env = gym.make(env_name)
        self.dtype = dtype
        self.gamma = float(gamma)

        self.__observationSpec = BoundedArraySpec(shape=(96, 96, 3), dtype=nu.int32, name='observation', minimum=0, maximum=255)
        self.__actionSpec = BoundedArraySpec(shape=(3,), dtype=dtype, name='action', minimum=nu.array([-1.,  0.,  0.]), maximum=nu.array([1., 1., 1.]))
        # self.__observationSpec = tf.cast(self.__env.observation_spec(), dtype=tf.int32)
        self.__accumulatedReward = 0.0


    @classmethod
    def castObservation(cls, observation, dtype=nu.int32):
        return observation.astype(dtype)


    # required
    def action_spec(self):
        return self.__actionSpec


    # required
    def observation_spec(self):
        return self.__observationSpec


    # required
    def _reset(self):
        self.__accumulatedReward = 0.0
        observation = self.__env.reset()
        return ts.restart(CarRacingEnv.castObservation(observation))


    # required
    def _step(self, action):
        observation, reward, done, info = self.__env.step(action)
        castedObservation = CarRacingEnv.castObservation(observation)
        done = done or self.__accumulatedReward < -1000

        if done is True:
            return ts.termination(castedObservation, 0.0)
        else:
            self.__accumulatedReward += reward
            return ts.transition(castedObservation, reward=reward, discount=self.gamma)
