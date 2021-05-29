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
from matplotlib import pyplot as pl
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
import cv2
from time import sleep


class CarRacingEnv(py_environment.PyEnvironment):
    def __init__(self, gamma, shouldAdjustDoneDetermination=True, dtype=nu.float32):
        env_name = "CarRacing-v0"
        # self.__env = suite_gym.load(env_name)
        self.__env = gym.make(env_name)
        self.dtype = dtype
        self.gamma = float(gamma)
        self.shouldAdjustDoneDetermination = shouldAdjustDoneDetermination

        self.__observationSpec = BoundedArraySpec(shape=(96, 96, 1), dtype=nu.int32, name='observation', minimum=0, maximum=255)
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
        for _ in range(100):
            observation, _, _, _ = self.__env.step(nu.zeros(3))
        grayObservation = CarRacingEnv.adjustObservationToGrayScale(observation)
        return ts.restart(grayObservation)


    # required
    def _step(self, action):
        observation, reward, done, info = self.__env.step(action)
        grayObservation = CarRacingEnv.adjustObservationToGrayScale(observation)
        isOutOfCourse = False
        if self.shouldAdjustDoneDetermination:
            isOutOfCourse = grayObservation[71, 50, 0] < 200 or grayObservation[71, 45, 0] < 200  or done
            done = isOutOfCourse or self.__accumulatedReward < -200

        if done:
            return ts.termination(grayObservation, -200.0 if isOutOfCourse else 0.0)
        else:
            self.__accumulatedReward += reward
            return ts.transition(grayObservation, reward=reward, discount=self.gamma)


    @classmethod
    def adjustObservationToGrayScale(cls, observation):
        grayObservation = nu.zeros((96, 96), dtype=nu.int32)
        roadIndex = observation[:, :, 1] < 200
        grayObservation[roadIndex] = 255
        return grayObservation[:, :, nu.newaxis]




if __name__ == '__main__':
    env = CarRacingEnv(gamma=0.99)
    # env = gym.make("CarRacing-v0")

    env.reset()
    # for _ in range(100):
    #     _, _, _, _ = env.step(nu.zeros(3))

    # observation, _, _, _ = env.step(nu.zeros(3))
    # img = cv2.imread(observation, cv2.IMREAD_COLOR)
    # cv2.imshow('image',observation)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    for i in range(500):
        # action = env.action_space.sample()                  # 行動の決定
        action = nu.array([0, 0.5, 0])
        timeStep = env.step(action)  # 行動による次の状態の決
        grayObservation, isDone = timeStep.observation, timeStep.step_type

        if isDone == 2:
            env.reset()
            continue

        # ob, _, _, _ = env.step(action)
        # grayObservation = CarRacingEnv.adjustObservationToGrayScale(ob)[:, :, 0]
        # grayObservation[71, 50] = 255
        # grayObservation[71, 45] = 255

        if i % 50 != 0:
            continue

        pl.imshow(grayObservation, cmap='Greys', interpolation='bilinear')
        pl.show()
