import numpy as nu
import pandas as pd
import gym
from matplotlib import pyplot as pl

# https://gym.openai.com/envs/CarRacing-v0/
# https://vigne-cla.com/9-19/#toc11
# https://qiita.com/nsd24/items/7758410128872d684e05#carracing-v0
# https://lib-arts.hatenablog.com/entry/rl_trend5


env = gym.make("CarRacing-v0")                          # GUI環境の開始(***)

for episode in range(20):
  observation = env.reset()                             # 環境の初期化
  for _ in range(5000):
    env.render()                                        # レンダリング(画面の描画)
    action = env.action_space.sample()                  # 行動の決定
    observation, reward, done, info = env.step(action)  # 行動による次の状態の決定
    # print("=" * 10)
    # print('action = {:.5f}  reward={:.1f}'.format(action, reward))
    print("action = {:+.5f} {:+.5f} {:+.5f}, reward = {:.8f}".format(action[0], action[1], action[2], reward))
    # print("observation=",observation)
    # print("reward=",reward)
    # print("done=",done)
    # print("info=",info)

env.close()                                             # GUI環境の終了



# env = gym.make('CarRacing-v0')
# observation = env.reset()
# for t in range(1000):
#     env.render()
#     observation, reward, done, info = env.step(env.action_space.sample())
# env.close()
