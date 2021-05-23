from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import numpy as nu
import pandas as pd
import datetime as dt
import multiprocessing as mp
from matplotlib import pyplot as pl
import pickle
import sys
import os
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
from tensorflow import keras as kr
from tf_agents.agents.sac import sac_agent
from tf_agents.agents import DdpgAgent
from tf_agents.agents.reinforce.reinforce_agent import ReinforceAgent
from tf_agents.networks import encoding_network, utils
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.networks.network import Network
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.train.utils import spec_utils, strategy_utils, train_utils
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.policies import greedy_policy, random_tf_policy, policy_saver
from tf_agents.utils import common, nest_utils
import shutil
import copy
from typing import Tuple
# Custom Modules
from CarRacingEnvClass import CarRacingEnv


if __name__ == '__main__':
    mp.freeze_support()
    # check if should continue from last stored checkpoint
    if len(sys.argv) == 2:
        shouldContinueFromLastCheckpoint = sys.argv[1] == '-c'
    else:
        shouldContinueFromLastCheckpoint = False
    # Environment

    # create environment and transfer it to Tensorflow version
    print('Creating environment ...')
    gamma = 0.999
    env_name = "CarRacing-v0"
    env = suite_gym.load(env_name)
    # env = CarRacingEnv(gamma=gamma)
    env.reset()
    env = tf_py_environment.TFPyEnvironment(env)

    evaluate_env = suite_gym.load(env_name)
    # evaluate_env = CarRacingEnv(gamma=gamma)
    evaluate_env.reset()
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    print('Environment created.')
    print(f"observation_spec: {observation_spec}")
    print(f"action_spec: {action_spec}")
    print(f"time_spec: {env.time_step_spec()}")

    # Hyperparameters
    num_iterations = int(1e5) # @param {type:"number"}
    collect_episodes_per_iteration = int(5)  # @param {type:"integer"}
    _storeFullEpisodes = collect_episodes_per_iteration  # @param {type:"integer"}
    # replayBufferCapacity = int(_storeFullEpisodes * episodeEndSteps * batchSize)
    replayBufferCapacity = int(100000)  # @param {type:"integer"}

    # observationConvParams = [(int(observation_spec['observation_market'].shape[0]//100), 3, 1)]


    learning_rate = 1e-4 # @param {type:"number"}
    entropy_coeff = 0.2
    log_interval = num_iterations//1000
    eval_interval = num_iterations//100
    validateEpisodes = 10

    checkpointDir = './SACAgent_checkcpoints'
    if not os.path.exists(checkpointDir):
        os.mkdir(checkpointDir)

    policyDir = './SACAgent_savedPolicy'
    if not os.path.exists(policyDir):
        os.mkdir(policyDir)


    # Actor Network
    # A2C or REINFORCE Agent has an actor giving the distribution (or the mean and variance) of actions,
    # so an actorDistributionNetwork is needed, not ones who directly return actions.
    actor_denseLayerParams = (512, 9)
    # actor_convLayerParams = [(tf.cast(64, tf.int32), tf.cast(16, tf.int32), tf.cast(5, tf.int32)), (tf.cast(64, tf.int32), tf.cast(16, tf.int32), tf.cast(5, tf.int32))]
    actor_convLayerParams = critic_observationConvLayerParams = [(int(32), int(8), int(4)), (int(64), int(4), int(2)), (int(64), int(3), int(1))]
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec,
        preprocessing_layers=kr.models.Sequential([
            kr.layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4), activation='relu', input_shape=(1, 96, 96, 3)),
            kr.layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2), activation='relu', input_shape=(1, 96, 96, 3)),
            kr.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu', input_shape=(1, 96, 96, 3)),
            kr.layers.Flatten(),
            kr.layers.Dense(5, activation='tanh'),
        ]),
        conv_layer_params=None,
        fc_layer_params=actor_denseLayerParams,
        continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork,
        name='ActorDistributionNetwork'
    )
    print('Actor Network Created.')


    # Critic Network
    # critic_net = ValueNetwork(
    #     observation_spec,
    #     preprocessing_layers={
    #         'observation_market': kr.models.Sequential([
    #             kr.layers.Conv2D(filters=int((observation_spec['observation_market'].shape[0]*observation_spec['observation_market'].shape[1])//100), kernel_size=3, activation='relu', input_shape=(observation_spec['observation_market'].shape[0], observation_spec['observation_market'].shape[1], 1)),
    #             kr.layers.Conv2D(filters=int((observation_spec['observation_market'].shape[0]*observation_spec['observation_market'].shape[1])//100), kernel_size=3, activation='relu', input_shape=(observation_spec['observation_market'].shape[0], observation_spec['observation_market'].shape[1], 1)),
    #             kr.layers.Flatten(),
    #             kr.layers.Dense(5, activation='tanh'),
    #             kr.layers.Flatten()
    #         ]),
    #         # 'observation_market': kr.layers.Conv2D(filters=int((observation_spec['observation_market'].shape[0]*observation_spec['observation_market'].shape[1])//100), kernel_size=3, activation='relu', input_shape=(observation_spec['observation_market'].shape[0], observation_spec['observation_market'].shape[1], 1)),
    #         'observation_holdingRate': kr.layers.Dense(2, activation='tanh')
    #     },
    #     preprocessing_combiner=kr.layers.Concatenate(axis=-1),
    #     conv_layer_params=None,
    #     fc_layer_params=critic_commonDenseLayerParams,
    #     dtype=tf.float32,
    #     name='Critic Network'
    # )
    critic_observationConvLayerParams = [(int(32), int(8), int(4)), (int(64), int(4), int(2)), (int(64), int(3), int(1))]
    critic_observationDenseLayerParams = (512, 9)
    critic_commonDenseLayerParams = (64, 8)
    critic_actionDenseLayerParams = (int(8),)
    critic_net = CriticNetwork(
        (observation_spec, action_spec),
        observation_conv_layer_params=critic_observationConvLayerParams,
        observation_fc_layer_params=critic_observationDenseLayerParams,
        action_fc_layer_params=critic_actionDenseLayerParams,
        joint_fc_layer_params=critic_commonDenseLayerParams,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform'
    )


    # Agent
    # https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/ReinforceAgent
    global_step = tf.compat.v1.train.get_or_create_global_step()
    if shouldContinueFromLastCheckpoint:
        global_step = tf.compat.v1.train.get_global_step()
    criticLearningRate = 1e-4
    actorLearningRate = 1e-4
    alphaLearningRate = 1e-4
    gradientClipping = None
    target_update_tau = 1e-4
    tf_agent = sac_agent.SacAgent(
        env.time_step_spec(),
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=actorLearningRate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=criticLearningRate),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=alphaLearningRate),
        target_update_tau=target_update_tau,
        gamma=gamma,
        gradient_clipping=gradientClipping,
        train_step_counter=global_step,
    )
    tf_agent.initialize()
    print('SAC Agent Created.')

    # Policies
    evaluate_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    collect_policy = tf_agent.collect_policy


    # Evaluation
    def compute_avg_return(environment, policy, num_episodes=5):
        total_return = 0.0
        for _ in range(num_episodes):
            time_step = environment.reset()
            episode_return = 0.0
            while not time_step.is_last():
                action_step = policy.action(time_step)
                action = [ float(a) for a in action_step.action ]
                time_step = environment.step(action)
                episode_return += time_step.reward
            total_return += float(episode_return)
        avg_return = total_return / num_episodes
        return avg_return

    # Replay Buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=env.batch_size,
        max_length=replayBufferCapacity
    )
    print('Replay Buffer Created, start warming-up ...')
    _startTime = dt.datetime.now()

    # Drivers
    initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        env,
        collect_policy,
        observers=[replay_buffer.add_batch],
        num_episodes=_storeFullEpisodes
    )
    initial_collect_driver.run()
    _timeCost = (dt.datetime.now() - _startTime).total_seconds()
    print('Replay Buffer Warm-up Done. (cost {:.3g} hours)'.format(_timeCost/3600.0))
    _startTime = dt.datetime.now()

    # run restore process
    if shouldContinueFromLastCheckpoint:
        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpointDir,
            max_to_keep=1,
            agent=tf_agent,
            policy=tf_agent.policy,
            replay_buffer=replay_buffer,
            global_step=global_step
        )
        train_checkpointer.initialize_or_restore()

    print('Prepare for training ...')
    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        env,
        collect_policy,
        observers=[replay_buffer.add_batch],
        num_episodes=collect_episodes_per_iteration
    )
    tf_agent.train_step_counter.assign(0)

    # Initialize avg_return
    # avg_return = compute_avg_return(evaluate_env, evaluate_policy, 1)
    # returns = [avg_return]

    # Training
    dataset = replay_buffer.as_dataset(num_parallel_calls=4, sample_batch_size=env.batch_size, num_steps=2)
    iterator = iter(dataset)
    returns = nu.array([])
    steps = nu.array([])
    losses = nu.array([])
    _timeCost = (dt.datetime.now() - _startTime).total_seconds()
    print('All preparation is done (cost {:.3g} hours). Start training...'.format(_timeCost/3600.0))
    _startTimeFromStart = dt.datetime.now()
    compute_avg_return(evaluate_env, evaluate_policy, validateEpisodes)
    for _ in range(num_iterations):
        _startTime = dt.datetime.now()
        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_driver.run()
        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = tf_agent.train(experience)
        step = tf_agent.train_step_counter.numpy()
        # print time cost and loss
        if step % log_interval == 0:
            _timeCost = (dt.datetime.now() - _startTime).total_seconds()
            _timeCostFromStart = (dt.datetime.now() - _startTimeFromStart).total_seconds()
            if _timeCost <= 60:
                print('step = {:>5}: loss = {:+10.6f}  (cost {:>5.2f} [sec]; {:>.2f} [hrs] from start.)'.format(step, train_loss.loss, _timeCost, _timeCostFromStart/3600.0))
            elif _timeCost <= 3600:
                print('step = {:>5}: loss = {:+10.6f}  (cost {:>5.2f} [min]; {:>.2f} [hrs] from start.)'.format(step, train_loss.loss, _timeCost/60.0, _timeCostFromStart/3600.0))
            else:
                print('step = {:>5}: loss = {:+10.6f}  (cost {:>5.2f} [hrs]; {:>.2f} [hrs] from start.)'.format(step, train_loss.loss, _timeCost/3600.0, _timeCostFromStart/3600.0))
        # evaluate policy and show average return
        if step % eval_interval == 0:
            avg_return = compute_avg_return(evaluate_env, evaluate_policy, validateEpisodes)
            print('step = {:>5}: Average Return = {}'.format(step, avg_return))
            steps = nu.append(steps, step)
            returns = nu.append(returns, avg_return)
            losses = nu.append(losses, train_loss.loss)
            # save temp results
            with open('SACAgent_tempResults.pickle', 'wb') as file:
                pickle.dump(nu.concatenate([steps.reshape(-1, 1), returns.reshape(-1, 1), losses.reshape(-1, 1)], axis=-1), file)
            # # save models
            # # a checkpoint of a agent model can be used to restart a training
            # # https://www.tensorflow.org/agents/tutorials/10_checkpointer_policysaver_tutorial?hl=en
            # train_checkpointer = common.Checkpointer(
            #     ckpt_dir=checkpointDir,
            #     max_to_keep=1,
            #     agent=tf_agent,
            #     policy=tf_agent.policy,
            #     replay_buffer=replay_buffer,
            #     global_step=global_step
            # )
            # train_checkpointer.save(global_step)
            # # save policy
            # # saved policies can only be used to evaluate, not to train.
            # tf_policy_saver = policy_saver.PolicySaver(tf_agent.policy)
            # tf_policy_saver.save(policy_dir)
    # save results
    with open('SACAgent_results.pickle', 'wb') as file:
        pickle.dump(nu.concatenate([steps.reshape(-1, 1), returns.reshape(-1, 1), losses.reshape(-1, 1)], axis=-1), file)
    # save models
    # a checkpoint of a agent model can be used to restart a training
    # https://www.tensorflow.org/agents/tutorials/10_checkpointer_policysaver_tutorial?hl=en
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpointDir,
        max_to_keep=1,
        agent=tf_agent,
        policy=tf_agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )
    train_checkpointer.save(global_step)
    # # save policy
    # # saved policies can only be used to evaluate, not to train.
    # tf_policy_saver = policy_saver.PolicySaver(tf_agent.policy)
    # tf_policy_saver.save(policy_dir)
    # plot
    pl.xlabel('Step', fontsize=22)
    pl.ylabel('Returns', fontsize=22)
    pl.tick_params(labelsize=16)
    pl.plot(steps, returns)
    pl.show()

    pl.xlabel('Step', fontsize=22)
    pl.ylabel('Loss', fontsize=22)
    pl.tick_params(labelsize=16)
    pl.plot(steps, losses)
    pl.show()
