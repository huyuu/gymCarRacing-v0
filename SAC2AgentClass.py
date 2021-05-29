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
from tf_agents.policies import greedy_policy, random_tf_policy, policy_saver, py_tf_eager_policy
from tf_agents.utils import common, nest_utils
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
import shutil
import copy
from typing import Tuple
import tempfile
from reverb import Server
# Custom Modules
from CarRacingEnvClass import CarRacingEnv


if __name__ == '__main__':
    mp.freeze_support()
    # check if should continue from last stored checkpoint
    if len(sys.argv) == 2:
        shouldContinueFromLastCheckpoint = sys.argv[1] == '-c'
    else:
        shouldContinueFromLastCheckpoint = False


    # Hyperparameters

    tempdir = tempfile.gettempdir()

    # Use "num_iterations = 1e6" for better results (2 hrs)
    # 1e5 is just so this doesn't take too long (1 hr)
    num_iterations = int(1e3) # @param {type:"integer"}

    collect_steps_per_iteration = 3 # @param {type:"integer"}
    initial_collect_episodes = collect_steps_per_iteration # @param {type:"integer"}
    replay_buffer_capacity = int(1e5) # @param {type:"integer"}

    batch_size = 256 # @param {type:"integer"}

    learning_rate = 3e-4 # @param {type:"number"}
    critic_learning_rate = 3e-4 # @param {type:"number"}
    actor_learning_rate = 3e-4 # @param {type:"number"}
    alpha_learning_rate = 3e-4 # @param {type:"number"}
    target_update_tau = 0.005 # @param {type:"number"}
    target_update_period = 1 # @param {type:"number"}
    entropy_coeff = 0.2
    gamma = 0.99 # @param {type:"number"}
    reward_scale_factor = 1.0 # @param {type:"number"}

    actor_fc_layer_params = (256, 256)
    critic_joint_fc_layer_params = (256, 256)

    num_eval_episodes = 5 # @param {type:"integer"}
    log_interval = num_iterations//1000
    eval_interval = num_iterations//100 # @param {type:"integer"}

    policy_save_interval = 5000 # @param {type:"integer"}

    use_gpu = False
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)


    checkpointDir = './SAC2Agent_checkcpoints'
    if not os.path.exists(checkpointDir):
        os.mkdir(checkpointDir)

    policyDir = './SAC2Agent_savedPolicy'
    if not os.path.exists(policyDir):
        os.mkdir(policyDir)


    # Environment

    # create environment and transfer it to Tensorflow version
    print('Creating environment ...')
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
    observation_spec, action_spec, time_step_spec = (spec_utils.get_tensor_specs(env))


    # Actor Network
    # A2C or REINFORCE Agent has an actor giving the distribution (or the mean and variance) of actions,
    # so an actorDistributionNetwork is needed, not ones who directly return actions.
    actor_denseLayerParams = (512, 9)
    # actor_convLayerParams = [(tf.cast(64, tf.int32), tf.cast(16, tf.int32), tf.cast(5, tf.int32)), (tf.cast(64, tf.int32), tf.cast(16, tf.int32), tf.cast(5, tf.int32))]
    actor_convLayerParams = [(int(32), int(8), int(4)), (int(64), int(4), int(2)), (int(64), int(3), int(1))]
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

    with strategy.scope():
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            preprocessing_layers=kr.models.Sequential([
                kr.layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4), activation='relu', input_shape=(1, 96, 96, 3)),
                kr.layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2), activation='relu', input_shape=(1, 96, 96, 3)),
                kr.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu', input_shape=(1, 96, 96, 3)),
                kr.layers.Flatten(),
                kr.layers.Dense(5, activation='tanh'),
            ]),
            conv_layer_params=None,
            fc_layer_params=actor_denseLayerParams,
            continuous_projection_net=(tanh_normal_projection_network.TanhNormalProjectionNetwork)
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
    with strategy.scope():
        critic_net = CriticNetwork(
            (observation_spec, action_spec),
            observation_conv_layer_params=critic_observationConvLayerParams,
            observation_fc_layer_params=critic_observationDenseLayerParams,
            action_fc_layer_params=critic_actionDenseLayerParams,
            joint_fc_layer_params=critic_commonDenseLayerParams,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform')
    print('Critic Network Created.')


    # Agent
    # https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/ReinforceAgent
    global_step = tf.compat.v1.train.get_or_create_global_step()
    if shouldContinueFromLastCheckpoint:
        global_step = tf.compat.v1.train.get_global_step()
    criticLearningRate = 3e-4
    actorLearningRate = 3e-4
    alphaLearningRate = 3e-4
    gradientClipping = None
    target_update_tau = 0.005
    with strategy.scope():
        train_step = train_utils.create_train_step()
        tf_agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=actor_learning_rate),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=critic_learning_rate),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=alpha_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            train_step_counter=train_step
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

    rate_limiter=reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=3.0, min_size_to_sample=3, error_buffer=3.0)
    table_name = 'uniform_table'
    table = reverb.Table(
        table_name,
        max_size=replay_buffer_capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1)
    )
    reverb_server = Server([table])

    reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
        tf_agent.collect_data_spec,
        sequence_length=2,
        table_name=table_name,
        local_server=reverb_server
    )

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        reverb_replay.py_client,
        table_name,
        sequence_length=2,
        stride_length=1
    )
    dataset = reverb_replay.as_dataset(sample_batch_size=batch_size, num_steps=2).prefetch(50)
    experience_dataset_fn = lambda: dataset
    print('Replay Buffer Created, start warming-up ...')


    # Policies

    tf_eval_policy = tf_agent.policy
    eval_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_eval_policy, use_tf_function=True)
    tf_collect_policy = tf_agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_collect_policy, use_tf_function=True)


    # Drivers

    _startTime = dt.datetime.now()
    initial_collect_actor = actor.Actor(
        collect_env,
        random_policy,
        train_step,
        steps_per_run=initial_collect_steps,
        observers=[rb_observer]
    )
    initial_collect_actor.run()

    initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        env,
        collect_policy,
        observers=[replay_buffer.add_batch],
        num_episodes=_storeFullEpisodes
    )
    initial_collect_driver.run()
    _timeCost = (dt.datetime.now() - _startTime).total_seconds()
    print('Replay Buffer Warm-up Done. (cost {:.3g} hours)'.format(_timeCost/3600.0))


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


    # Collect Driver

    # training driver
    env_step_metric = py_metrics.EnvironmentSteps()
    collect_actor = actor.Actor(
        collect_env,
        collect_policy,
        train_step,
        steps_per_run=1,
        metrics=actor.collect_metrics(10),
        summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
        observers=[rb_observer, env_step_metric]
    )
    # evaluation driver
    eval_actor = actor.Actor(
        eval_env,
        eval_policy,
        train_step,
        episodes_per_run=num_eval_episodes,
        metrics=actor.eval_metrics(num_eval_episodes),
        summary_dir=os.path.join(tempdir, 'eval'),
    )


    # Leaner

    saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)
    # Triggers to save the agent's policy checkpoints.
    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir,
            tf_agent,
            train_step,
            interval=policy_save_interval),
        triggers.StepPerSecondLogTrigger(train_step, interval=1000),
    ]

    agent_learner = learner.Learner(
        tempdir,
        train_step,
        tf_agent,
        experience_dataset_fn,
        triggers=learning_triggers
    )


    # Evaluation

    def get_eval_metrics():
        eval_actor.run()
        results = {}
        for metric in eval_actor.metrics:
            results[metric.name] = metric.result()
        return results
    metrics = get_eval_metrics()


    def log_eval_metrics(step, metrics):
        eval_results = (', ').join('{} = {:.6f}'.format(name, result) for name, result in metrics.items())
        print('step = {0}: {1}'.format(step, eval_results))
    log_eval_metrics(0, metrics)


    # Train

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = get_eval_metrics()["AverageReturn"]
    returns = [avg_return]

    for _ in range(num_iterations):
        # Training.
        collect_actor.run()
        loss_info = agent_learner.run(iterations=1)

        # Evaluating.
        step = agent_learner.train_step_numpy

        if eval_interval and step % eval_interval == 0:
            metrics = get_eval_metrics()
            log_eval_metrics(step, metrics)
            returns.append(metrics["AverageReturn"])

        if log_interval and step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

    rb_observer.close()
    reverb_server.stop()





    # Initialize avg_return
    # avg_return = compute_avg_return(evaluate_env, evaluate_policy, 1)
    # returns = [avg_return]

    # Training
    dataset = replay_buffer.as_dataset(num_parallel_calls=4, sample_batch_size=env.batch_size, num_steps=2)
    iterator = iter(dataset)
    returns = nu.array([])
    steps = nu.array([])
    losses = nu.array([])
    print('All preparation is done. Start training...')
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
