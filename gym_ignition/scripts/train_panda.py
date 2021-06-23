import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
import math
import torch
import torch.nn as nn

from ruamel.yaml import YAML, dump, RoundTripDumper
from gym_ignition.env.GymIgnitionVecEnv import GymIgnitionVecEnv as Environment
from gym_ignition.env.env.panda import __PANDA_RESOURCE_DIRECTORY__ as __RSCDIR__
from _gym_ignition import GymIgnitionEnv
from gym_ignition.helper.helpers import ConfigurationSaver

import gym_ignition.algo.ppo.module as ppo_module
import gym_ignition.algo.ppo.ppo as PPO

def main(args):

    # directories
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    __RSCDIR__ =  os.path.join(root, 'gym_ignition', 'env', 'env', 'panda', 'rsc') # TODO: fix the import statement above so that the __RSCDIR__ is not on the install path
    log_path = os.path.join(root, 'data', 'Panda')

    cfg = YAML().load(open(os.path.join(__RSCDIR__, 'cfg.yaml'), 'r'))

    # create an environment
    env = Environment(GymIgnitionEnv(
        __RSCDIR__, 
        dump(cfg['environment'], Dumper=RoundTripDumper)))

    # save the current training configuration
    saver = ConfigurationSaver(
        log_dir=log_path,
        save_items=[os.path.join(__RSCDIR__, 'cfg.yaml'), os.path.join(__RSCDIR__, '..', 'Environment.hpp')]
    )

    print('[train_panda.py] Saving log files for current training run in: {}'.format(saver.data_dir))

    device = cfg['device'] # 'cuda' or 'cpu'
    env_node = cfg['environment']
    algo_node = env_node['algorithm']

    # training
    n_steps = math.floor(env_node['max_time'] / env_node['control_dt'])

    actor = ppo_module.Actor(
        ppo_module.MLP(env_node['architecture']['policy'],  # number of layers and neurons in each layer
        getattr(nn, env_node['architecture']['activation']),  # activation function at each layer
        env.num_obs,  # number of states (input dimension)
        env.num_acts,  # number of actions (output)
        env_node['architecture']['init_scale']),
        ppo_module.MultivariateGaussianDiagonalCovariance(env.num_acts, 1.0),
        device)

    critic = ppo_module.Critic(
        ppo_module.MLP(env_node['architecture']['value_net'],
        getattr(nn, env_node['architecture']['activation']),
        env.num_obs,
        1,
        env_node['architecture']['init_scale']), device)

    ppo = PPO.PPO(actor=actor,
                critic=critic,
                num_envs=env_node['num_envs'],
                num_transitions_per_env=n_steps,
                num_learning_epochs=algo_node['epoch'],
                clip_param=algo_node['clip_param'],
                gamma=algo_node['gamma'],
                lam=algo_node['lambda'],
                entropy_coef=algo_node['entropy_coeff'],
                learning_rate=algo_node['learning_rate'],
                num_mini_batches=algo_node['minibatch'],
                device=device
                )

    ppo.learn(env, saver, env_node)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: add arguments

    args = parser.parse_args()

    main(args)