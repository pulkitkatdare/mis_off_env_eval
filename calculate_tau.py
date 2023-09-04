from deep_rl import *
import argparse
from models.BetaNet import BetaNetwork
import sys
import importlib
import sys
import os
import pickle 
import numpy as np
import torch as th
import yaml
from stable_baselines3.common.utils import set_random_seed
import roboschool 
import gym 
import sunblaze_envs
#import utils.import_envs  # noqa: F401 pylint: disable=unused-import
#from utils.exp_manager import ExperimentManager
#from utils.load_from_hub import download_from_hub
#from utils.utils_for_tau import StoreDict, get_model_path

from utils.utils_for_tau import get_saved_hyperparams



from utils.environments.cartpole import CartPoleEnv

from calculate_beta import calculate_beta

def calculate_tau(model, file_p, file_q, log_path,       
                num_epochs = 5,
                batch_size = 64,
                learning_rate = 1e-4,
                l2_regularization = 0.01,
                seed = 1,
                env_id = "RoboschoolHalfCheetah-v1", 
                log = "./debug/beta_log/", 
                use_cuda = True,
                num_threads = -1,
                norm_reward = False,
                reward_log = "",
                deterministic = False,
                index = 0,
                verbose = 1,
                log_level = 0,
                correction = 'Beta-GradientDICE',
                lam = 1,
                debug = False,
                discount = 0.99,
                lr = 0,
                collect_data = True,
                target_network_update_freq = 1):


    beta_network = calculate_beta(env_id, log, file_p, file_q, num_epochs, batch_size, learning_rate, l2_regularization, use_cuda)
    if use_cuda:
        beta_network = beta_network.cuda()
    
    

    for params in beta_network.parameters():
        params.requires_grad = False
        
    set_random_seed(seed)

    if num_threads > 0:
        if verbose > 1:
            print(f"Setting torch.num_threads to {num_threads}")
        th.set_num_threads(num_threads)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if env_kwargs is not None:
        print (env_kwargs)
        env_kwargs.update(env_kwargs)

    lr=1e-2
    with_beta=True

    # Note to Pulkit: in off_policy_evaluation function, from train_oee_estimator_continous.py,
    #                 'discount' is set to None, even though it's passed in as '0.99' 

    config_dict = {
            'tag' : 'cartpole_dice_integration_with_oee',
            'collect_data' : collect_data,
            'game' : env_id,
            'correction' : correction,
            'discount' : discount,
            'lr' : lr,
            'lam' : lam,
            'target_network_update_freq' : target_network_update_freq,
            'expert_policy' : model,
            'beta_factor' : beta_network, 
            'deterministic' : deterministic,
            'file_appender' : '', 
            'index' : index, 
            'with_beta' : with_beta,
            'log_level' : log_level,
            'debug' : debug,
            'dataset' : 1, 
            'real_env': sunblaze_envs.make('SunblazeStrongHalfCheetah-v0'), 
            'sim_env': gym.make('RoboschoolHalfCheetah-v1'),
            'file_p': file_p, 
            'file_q': file_q,
            'batch_size': 64, 
            'use_cuda': True
    }

    config = Config()
    config.merge(config_dict)

    if config.correction in ['Beta-GradientDICE', 'GradientDICE', 'DualDICE']:
        config.activation = 'linear'
        config.lam = .1
    elif config.correction in ['GenDICE']:
        config.activation = 'squared'
        config.lam = 1
    else:
        raise NotImplementedError

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(1000)
    config.eval_interval = config.max_steps // 10
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=batch_size)

    config.dice_net_fn = lambda: GradientDICEContinuousNet(
        body_tau_fn=lambda: FCBody(config.state_dim + 6, gate=F.relu),
        body_f_fn=lambda: FCBody(config.state_dim + 6, gate=F.relu),
        opt_fn=lambda params: torch.optim.SGD(params, lr=config.lr),
        activation=config.activation
    )

    sample_init_env = Task(config.game, num_envs=batch_size)
    config.sample_init_states = lambda: sample_init_env.reset()

    '''
    if config.collect_data:
        agent = OffPolicyEvaluationContinuous(config)
        agent.collect_data()
        run_steps(agent)
        filename = './log_dr_halfcheetah' +'/' + algo_type + '_' + file_appender + '_' + str(config.index) + '.ptr'
        print (filename)
        torch.save(agent.DICENet.state_dict(), filename)
        with open('./log_dr_halfcheetah' + '/' + algo_type + '_' + file_appender + '_' + str(config.index) + '.pkl', 'wb') as f:
            pickle.dump(agent.loss_history, f)
    else:
        run_steps(OffPolicyEvaluationContinuous(config))
    '''

    agent = OffPolicyEvaluationContinuous(config)
    run_steps(agent)
    agent.DICENet.load_state_dict(agent.best_model_state_dict)
    
    return agent.DICENet#.state_dict()