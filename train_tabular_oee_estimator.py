from deep_rl import *
import argparse
from models.BetaNet import BetaNetwork
import sys
import importlib
import sys
import os
import pickle 
print (os.getcwd())
import numpy as np
import torch as th
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.utils import set_random_seed
import roboschool 
import sunblaze_envs
from train_beta import main_beta

import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.load_from_hub import download_from_hub
from utils.utils import StoreDict, get_model_path

from utils.environments.cartpole import CartPoleEnv
from utils.taxi_environment import taxi_with_transitions


parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", default=3, type=int,
                    help="Number of Epochs required for training the model")
parser.add_argument("--batch_size", default=16, type=str,
                    help="Batch Size per Epoch")
parser.add_argument("--learning_rate", default=1e-4, type=float,
                    help="Learning Rate of the model")
parser.add_argument("--l2_regularization", default=0.01, type=float,
                    help="L2 regularization in the model")
parser.add_argument("--file_p", default="./offline_data/real_world_data.pkl", type=str, help="file location for transitions stored in p")
parser.add_argument("--file_q", default="./offline_data/sim_world_data.pkl", type=str, help="file location for transitions stored in q")
parser.add_argument("--params_p", default=0.1, type=float, help="environment parameters for p environment")
parser.add_argument("--params_q", default=0.0, type=float, help="environment parameters for q-environment")

parser.add_argument("--env", default="Taxi-v3", type=EnvironmentName, help="RL Environment over which the experiment is being run")
parser.add_argument("--log", default='./dice_log', type=str, help="log directory where the experiment details plus the model will be stored")
parser.add_argument("--use_cuda", type=bool, default=True)
parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=3, type=int)
parser.add_argument("--folder", help="Log folder", type=str, default="./logs")
parser.add_argument("--trained_agent_algo", help="Trained Agent Algo", type=str, default="ppo")
parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="cuda", type=str)
parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
parser.add_argument(
    "--gym-packages",
    type=str,
    nargs="+",
    default=[],
    help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
)
parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument(
    "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
)
parser.add_argument(
    "--load-checkpoint",
    type=int,
    help="Load checkpoint instead of last model if available, "
    "you must pass the number of timesteps corresponding to it",
)
parser.add_argument(
    "--load-last-checkpoint",
    action="store_true",
    default=False,
    help="Load last checkpoint instead of last model if available",
)
parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
parser.add_argument(
    "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
)
parser.add_argument(
    "--env-kwargs", default= {'gravity': 10.0}, type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
)
parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
parser.add_argument(
    "--no-render", action="store_true", default=True, help="Do not render the environment (useful for tests)"
)
parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
parser.add_argument("--sim_policy", type=float, default=0.1)
parser.add_argument("--real_policy", type=float, default=0.4)
parser.add_argument("--index", type=int, default=0)
parser.add_argument("--algo_type", type=str, default='Beta-DICE')
parser.add_argument("--timesteps", type=int, default=150)
args = parser.parse_args()


def off_policy_evaluation(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('correction', 'no')
    kwargs.setdefault('lam', 1)
    kwargs.setdefault('debug', False)
    #kwargs.setdefault('noise_std', 0.05)
    kwargs.setdefault('dataset', 1)
    kwargs.setdefault('discount', None)
    kwargs.setdefault('lr', 0)
    kwargs.setdefault('collect_data', False)
    kwargs.setdefault('target_network_update_freq', 1)
    config = Config()
    config.merge(kwargs)
    config.use_cuda = False
    if args.use_cuda:
        if torch.cuda.is_available():
            config.use_cuda = True 
    config.file_p = './offline_data/real_world_data.pkl'
    config.file_q = './offline_data/sim_world_data.pkl'
    

    if config.correction in ['GradientDICE', 'DualDICE']:
        config.activation = 'linear'
        if config.with_beta:
            config.lam = 1.0
        else:
            config.lam = 0.1
    elif config.correction in ['GenDICE']:
        config.activation = 'squared'
        if config.with_beta:
            config.lam = 1.0
        else:
            config.lam = 0.1
    else:
        raise NotImplementedError

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(1000)
    config.eval_interval = 5#config.max_steps // 100
    config.std = args.sim_policy
    print ("state and action dim:", config.state_dim, config.action_dim)

    config.network_fn = lambda: TD3Net(
        config.action_dim,
        actor_body_fn=lambda: FCBody(config.state_dim, (32, 32), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            config.state_dim + config.action_dim, (32, 32), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    batch_size = 32
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=batch_size)

    config.dice_net_fn = lambda: GradientDICEContinuousNet(
        body_tau_fn=lambda: FCBody(5, gate=F.relu),
        body_f_fn=lambda: FCBody(5, gate=F.relu),
        opt_fn=lambda params: torch.optim.SGD(params, lr=config.lr),
        activation=config.activation
    )

    sample_init_env = Task(config.game, num_envs=batch_size)
    config.sample_init_states = lambda: sample_init_env.reset()

    if config.collect_data:
        agent = OffPolicyEvaluationDiscrete(config)
        run_steps(agent)
        filename = './dice_log' +'/' + args.algo_type + '_' + file_appender + '_' + str(args.index) + '.ptr'
        print (filename)
        torch.save(agent.DICENet.state_dict(), filename)
        with open('./dice_log' + '/' + args.algo_type + '_' + file_appender + '_' + str(args.index) + '_' + config.correction + '.pkl', 'wb') as f:
            pickle.dump(agent.loss_history, f)
    else:
        run_steps(OffPolicyEvaluationDiscrete(config))


if __name__ == '__main__':
    
    if args.algo_type == 'Beta-DICE':
        file_appender = str(int(10*args.params_p)) + '_' + str(int(10*args.params_q)) + str(int(10*args.real_policy)) + '_' + str(int(10*args.sim_policy)) + '_' + str(args.timesteps)
        beta_network = [main_beta(args=args) for _ in range(1)]
        for i in range(len(beta_network)):
            for params in beta_network[i].parameters():
                params.requires_grad = False

        # Going through custom gym packages to let them register in the global registory
        q_table = pickle.load(open('./logs/dqn/taxi_v3_1.pkl', 'rb'))
                                    
        game = args.env
        off_policy_evaluation(
            tag = 'cartpole_dice_integration_with_oee',
            collect_data=True,
            game=game,
            correction='GradientDICE',
            algorithm=args.algo_type,
            discount=0.9,
            lr=3e-3,
            lam=1,
            sim_policy=args.sim_policy,
            target_network_update_freq=1,
            expert_policy=q_table,
            beta_factor=beta_network, 
            environment_p= taxi_with_transitions(transition_probability=args.params_p), 
            environment_q=taxi_with_transitions(transition_probability=args.params_q),
            noise_std=args.sim_policy,
            data_collection_noise=args.real_policy, 
            deterministic=False,
            file_appender = str(args.params_p) + '_' + str(args.params_q) + str(int(10*args.real_policy)) + '_' + str(int(10*args.sim_policy)) + '_' + str(args.timesteps), 
            index=args.index, 
            with_beta=True)
    else:
        file_appender = str(int(10*args.params_p)) + '_' + str(int(10*args.params_q)) + str(int(10*args.real_policy)) + '_' + str(int(10*args.sim_policy)) + '_' + str(args.timesteps)
        q_table = pickle.load(open('./logs/dqn/taxi_v3_1.pkl', 'rb'))
        game = args.env
        off_policy_evaluation(
            tag = 'cartpole_dice_integration_with_oee',
            collect_data=True,
            game=game,
            correction=args.algo_type,
            algorithm=args.algo_type,
            discount=0.9,
            lr=1e-2,
            lam=1,
            sim_policy=args.sim_policy,
            target_network_update_freq=1,
            expert_policy=q_table,
            beta_factor=None, 
            environment_p=taxi_with_transitions(transition_probability=args.params_p), 
            environment_q=taxi_with_transitions(transition_probability=args.params_p),
            noise_std=args.sim_policy,
            data_collection_noise=args.real_policy, 
            deterministic=False,
            file_appender = str(args.params_p) + '_' + str(args.params_q) + str(int(10*args.real_policy)) + '_' + str(int(10*args.sim_policy)) + '_' + str(args.timesteps), 
            index=args.index, 
            with_beta=False)
        
