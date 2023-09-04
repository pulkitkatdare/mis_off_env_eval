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
parser.add_argument("--params_p", default=15.0, type=float, help="environment parameters for p environment")
parser.add_argument("--params_q", default=10.0, type=float, help="environment parameters for q-environment")

parser.add_argument("--env", default="RoboschoolReacher-v1", type=EnvironmentName, help="RL Environment over which the experiment is being run")
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
    config.max_steps = int(2000)
    config.eval_interval = 5#config.max_steps // 100
    config.std = args.sim_policy
    print ("state and action dim:", config.state_dim, config.action_dim)

    config.network_fn = lambda: TD3Net(
        config.action_dim,
        actor_body_fn=lambda: FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            config.state_dim + config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    batch_size = 32
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=batch_size)

    config.dice_net_fn = lambda: GradientDICEContinuousNet(
        body_tau_fn=lambda: FCBody(config.state_dim + config.action_dim, gate=F.relu),
        body_f_fn=lambda: FCBody(config.state_dim + config.action_dim, gate=F.relu),
        opt_fn=lambda params: torch.optim.SGD(params, lr=config.lr),
        activation=config.activation
    )

    sample_init_env = Task(config.game, num_envs=batch_size)
    config.sample_init_states = lambda: sample_init_env.reset()

    if config.collect_data:
        agent = OffPolicyEvaluationContinuous(config)
        run_steps(agent)
        filename = './dice_log' +'/' + args.algo_type + '_' + file_appender + '_' + str(args.index) + '.ptr'
        print (filename)
        torch.save(agent.DICENet.state_dict(), filename)
        with open('./dice_log' + '/' + args.algo_type + '_' + file_appender + '_' + str(args.index) + '_' + config.correction + '.pkl', 'wb') as f:
            pickle.dump(agent.loss_history, f)
    else:
        run_steps(OffPolicyEvaluationContinuous(config))


if __name__ == '__main__':
    file_appender = str(int(10*args.params_p)) + '_' + str(int(10*args.params_q)) + str(int(10*args.real_policy)) + '_' + str(int(10*args.sim_policy)) + '_' + str(args.timesteps)
    #beta_network = BetaNetwork(state_dim=11, learning_rate=args.learning_rate, tau=args.l2_regularization, seed=1234, action_dim = 1)
    beta_network = [main_beta(args=args) for _ in range(5)]
    #beta_network.load_state_dict(th.load(args.log + '/beta_model_' + file_appender + '_4' + '.ptr')) # example code
    for i in range(5):
        for params in beta_network[i].parameters():
            params.requires_grad = False

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_name: EnvironmentName = args.env
    algo = args.algo
    folder = args.folder

    try:
        _, model_path, log_path = get_model_path(
            args.exp_id,
            folder,
            algo,
            env_name,
            args.load_best,
            args.load_checkpoint,
            args.load_last_checkpoint,
        )
    except (AssertionError, ValueError) as e:
        # Special case for rl-trained agents
        # auto-download from the hub
        if "rl-trained-agents" not in folder:
            raise e
        else:
            print("Pretrained model not found, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3")
            # Auto-download
            #download_from_hub(
            #    algo=algo,
            #    env_name=env_name,
            #    exp_id=args.exp_id,
            #    folder=folder,
            #    organization="sb3",
            #    repo_name=None,
            #    force=False,
            #)
            # Try again
            _, model_path, log_path = get_model_path(
                args.exp_id,
                folder,
                algo,
                env_name,
                args.load_best,
                args.load_checkpoint,
                args.load_last_checkpoint,
            )

    print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_name.gym_id)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        print (args.env_kwargs)
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None
    env_p = create_test_env(
        env_name.gym_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs={'env_id': args.params_p},
    )
    env_q = create_test_env(
        env_name.gym_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs={'env_id': 1},
    ) #simulation


    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))
        # Hack due to breaking change in v1.6
        # handle_timeout_termination cannot be at the same time
        # with optimize_memory_usage
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model = ALGOS[algo].load(model_path, env=env_q, custom_objects=custom_objects, device=args.device, **kwargs)
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic
    print ("done")
    
    mkdir('data')
    mkdir ('log')
                                 
    game = args.env
    if args.algo_type == 'Beta-DICE':
        '''
        off_policy_evaluation(
            tag = 'cartpole_dice_integration_with_oee',
            collect_data=True,
            game=game,
            correction='GenDICE',
            algorithm=args.algo_type,
            discount=0.99,
            lr=1e-3,
            lam=1,
            sim_policy=args.sim_policy,
            target_network_update_freq=1,
            expert_policy=model,
            beta_factor=beta_network, 
            environment_p=env_p, 
            environment_q=env_q,
            noise_std=args.sim_policy,
            data_collection_noise=args.real_policy, 
            deterministic=deterministic,
            file_appender = str(args.params_p) + '_' + str(args.params_q) + str(int(10*args.real_policy)) + '_' + str(int(10*args.sim_policy)) + '_' + str(args.timesteps), 
            index=args.index, 
            with_beta=True)
        '''
            
        off_policy_evaluation(
            tag = 'cartpole_dice_integration_with_oee',
            collect_data=True,
            game=game,
            correction='GradientDICE',
            algorithm=args.algo_type,
            discount=0.99,
            lr=3e-3,
            lam=1,
            sim_policy=args.sim_policy,
            target_network_update_freq=1,
            expert_policy=model,
            beta_factor=beta_network, 
            environment_p=env_p, 
            environment_q=env_q,
            noise_std=args.sim_policy,
            data_collection_noise=args.real_policy, 
            deterministic=deterministic,
            file_appender = str(args.params_p) + '_' + str(args.params_q) + str(int(10*args.real_policy)) + '_' + str(int(10*args.sim_policy)) + '_' + str(args.timesteps), 
            index=args.index, 
            with_beta=True)
    else:
        off_policy_evaluation(
            tag = 'cartpole_dice_integration_with_oee',
            collect_data=True,
            game=game,
            correction=args.algo_type,
            algorithm=args.algo_type,
            discount=0.99,
            lr=5e-3,
            lam=1,
            sim_policy=args.sim_policy,
            target_network_update_freq=1,
            expert_policy=model,
            beta_factor=beta_network, 
            environment_p=env_p, 
            environment_q=env_p,
            noise_std=args.sim_policy,
            data_collection_noise=args.real_policy, 
            deterministic=deterministic,
            file_appender = str(args.params_p) + '_' + str(args.params_q) + str(int(10*args.real_policy)) + '_' + str(int(10*args.sim_policy)) + '_' + str(args.timesteps), 
            index=args.index, 
            with_beta=False)
        
