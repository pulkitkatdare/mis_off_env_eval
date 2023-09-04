from asyncio.unix_events import BaseChildWatcher
from dataclasses import dataclass
from models.BetaNet import BetaNetwork
import argparse
import numpy as np 
import pickle 
from torch.utils.data import DataLoader
from tqdm import tqdm
from oee_dataset import BetaEstimationDataset
import torch


def main_beta(args):
    if args.env == 'RoboschoolHalfCheetah-v1':
        dataset = BetaEstimationDataset(filename_p=args.file_p, filename_q=args.file_q, action_space=6)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        file_appender = str(args.params_p) + '_' + str(args.params_q) + str(int(10*args.real_policy)) + '_' + str(int(10*args.sim_policy)) + '_' + str(args.timesteps)
        beta_network = BetaNetwork(state_dim=32, learning_rate=args.learning_rate, tau=args.l2_regularization, seed=1234, action_dim = 6)
        if args.use_cuda:
            beta_network = beta_network.to('cuda:0')

    if args.env == 'CartPole-v1':
        dataset = BetaEstimationDataset(filename_p=args.file_p, filename_q=args.file_q, action_space=1)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        file_appender = str(args.params_p) + '_' + str(args.params_q) + str(int(10*args.real_policy)) + '_' + str(int(10*args.sim_policy)) + '_' + str(args.timesteps)
        beta_network = BetaNetwork(state_dim=5, learning_rate=args.learning_rate, tau=args.l2_regularization, seed=1234, action_dim = 1)
        if args.use_cuda:
            beta_network = beta_network.to('cuda:0')

    if args.env == 'RoboschoolReacher-v1':
        dataset = BetaEstimationDataset(filename_p=args.file_p, filename_q=args.file_q, action_space=2)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        file_appender = str(args.params_p) + '_' + str(args.params_q) + str(int(10*args.real_policy)) + '_' + str(int(10*args.sim_policy)) + '_' + str(args.timesteps)
        beta_network = BetaNetwork(state_dim=11, learning_rate=args.learning_rate, tau=args.l2_regularization, seed=1234, action_dim = 2)
        if args.use_cuda:
            beta_network = beta_network.to('cuda:0')

    if args.env == 'Taxi-v3':
        dataset = BetaEstimationDataset(filename_p=args.file_p, filename_q=args.file_q, action_space=1)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        file_appender = str(args.params_p) + '_' + str(args.params_q) + str(int(10*args.real_policy)) + '_' + str(int(10*args.sim_policy)) + '_' + str(args.timesteps)
        beta_network = BetaNetwork(state_dim=5, learning_rate=args.learning_rate, tau=args.l2_regularization, seed=1234, action_dim = 1)
        if args.use_cuda:
            beta_network = beta_network.to('cuda:0')
    
    

    
    for epoch in tqdm(range(args.num_epochs)):
        epoch_losses = []
        for iteration, data in enumerate(tqdm(dataloader)):
            data_p = data[0]
            data_q = data[1]
            if args.use_cuda:
                data_p = data_p.to('cuda:0')
                data_q = data_q.to('cuda:0')
            
            loss = beta_network.train_step(states_p=data_q, states_q=data_p)
            epoch_losses.append(loss.item())
            
            if (iteration%100 == 0):
                print ("loss:", loss.item())
                with open(args.log + '/epoch_loss_' + file_appender + '_' + str(epoch) + '.pkl', 'wb') as fp:
                    pickle.dump(epoch_losses, fp,  protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(beta_network.state_dict(), args.log + '/beta_model_' + file_appender + '_' + str(epoch) + '.ptr')
    
    return beta_network
        
                





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", default=5, type=int,
                        help="Number of Epochs required for training the model")
    parser.add_argument("--batch_size", default=64, type=str,
                        help="Batch Size per Epoch")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="Learning Rate of the model")
    parser.add_argument("--l2_regularization", default=0.01, type=float,
                        help="L2 regularization in the model")
    parser.add_argument("--file_p", default="./offline_data/real_world_data.pkl", type=str, help="file location for transitions stored in p")
    parser.add_argument("--file_q", default="./offline_data/sim_world_data.pkl", type=str, help="file location for transitions stored in q")
    parser.add_argument("--params_p", default=150, type=int, help="environment parameters for p environment")
    parser.add_argument("--params_q", default=100, type=int, help="environment parameters for q-environment")
    
    parser.add_argument("--env", default="RoboschoolReacher-v1", type=str, help="RL Environment over which the experiment is being run")
    parser.add_argument("--log", default='./dice_log/', type=str, help="log directory where the experiment details plus the model will be stored")
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--timesteps", type=int, default=150)
    parser.add_argument("--real_policy", type=float, default=0.4)
    parser.add_argument("--sim_policy", type=float, default=0.1)
    args = parser.parse_args()
    #main_beta(args=args)
