import gym 
import numpy as np 
import torch 



class Tau_Wrapper(gym.Wrapper):
    def __init__(self, env, tau):
        self.env = env
        self.state = env.reset()
        self.dicenet = tau
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        obs_tensor = torch.tensor(self.state).type(torch.FloatTensor)
        action_tensor = torch.tensor(action).type(torch.FloatTensor)
        tau = self.dicenet.tau(obs_tensor.cuda(), action_tensor.cuda()).cpu().detach().numpy().reshape(-1)
        # can add clipping here...
        reward = np.multiply(reward, np.clip(tau, 1-0.4, 1+0.4))
        self.state = next_state
        return next_state, reward, done, info
    
    def reset(self):
        self.state = self.env.reset()
        return self.state