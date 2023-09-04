
import torch 
from torch.utils.data import Dataset
import pickle 
import numpy as np 

class BetaEstimationDataset(Dataset):
    
    def __init__(self, filename_p, filename_q, action_space=1):
        self.filename_p = filename_p
        self.filename_q = filename_q
        self.action_space = action_space
        
        self.data_p = pickle.load(open(self.filename_p, "rb"))
        self.data_q = pickle.load(open(self.filename_q, "rb"))
    
    def __len__(self):
        return min(len(self.data_p), len(self.data_q))
 
    def __getitem__(self, index):
        
        '''
        every index has the following information 
        old_obs_p, action_p, obs_p, reward_p, done_p = self.data_p[index]
        old_obs_q, action_q, obs_q, reward_q, done_q = self.data_q[index]
        '''
        old_obs_p, action_p, obs_p, reward_p, done_p = self.data_p[index]
        old_obs_q, action_q, obs_q, reward_q, done_q = self.data_q[index]
        
        action_p = np.asarray(action_p)
        action_q = np.asarray(action_q)
        
        
        action_p = np.reshape(action_p.astype(float), (1, self.action_space))
        action_q = np.reshape(action_q.astype(float), (1, self.action_space))
        
        data_p = np.concatenate((old_obs_p, action_p), axis=1)
        data_q = np.concatenate((old_obs_q, action_q), axis=1)
        
        data_p = torch.from_numpy(data_p)
        data_q = torch.from_numpy(data_q)
        
        return data_p.type(torch.FloatTensor), data_q.type(torch.FloatTensor)
        
        
        

        