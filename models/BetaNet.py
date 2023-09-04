import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class BetaNetwork(nn.Module):
 
    def __init__(self, state_dim, learning_rate, tau, seed, action_dim = 1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.seed = seed

        super(BetaNetwork, self).__init__()
        torch.manual_seed(self.seed)
        self.layer1 = layer_init(nn.Linear(self.state_dim, 256), 1e-3)
        torch.manual_seed(self.seed)
        self.layer2 = layer_init(nn.Linear(256, 1), 1e-3)
        

        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay=0.00)

    def forward(self, x):

        y = F.relu(self.layer1(x))
        y = ((self.layer2(y)))**2

        scaled_y = y + 1e-10

        return x, y, scaled_y

    def predict(self, x):
        x, y, scaled_y = self.forward(x)
        return scaled_y

    def train_step(self, states_p, states_q):
        self.optimizer.zero_grad()
        output1 = torch.pow(self.predict(states_q), 2)
        output1 = torch.mean((output1))
        output2 = self.predict(states_p)
        output2 = torch.mean(torch.log(output2))
        output = 0.5*output1 - output2
        for param in self.parameters():
            output += 0.5*self.tau * torch.norm(param) ** 2
        output.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
  
        return output
