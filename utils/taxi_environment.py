import gym
import numpy as np 
import random
from os import system, name
from time import sleep
import pickle as pkl 


class taxi_with_transitions(object):
    def __init__(self, taxi_env=gym.make('Taxi-v3').env, transition_probability=0.0):
        self.env = taxi_env
        self.transition_probability = transition_probability
        
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        action_probability  = np.zeros(6)
        action_probability[action] = 1.0
        for i in range(6):
            if i == action:
                action_probability[i] -= (5/6)*self.transition_probability
            else:
                action_probability[i] += (1/6)*self.transition_probability
        final_action = np.random.choice(np.arange(0, 6), p=action_probability)
        return self.env.step(final_action)
            