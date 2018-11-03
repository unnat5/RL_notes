

import pickle
import gym
import numpy as np


import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super().__init__()
        self.fc1 = nn.Linear(s_size,h_size)
        self.fc2 = nn.Linear(h_size,a_size)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x,dim=1)
    
    def act(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs =self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(),m.log_prob(action)
    
    


































policy = Policy()
policy.load_state_dict(torch.load('policy_G_cartpole.pth'))
env = gym.make('CartPole-v0')

for i in range(3):
    state = env.reset()
    while True:
        env.render()
        action,_ = policy.act(state)
        state,reward,done,_=env.step(action)
        if done:
            break