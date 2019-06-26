from collections import deque
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import random

from mylib.brain import Brain


# Input: Environment State
# Output: State Value
class Critic(nn.Module):
    def __init__(self, input_sz, seed=10, LR=5e-4, gamma=0.99, batch_sz=64, drop_rate=0.7, enable_cuda=True):
        super().__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.gamma = gamma
        self.enable_cuda = enable_cuda
        self.brain = Brain(input_sz, 1, softmax=False, enable_cuda=enable_cuda)
        self.brain = self.to_cuda(self.brain)
        self.optimizer = optim.Adam(self.brain.parameters(), lr=LR)
        self.memory = deque()
        self.drop_rate = drop_rate
        self.batch_sz = batch_sz


    def exp_replay(self, state, action, reward):
        self.memory.append((state, action, reward))

        if(len(self.memory) >= self.batch_sz):
            while(self.memory):
                (state, action, reward) = self.memory.popleft()

                if(random.random() >= self.drop_rate):
                    self.learn(state, action, reward)


    def learn(self, state, action, reward):
        self.optimizer.zero_grad()
        value = self.brain.forward(np.append(state, action))
        reward_tensor = self.to_cuda(torch.FloatTensor(reward))
        loss = self.to_cuda(F.smooth_l1_loss(value, reward_tensor))
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def td_error(self, state, action, reward, next_state, next_action):
        value = self.brain.forward(np.append(state, action))
        next_value = self.brain.forward(np.append(next_state, next_action))
        td_error = reward + (self.gamma * next_value.item()) - value.item()

        return td_error


    def to_cuda(self, tensor):
        if(self.enable_cuda == True):
            tensor = tensor.cuda()

        return tensor


    def save_model(self, name="critic.pkl"):
        try:
            torch.save(self.brain.cpu(), "models/"+name)
            print("*Successfully Saved Model: {}".format(name))
        except:
            print("*Failed to Saved Model: {}".format(name))