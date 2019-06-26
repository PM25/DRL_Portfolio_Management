from collections import deque
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np

from mylib.brain import Brain


# Input: Environment State
# Output: State Value
class Critic(nn.Module):
    def __init__(self, input_sz, seed=10, LR=5e-4, enable_cuda=True):
        super().__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.enable_cuda = enable_cuda
        self.brain = Brain(input_sz, 1, softmax=False, enable_cuda=enable_cuda)
        self.brain = self.to_cuda(self.brain)
        self.optimizer = optim.Adam(self.brain.parameters(), lr=LR)


    def exp_replay(self, state, action, reward):
        pass

    def learn(self, state, action, reward, next_state, next_action):
        value = self.brain.forward(np.append(state, action))
        next_value = self.brain.forward(np.append(next_state, next_action))
        td_error = reward + next_value.item() - value.item()

        self.optimizer.zero_grad()
        reward_tensor = self.to_cuda(torch.FloatTensor(reward))
        loss = self.to_cuda(F.smooth_l1_loss(value, reward_tensor))
        loss.backward()
        self.optimizer.step()

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