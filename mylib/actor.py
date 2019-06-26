from collections import deque
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random

from mylib.brain import Brain


# Input: Environment State
# Output: Actions' Probability
class Actor:
    def __init__(self, env, action_sz=3, default_cash=2000, seed=10, enable_cuda=True, buy_tax=.1425e-2, sell_tax=.4425e-2, model=None):
        super().__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.env = env
        self.enable_cuda = enable_cuda
        self.action_log_prob = 0
        self.default_cash = default_cash
        self.cash = self.default_cash
        self.hold_stock_count = 0
        self.action = 0
        self.buy_tax = buy_tax
        self.sell_tax = sell_tax
        self.loss = self.to_cuda(torch.tensor(0))
        self.history = { "DATE": [], "ACTION": [], "LOSS": [], "CASH": [], "PORTFOLIO_VALUE": [], "STOCK_HOLD": [], "PENALTY": [] }
        self.more_features_idx = ["CASH", "HOLD_STOCK"]
        self.input_sz = self.env.features_sz + len(self.more_features_idx)
        self.penalty = 1
        self.action_sz = action_sz
        self.action_median = int(self.action_sz/2)

        if(model != None):
            self.brain = self.load_brain(model)
        else:
            self.brain = Brain(self.input_sz, self.action_sz, enable_cuda=enable_cuda)
            self.brain = self.to_cuda(self.brain)

        self.optimizer = optim.Adam(self.brain.parameters(), lr=1e-2)


    def load_brain(self, name):
        path = "models/" + name
        model = torch.load(path)
        self.to_cuda(model)

        return model


    def update_env(self, new_env):
        self.env = new_env


    # 0: Buy, 1: Hold, 2: Sell
    def choose_action(self, state):
        actions_prob = self.brain.forward(state)
        m = Categorical(actions_prob)
        action = m.sample()
        self.action_log_prob = m.log_prob(action)

        return action.item()


    def buy(self, count=1):
        curr_close_price = self.env.get_close_price()
        total_price = float(curr_close_price) * count * (1 + self.buy_tax)

        if(self.cash >= total_price):
            self.cash -= total_price
            self.hold_stock_count += count
        else:
            count = 0

        return count


    def sell(self, count=1):
        if(self.hold_stock_count >= count):
            self.hold_stock_count -= count
            curr_close_price = self.env.get_close_price()
            self.cash += float(curr_close_price) * count * (1 - self.sell_tax)
        else:
            count = 0

        return count


    def step(self, action):
        self.action = action
        curr_close_price =  self.env.get_close_price()

        if(action == self.action_median): # Hold
            self.penalty *= 1.01
        elif(action < self.action_median): # Buy
            buy_count = self.action_median - action
            if(self.cash >= (curr_close_price*buy_count)):
                self.buy(buy_count)
                self.penalty /= 1.01
            else:
                self.penalty *= 1.01
        elif(action > self.action_median): # Sell
            sell_count = action - self.action_median
            if(self.hold_stock_count >= sell_count):
                self.sell(sell_count)
                self.penalty /= 1.01
            else:
                self.penalty *= 1.01
        else:
            print("*Error: Action out of range.")

        self.env.step()
        next_state = self.get_state()
        reward = self.portfolio_value() - self.default_cash - self.penalty

        return next_state, reward


    def learn(self, td_error, drop_rate=0):
        if(random.random() >= drop_rate):
            self.optimizer.zero_grad()
            self.loss = -self.action_log_prob * self.to_cuda(torch.FloatTensor(td_error))
            self.loss.backward()
            self.optimizer.step()


    def get_state(self):
        state = self.env.get_state().values
        state = state.squeeze()
        state = np.append(state, self.get_more_features())

        return state


    def to_cuda(self, tensor):
        if(self.enable_cuda == True):
            tensor = tensor.cuda()

        return tensor


    def get_more_features(self):
        features = []
        for idx in self.more_features_idx:
            if(idx == "CASH"):
                features.append(self.cash)
            elif(idx == "HOLD_STOCK"):
                features.append(self.hold_stock_count)

        return features


    def record(self):
        self.history["DATE"].append(self.env.get_date(-1))
        self.history["ACTION"].append(self.action)
        self.history["LOSS"].append(self.loss.item())
        self.history["CASH"].append(self.cash)
        self.history["PORTFOLIO_VALUE"].append(self.portfolio_value())
        self.history["STOCK_HOLD"].append(self.hold_stock_count)
        self.history["PENALTY"].append(self.penalty)


    def save_model(self, name="actor.pkl"):
        try:
            torch.save(self.brain.cpu(), "models/"+name)
            self.to_cuda(self.brain)
            print("*Successfully Saved Model: {}".format(name))
        except:
            print("*Failed to Saved Model: {}".format(name))


    def portfolio_value(self):
        cash = self.hold_stock_count * self.env.get_close_price() * (1 - self.sell_tax)
        cash += self.cash

        return cash


    def get_reward(self):
        return self.portfolio_value() - self.default_cash


    # Reset every parameter to default value except {brain}.
    def reset(self):
        self.env.reset()
        self.action_log_prob = 0
        self.hold_stock_count = 0
        self.cash = self.default_cash
        self.action = 0
        self.loss = self.to_cuda(torch.tensor(0))
        self.penalty = 1
        self.history = { "DATE": [], "ACTION": [], "LOSS": [], "CASH": [], "PORTFOLIO_VALUE": [], "STOCK_HOLD": [], "PENALTY": [] }