from collections import deque
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical
from mylib.brain import Brain


# Input: Environment State
# Output: Actions' Probability
class Actor:
    def __init__(self, env, action_sz=3, default_cash=2000, seed=10, enable_cuda=True, trans_tax=3e-3, model=None):
        super().__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.env = env
        self.enable_cuda = enable_cuda
        self.actions_log_prob_his = deque()
        self.default_cash = default_cash
        self.cash = self.default_cash
        self.hold_stock_count = 0
        self.action = 0
        self.transaction_tax = trans_tax
        self.loss = torch.tensor(0)
        self.history = {"DATE": [], "ACTION": [], "LOSS": [], "CASH": [], "PORTFOLIO_VALUE": []}
        self.more_features_idx = ["CASH", "HOLD_STOCK"]
        self.input_sz = self.env.features_sz + len(self.more_features_idx)
        self.penalty = 1
        self.action_sz = action_sz

        if(model != None):
            self.brain = self.load_brain(model)
        else:
            self.brain = Brain(self.input_sz, self.action_sz).cuda()


    def load_brain(self, name):
        path = "models/" + name
        model = torch.load(path)

        return model


    # 0: Hold, 1: Buy, 2: Sell
    def choose_action(self, state):
        actions_prob = self.brain.forward(state)
        m = Categorical(actions_prob)
        action = m.sample()
        self.actions_log_prob_his.append(m.log_prob(action))

        return action.item()


    def buy(self, count=1):
        curr_close_price = self.env.get_close_price()
        total_price = float(curr_close_price) * count * (1 + self.transaction_tax)

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
            self.cash += float(curr_close_price) * count
        else:
            count = 0

        return count


    def step(self, action):
        self.action = action
        curr_close_price =  self.env.get_close_price()

        if(action == 0): # Hold
            self.penalty *= 1.005
        elif(action == 1): # Buy
            if(self.cash >= curr_close_price):
                self.buy()
            else:
                self.penalty *= 1.001
        elif(action == 2): # Sell
            if(self.hold_stock_count >= 0):
                self.sell()
            else:
                self.penalty *= 1.01
        else:
            print("*Error: Action out of range.")

        self.env.step()
        next_state = self.get_state()
        reward = self.portfolio_value() - self.default_cash - self.penalty

        return next_state, reward


    def learn(self, td_error):
        self.brain.optimizer.zero_grad()
        log_prob = self.actions_log_prob_his.popleft()
        self.loss = -log_prob * torch.FloatTensor(td_error).cuda()
        self.loss.backward()
        self.brain.optimizer.step()

        return self.loss.item()


    def get_state(self):
        state = self.env.get_state().values
        state_tensor = torch.FloatTensor(state).cuda().squeeze()
        more_features = self.get_more_features()
        more_features = torch.FloatTensor(more_features).cuda().squeeze()
        state_tensor = torch.cat([state_tensor, more_features], dim=0)

        return state_tensor


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


    def save_model(self, name="actor.pkl"):
        try:
            torch.save(self.brain, "models/"+name)
            print("*Successfully Saved Model: {}".format(name))
        except:
            print("*Failed to Saved Model: {}".format(name))


    def portfolio_value(self):
        cash = self.hold_stock_count * self.env.get_close_price()
        cash += self.cash

        return cash


    def reset(self):
        self.env.reset()
        self.actions_log_prob_his.clear()
        self.hold_stock_count = 0
        self.cash = self.default_cash
        self.action = 0
        self.loss = torch.tensor(0)
        self.history = { "DATE": [], "ACTION": [], "LOSS": [], "CASH": [], "PORTFOLIO_VALUE": [] }