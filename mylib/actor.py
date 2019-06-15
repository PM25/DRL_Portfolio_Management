from collections import deque
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical


# Input: Environment State
# Output: Actions' Probability
class Actor(nn.Module):
    def __init__(self, features_sz, output_sz, env, default_cash=2000, seed=10, enable_cuda=True, LR=10e-3):
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
        self.loss = torch.tensor(0)
        self.history = {"DATE": [], "ACTION": [], "LOSS": [], "CASH": [], "PORTFOLIO_VALUE": []}
        self.more_features_idx = ["CASH", "HOLD_STOCK"]
        self.input_sz = features_sz + len(self.more_features_idx)

        self.input_layer = nn.Linear(in_features=self.input_sz, out_features=128)
        self.hidden_1 = nn.Linear(in_features=128, out_features=128)
        self.rnn = nn.GRU(input_size=128, hidden_size=64, num_layers=3)
        self.hidden_2 = nn.Linear(in_features=64, out_features=32)
        self.hidden_3 = nn.Linear(in_features=32, out_features=16)
        self.out = nn.Linear(in_features=16, out_features=output_sz)

        self.hidden_state = self.reset_hidden()
        self.optimizer = optim.Adam(self.parameters(), lr=LR)


    def reset_hidden(self):
        hidden_sz = self.rnn.hidden_size
        n_layers = self.rnn.num_layers
        hidden_state = torch.zeros(n_layers, 1, hidden_sz)
        self.hidden_state = hidden_state.cuda() if(self.enable_cuda) else hidden_state

        return self.hidden_state


    def forward(self, x):
        x = torch.sigmoid(self.input_layer(x))
        x = torch.tanh(self.hidden_1(x))
        x, self.hidden_state = self.rnn(x.view(1, -1, 128), self.hidden_state.data)
        x = F.relu(self.hidden_2(x.squeeze()))
        x = F.relu(self.hidden_3(x))
        actions_prob = F.softmax(self.out(x), dim=-1)

        return actions_prob


    # 0: Hold, 1: Buy, 2: Sell
    def choose_action(self, state):
        actions_prob = self.forward(state)
        m = Categorical(actions_prob)
        action = m.sample()
        self.actions_log_prob_his.append(m.log_prob(action))

        return action.item()


    def buy(self, count=1):
        curr_close_price = self.env.get_close_price()
        total_price = float(curr_close_price) * count

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
            pass
        elif(action == 1): # Buy
            if(self.cash >= curr_close_price):
                self.buy()
        elif(action == 2): # Sell
            if(self.hold_stock_count >= 0):
                self.sell()

        self.env.step()
        next_state = self.get_state()
        reward = self.portfolio_value() - self.default_cash

        return next_state, reward


    def learn(self, td_error):
        self.optimizer.zero_grad()
        log_prob = self.actions_log_prob_his.popleft()
        self.loss = -log_prob * torch.FloatTensor(td_error).cuda()
        self.loss.backward()
        self.optimizer.step()

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
        torch.save(self, "models/"+name)


    def portfolio_value(self):
        cash = self.hold_stock_count * self.env.get_close_price()
        cash += self.cash

        return cash


    def reset(self):
        self.env.reset()
        self.hidden_state = self.reset_hidden()
        self.actions_log_prob_his.clear()
        self.hold_stock_count = 0
        self.cash = self.default_cash
        self.action = 0
        self.loss = torch.tensor(0)
        self.history = { "DATE": [], "ACTION": [], "LOSS": [], "CASH": [], "PORTFOLIO_VALUE": [] }