from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


# Input: Environment State
# Output: Actions' Probability
class Actor(nn.Module):
    def __init__(self, input_sz, output_sz, env):
        super().__init__()
        self.input_layer = nn.Linear(in_features=input_sz, out_features=128)
        self.hidden_1 = nn.Linear(in_features=128, out_features=128)
        self.rnn = nn.GRU(input_size=128, hidden_size=64, num_layers=3)
        self.hidden_2 = nn.Linear(in_features=64, out_features=32)
        self.hidden_3 = nn.Linear(in_features=32, out_features=16)
        self.out = nn.Linear(in_features=16, out_features=output_sz)

        self.env = env
        self.hidden_state = self.reset_hidden()
        self.actions_log_prob_his = deque()


    def reset_hidden(self, cuda=True):
        hidden_sz = self.rnn.hidden_size
        n_layers = self.rnn.num_layers
        hidden_state = torch.zeros(n_layers, 1, hidden_sz)
        self.hidden_state = hidden_state.cuda() if(cuda) else hidden_state

        return self.hidden_state


    def forward(self, x):
        x = torch.tensor(x).cuda()
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


    def step(self, action):
        self.env.step()
        next_state = self.env.get_state()
        reward = 0
        return next_state, reward


    def learn(self, state, action, td_error):
        pass