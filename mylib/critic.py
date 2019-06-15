from collections import deque
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


# Input: Environment State
# Output: State Value
class Critic(nn.Module):
    def __init__(self, input_sz):
        super().__init__()
        self.input_layer = nn.Linear(in_features=input_sz, out_features=128)
        self.hidden_1 = nn.Linear(in_features=128, out_features=128)
        self.rnn = nn.GRU(input_size=128, hidden_size=64, num_layers=3)
        self.hidden_2 = nn.Linear(in_features=64, out_features=32)
        self.hidden_3 = nn.Linear(in_features=32, out_features=16)
        self.out = nn.Linear(in_features=16, out_features=1)

        self.hidden_state = self.reset_hidden()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)


    def reset_hidden(self, cuda=True):
        hidden_sz = self.rnn.hidden_size
        n_layers = self.rnn.num_layers
        hidden_state = torch.zeros(n_layers, 1, hidden_sz)
        self.hidden_state = hidden_state.cuda() if(cuda) else hidden_state

        return self.hidden_state


    def forward(self, x):
        x = torch.FloatTensor(x).cuda().squeeze()
        x = torch.sigmoid(self.input_layer(x))
        x = torch.tanh(self.hidden_1(x))
        x, self.hidden_state = self.rnn(x.view(1, -1, 128), self.hidden_state.data)
        x = F.relu(self.hidden_2(x.squeeze()))
        x = F.relu(self.hidden_3(x))
        value = self.out(x)

        return value


    def learn(self, state, reward, next_state):
        value = self.forward(state)

        self.optimizer.zero_grad()
        reward_tensor = torch.FloatTensor(reward).cuda()
        loss = F.smooth_l1_loss(value, reward_tensor).cuda()
        loss.backward()
        self.optimizer.step()

        next_value = self.forward(next_state)
        td_error = reward + next_value.item() - value.item()

        return td_error