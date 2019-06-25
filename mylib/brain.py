import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


class Brain(nn.Module):
    def __init__(self, input_sz, output_sz, lr=1e-3, softmax=True, enable_cuda=True):
        super().__init__()
        self.softmax = softmax
        self.enable_cuda = enable_cuda

        self.input_layer = nn.Linear(in_features=input_sz, out_features=128)
        self.hidden_1 = nn.Linear(in_features=128, out_features=128)
        self.rnn = nn.GRU(input_size=128, hidden_size=64, num_layers=3)
        self.hidden_2 = nn.Linear(in_features=64, out_features=32)
        self.hidden_3 = nn.Linear(in_features=32, out_features=16)
        self.out = nn.Linear(in_features=16, out_features=output_sz)

        self.hidden_state = self.reset_hidden()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)


    def forward(self, x):
        if(self.enable_cuda == True):
            x = torch.FloatTensor(x).cuda()
        else:
            x = torch.FloatTensor(x)
        x = torch.sigmoid(self.input_layer(x))
        x = torch.tanh(self.hidden_1(x))
        x, self.hidden_state = self.rnn(x.view(1, -1, 128), self.hidden_state.data)
        x = F.relu(self.hidden_2(x.squeeze()))
        x = F.relu(self.hidden_3(x))
        out = self.out(x)
        if(self.softmax): out = torch.softmax(out, dim=-1)

        return  out


    def reset_hidden(self):
        hidden_sz = self.rnn.hidden_size
        n_layers = self.rnn.num_layers
        hidden_state = torch.zeros(n_layers, 1, hidden_sz)
        self.hidden_state = hidden_state.cuda() if(self.enable_cuda) else hidden_state

        return self.hidden_state