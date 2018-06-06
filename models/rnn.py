# encoding: utf-8

import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, h_state):

        r_out, h_state = self.rnn(x, h_state)
        
        # hidden_size = h_state[-1].size(-1)

        r_out = r_out.view(-1, self.hidden_size)
        outs = self.fc(r_out)
        return outs, h_state