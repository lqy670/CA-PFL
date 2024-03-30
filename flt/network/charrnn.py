#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable

__all__ = ["chargru", "charlstm"]


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="gru", n_layers=1, device=None):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.device = device

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        encoded = self.encoder(input)
        output, _ = self.rnn(encoded)
        output = self.decoder(output[:, -1, :])
        return output

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (
                Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).to(self.device),
                Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).to(self.device)
            )
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).to(self.device)


def chargru(input_size: int = 80, hidden_size: int = 128, output_size: int = 80, n_layers = 2, *args, **kwargs):
    return CharRNN(input_size, hidden_size, output_size, model="gru", n_layers=n_layers)


def charlstm(input_size: int = 80, hidden_size: int = 128, output_size: int = 80, n_layers = 2, *args, **kwargs):
    return CharRNN(input_size, hidden_size, output_size, model="lstm", n_layers=n_layers)