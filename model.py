import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBlock(nn.Module):
    """
    A convolutional unit with bn and activation
    """

    def __init__(self, input_size, output_size, kernel_size, stride, activation):
        super(ConvBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.pad = nn.ReplicationPad2d(self.padding)
        self.conv = nn.Conv2d(
            input_size, output_size, kernel_size=kernel_size, stride=stride)
        # self.bn = nn.BatchNorm2d(output_size)
        self.activation = activation

    def forward(self, x):
        x=  self.pad(x)
        x = self.conv(x)
        # x = self.bn(x)
        return self.activation(x)

    def size_out(self, size):
        return (size - self.kernel_size + self.padding * 2) // self.stride + 1


class Model(nn.Module):

    def __init__(self, cfg: Config):
        super(Model, self).__init__()

        self.cfg = cfg

        # global branch
        conv_blocks = []
        size = np.array(cfg.stage_size)
        for s in cfg.global_conv_setting:
            block = ConvBlock(s[0], s[1], s[2], s[3],
                              activation=cfg.activation)
            conv_blocks.append(block)
            size = block.size_out(size)
        self.global_conv = nn.Sequential(*conv_blocks)

        size_out = size[0] * size[1] * \
            cfg.global_conv_setting[-1][1]

        self.global_hidden = nn.Sequential(
            Flatten(),
            nn.Linear(size_out, cfg.hidden_size // 2),
        )

        # local branch
        # conv_blocks = []
        # size = np.array((cfg.local_map_size, cfg.local_map_size))
        # for s in cfg.local_conv_setting:
        #     block = ConvBlock(s[0], s[1], s[2], s[3],
        #                       activation=cfg.activation)
        #     conv_blocks.append(block)
        #     size = block.size_out(size)
        # self.local_conv = nn.Sequential(*conv_blocks)

        # size_out = size[0] * size[1] * \
        #     cfg.local_conv_setting[-1][1]

        self.local_hidden = nn.Sequential(
            Flatten(),
            nn.Linear(cfg.local_map_size * cfg.local_map_size * cfg.local_input_channel, cfg.hidden_size // 2),
        )

        # rnn step
        if cfg.rnn_type == "LSTM":
            rnn = nn.LSTM
        elif cfg.rnn_type == "GRU":
            rnn = nn.GRU
        else:
            raise NotImplementedError
        self.rnn = rnn(
            input_size=cfg.hidden_size,
            hidden_size=cfg.rnn_hidden_size,
            num_layers=cfg.rnn_layer_size,
            dropout=cfg.dropout
        )

        # output step
        self.hidden_a = nn.Linear(cfg.rnn_hidden_size, cfg.hidden_size)
        self.hidden_c = nn.Linear(cfg.rnn_hidden_size, cfg.hidden_size)

        self.mu = nn.Linear(cfg.hidden_size, 2)
        self.sigma = nn.Linear(cfg.hidden_size, 2)
        self.val = nn.Linear(cfg.hidden_size, 1)

    def forward(self, s, h):
        """
        Input:
            s:      state batch. shape (N, C, H, W)
            h:      previous hidden state. shape (rnn_layer_size, N, rnn_hidden_size) or a tuple of two

        Output:
            mu:     mu for the policy distribution. shape (N, 2)
            sigma:  sigma for the policy distribution. shape (N, 2)
            val:    state value of the input state. shape (N, 1)
            next_h: next hidden state. shape (rnn_layer_size, N, rnn_hidden_size) or a tuple of two
        """
        x_global = self.global_hidden(self.global_conv(s[0]))
        x_local = self.local_hidden(s[1])
        x = torch.cat((x_global, x_local), 1)
        x = self.cfg.activation(x)

        x, next_h = self.rnn(x.unsqueeze(0), h)
        x = x.squeeze(0)

        hidden_a = self.cfg.activation(self.hidden_a(x))
        hidden_c = self.cfg.activation(self.hidden_c(x))

        mu = self.cfg.max_a * torch.tanh(self.mu(hidden_a))
        sigma = torch.sigmoid(self.sigma(hidden_a)) + 1e-5

        val = self.val(hidden_c)

        return mu, sigma, val, next_h
