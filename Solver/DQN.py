from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential
import torch

conv1 = lambda f_in, f_out: nn.Conv2d(f_in, f_out, kernel_size=3, stride=1, padding=1)
conv2 = lambda f_in, f_out: nn.Conv2d(f_in, f_out, kernel_size=3, stride=1, padding=0)
bn = lambda f: nn.BatchNorm2d(f)
lr = nn.LeakyReLU

block1 = lambda f_in, f_out : [lr(inplace=True), bn(f_out), conv1(f_in, f_out)]
block2 = lambda f_in, f_out : [lr(inplace=True), bn(f_out), conv2(f_in, f_out)]

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()

        self.layers1 = Sequential(
            *block1(3, 8),
            *block1(8, 16),
            nn.Conv3d(16, 16, kernel_size=(2, 1, 1)),
            *block2(16, 32),
            *block1(32, 48),
            nn.Conv3d(48, 48, kernel_size=(2, 1, 1)),
            *block1(48, 64),
            nn.Flatten()
        )

        self.linear1 = nn.Linear(12, 4)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        conv_layers = self.layers1(x)
        pos_ranks = self.linear1(conv_layers)

        return pos_ranks