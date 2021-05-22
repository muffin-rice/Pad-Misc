from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential
import torch

conv1 = lambda f_in, f_out: nn.Conv3d(f_in, f_out, kernel_size=(1,3,3), stride=1, padding=(0,1,1))
conv2 = lambda f_in, f_out: nn.Conv3d(f_in, f_out, kernel_size=(1,3,3), stride=1, padding=(0,0,0))
bn = lambda f: nn.BatchNorm3d(f)
lr = nn.LeakyReLU

block1 = lambda f_in, f_out : [conv1(f_in, f_out), bn(f_out), lr(inplace=True)]
block2 = lambda f_in, f_out : [conv2(f_in, f_out), bn(f_out), lr(inplace=True)]

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        self.layers1 = Sequential(
            *block1(1, 8),
            *block1(8, 16),
            nn.Conv3d(16, 16, kernel_size=(3, 1, 1)),
            *block2(16, 32),
            *block1(32, 48),
            nn.Conv3d(48, 48, kernel_size=(2, 1, 1)),
            *block1(48, 64),
            *block1(64, 80),
            nn.Flatten()
        )

        self.linear1 = nn.Linear(960, 120)
        self.linear2 = nn.Linear(120, 4)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):

        conv_layers = self.layers1(x)
        l1 = lr()(self.linear1(conv_layers))
        pos_ranks = self.linear2(l1)

        return pos_ranks