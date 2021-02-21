from torch import nn
import torch.nn.functional as F
import random
import torch
import numpy as np
from collections import namedtuple
from Training_Board import Training_Board
from torch import optim
import copy
from itertools import count
from combo_tools import combo_sim

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 6, kernel_size=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 8, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=2)
        self.bn3 = nn.BatchNorm2d(8)

        #def conv2d_size_out(size, kernel_size=2, stride=1):
        #    return (size - (kernel_size - 1) - 1) // stride + 1
        #convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(6)))
        #convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(5)))
        #linear_input_size = convw * convh * 30

        self.head = nn.Linear(48, 4)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class Agent:
    def __init__(self, batch_size = 4, gamma = .999, eps_start = .9, eps_end = .05, eps_decay = 200, target_update = 10,
                 memory_size = 5000):
        self.batch_size, self.gamma, self.eps_start, self.eps_end, self.eps_decay, self.target_update \
            = batch_size, gamma, eps_start, eps_end, eps_decay, target_update

        self.steps, self.threshold, self.policy, self.target = 0, 0, DQN(), DQN()
        self.optimizer = optim.RMSprop(self.policy.parameters())
        self.memory = ReplayMemory(memory_size)
        self.training_history = []

    def select_action(self, state):
        sample = random.random()
        self.threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1 * self.steps / self.eps_decay)
        self.steps += 1
        if sample > self.threshold:
            return self.target(state).max(1)[1].view(1) #maybe view(1,1)?

        else: #random
            return torch.tensor([random.randrange(4)], dtype = torch.long)

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def get_weights(self):
        return self.target.state_dict()

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        batch = Transition(*zip(*self.memory.sample(self.batch_size)))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).view(self.batch_size, 1)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)

        state_action_values = self.policy(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size).view(self.batch_size, 1)
        next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values) #might be missing a dimensions

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, episodes = 10000):
        tb = Training_Board()
        num_starts = 4
        num_boards = episodes // num_starts
        thresh = num_boards // 3
        c = 2
        #1/3 bicolor 2/3 tricolor to train better?
        for i in range(episodes // num_starts):
            print(f'On board #{i}; board is\n')
            if i > thresh:
                c = 3
            choices = tuple(range(c))
            old_board = [[random.choice(choices) for j in range(6)] for i in range(5)] #save the board
            combo_sim.print_board(old_board)

            for j in range(num_starts): #randomize different starting positions to simulate choosing a starting loc
                tb.reset(colors = c, board = copy.deepcopy(old_board)) #set the board back to original
                tb.finger_loc = [random.randrange(5), random.randrange(6)] #randomize starting pos
                print('Reset board')
                combo_sim.print_board(tb.board)
                print(f'Finger loc is now {tb.finger_loc}\n')
                state = tb.get_state() #initial state

                for t in count(): #try to solve the board
                    action = self.select_action(state)
                    tb.update_state(action)
                    reward, done = tb.get_reward()
                    reward = torch.tensor([reward])

                    new_state = tb.get_state()
                    if random.random() < .01: #so it doesnt push a lot
                        self.memory.push(state, action, new_state, reward)

                    state = new_state
                    self.optimize()

                    if self.steps % 10000 == 0: #tracking process
                        print(f'Threshold is {self.threshold} on step {self.steps}')
                        print(f'On board {i}, move # {t+1}')
                        combo_sim.print_board(tb.board)
                        print(f'Position: {tb.finger_loc}\n')

                    if self.steps % self.target_update == 0:
                        self.update_target()

                    if done:
                        self.memory.push(state, action, new_state, reward)
                        print(f'Finished board {i} in {t+1} moves.')
                        combo_sim.print_board(tb.board)
                        print(f'Position: {tb.finger_loc}\n')
                        self.training_history.append(t + 1)
                        break