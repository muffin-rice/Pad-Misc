import torch.nn.functional as F
import random
import torch
import numpy as np
from collections import namedtuple
from Training_Board import Training_Board
from torch import optim
import copy
from itertools import count
from combo_classes import get_board_string
from DQN import DQN
import logging
from ReplayMemory import ReplayMemory

logging.basicConfig(filename = 'logs/training_logs.log', level = logging.INFO, format='%(message)s')
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Agent:
    def __init__(self, batch_size = 4, gamma = .999, eps_start = .95, eps_end = .05, eps_decay = 200, target_update = 10,
                 memory_size = 5000):
        self.batch_size, self.gamma, self.eps_start, self.eps_end, self.eps_decay, self.target_update \
            = batch_size, gamma, eps_start, eps_end, eps_decay, target_update

        self.steps, self.threshold, self.policy, self.target = 0, eps_start, DQN(), DQN()
        self.optimizer = optim.RMSprop(self.policy.parameters())
        self.memory = ReplayMemory(memory_size)
        self.training_history = []

    def select_action(self, state):
        #returns a 4-tensor

        if random.random() > self.threshold:
            choice = self.target(torch.unsqueeze(state, 0))
            return torch.tensor([torch.argmax(choice)], dtype = torch.int64)

        else: #random
            return torch.tensor([random.randrange(4)], dtype = torch.int64)

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def get_weights(self):
        return self.target.state_dict()

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        batch = Transition(*zip(*self.memory.sample(self.batch_size)))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).view(self.batch_size, 1)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

        state_batch = torch.stack(batch.state)
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
        num_starts = 2
        c, cutoff, capa, o_cap = 2, 75, 5, 4

        for i in range(episodes // num_starts):
            if i > episodes // num_starts // cutoff:
                random_colors = range(3)
            else:
                random_colors = random.sample(range(3), c)

            old_board = [[random.choice(random_colors) for jj in range(6)] for ii in range(5)] #save the board
            logging.info(f'On board #{i}; board is\n{get_board_string(old_board)}\n')

            for j in range(num_starts): #randomize different starting positions to simulate choosing a starting loc
                tb.reset(colors = c, board = copy.deepcopy(old_board)) #set the board back to original
                tb.finger_loc = [random.randrange(5), random.randrange(6)] #randomize starting pos
                state = tb.get_state() #initial state

                capa_i = 0

                for t in count(): #try to solve the board
                    action = self.select_action(state)
                    tb.update_state(action.item())
                    reward, done = tb.get_reward()
                    reward = torch.tensor([reward])

                    self.steps += 1

                    new_state = tb.get_state()
                    if capa_i < capa and random.random() < .01: #so it doesnt push a lot
                        capa_i += 1
                        self.memory.push(state, action, new_state, reward)

                    state = new_state
                    if capa_i < capa:
                        self.optimize()

                    if self.steps % 10000 == 0: #tracking process
                        logging.info(f'On board {i}, move # {t+1}')
                        logging.info(get_board_string(tb.board))
                        logging.info(f'Position: {tb.finger_loc}\n')

                    if self.steps % self.target_update == 0:
                        self.update_target()

                    if done:
                        self.memory.push(state, action, new_state, reward)
                        logging.info(f'Finished board {i} in {t+1} moves.')
                        logging.info(get_board_string(tb.board))
                        logging.info(f'Position: {tb.finger_loc}\n')
                        self.training_history.append(t + 1)
                        break

            self.steps += 1
            self.threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1 * self.steps / self.eps_decay)
