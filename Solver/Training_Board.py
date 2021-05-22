import random
from combo_classes import orbs_remaining
import torch
import copy

def board_valid(board):
    arr = [0,0,0]
    for row in board:
        for elem in row:
            arr[elem] += 1

    if 0 < arr[0] < 3 or 0 < arr[1] < 3 or 0 < arr[2] < 3:
        return False

    return True

class Training_Board:
    def __init__(self, board = None):
        self.finger_loc, self.change, self.prev_action, self.prev_match, self.total_moves, self.board = 0, 0, 0, 0 ,0, 0
        self.reset()
        #1 2 3 4 5
        #6 7 8 9 10 etc

    def get_legal_moves(self) -> {int}:
        '''returns legal moves, 0 left 1 up 2 right 3 down'''
        legal_moves = [True, True, True, True]
        if self.finger_loc[0] == 0: #on the top row
            legal_moves[1] = False
        elif self.finger_loc[0] == 4: # on the bottom row
            legal_moves[3] = False

        if self.finger_loc[1] == 0:  # on the left column
            legal_moves[0] = False
        elif self.finger_loc[1] == 5:  # on the right column
            legal_moves[2] = False

        return legal_moves

    def reset(self, colors = 3, board = None):
        '''resets everything, including the board and sets the finger location to nothing'''
        self.finger_loc = None
        self.change = False
        self.prev_action = -1
        self.prev_match = False
        self.total_moves = 0
        choices = tuple(range(colors))
        if board:
            self.board = board
        else:
            self.board = [[random.choice(choices) for j in range(6)] for i in range(5)]
            while not board_valid(self.board):
                self.board = [[random.choice(choices) for j in range(6)] for i in range(5)]

    def update_state(self, action) -> None:
        '''switches orbs depending on direction, and sets booleans for score evaluation '''
        if not self.get_legal_moves()[action]: #if illegal
            #don't change the previous action because nothing changed
            self.change = False #change variable is for scoring later
        else: #legal move
            self.total_moves += 1 #for score, 25+ moves is bad, but may not preserver markov
            self.change = True #legal action

            prev = (self.finger_loc[0], self.finger_loc[1]) #previous location for swapping
            if action == 0: #left
                self.finger_loc[1] -= 1
            elif action == 1: #up
                self.finger_loc[0] -= 1
            elif action == 2: #right
                self.finger_loc[1] += 1
            elif action == 3: #down
                self.finger_loc[0] += 1
            else:
                raise Exception('Invalid Action')

            self.board[prev[0]][prev[1]], self.board[self.finger_loc[0]][self.finger_loc[1]] = \
            self.board[self.finger_loc[0]][self.finger_loc[1]], self.board[prev[0]][prev[1]]  # swap

            if self.prev_action != -1: #board was not just reset
                if self.prev_action == (action+2)%4: #two opposite actions back to back
                    self.prev_match = True
                else: #normal move
                    self.prev_match = False

            self.prev_action = action


    def get_reward(self) -> (int, bool):
        '''returns the reward, and should be called after update state probably. Might not be Markov. '''
        if orbs_remaining(self.board) == 0: #if completely solved
            return 1.0, True
        if not self.change: #if chose an invalid move
            return -.75, False
        if self.prev_match: #if made a repeat action
            return -.1, False
        if self.change:
            return -.05, False #made a move and it changed

        return 0

    def get_state(self):
        '''returns the board as a 1x4x5x6 Tensor, 3x5x6 being the (tricolor luci) board and
        1x5x6 being the finger location '''
        reds = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        blues, darks = copy.deepcopy(reds), copy.deepcopy(reds)
        stack = [reds, blues, darks, copy.deepcopy(reds)]
        stack[3][self.finger_loc[0]][self.finger_loc[1]] = 2
        for i in range(5):
            for j in range(6):
                stack[self.board[i][j]][i][j] += 1

        stack = torch.Tensor([stack])
        return stack