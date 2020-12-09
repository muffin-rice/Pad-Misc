import random
import numpy as np
import copy
from match import Match

# import scipy.stats as stats

columns = 6
rows = 5
ITERATIONS = 10000
sf_distribution = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 0, 0]

def gen_random_board():
    board = []
    for i in range(rows):
        board.append([])
        for j in range(columns):
            board[i].append(random.choices('RBGLDHJP', sf_distribution)[0])

    return board

def print_board(board: [[str]]):  # debugging purposes
    print('\n'.join(''.join(orb for orb in row) for row in board))


def find_combos(board: [[str]]):
    '''returns a list of lists of tuples: board -> matches -> coordinates'''
    all_combos = []
    # first go horizontal
    for i, row in enumerate(board):
        counter = 1
        prev_letter = '*'
        for j in range(columns):
            if prev_letter == row[j]:
                counter += 1
            else:
                if counter >= 3 and prev_letter != ' ':
                    all_combos.append(Match((i, j - counter), counter, 'h', prev_letter))
                prev_letter = row[j]
                counter = 1

            if j == columns - 1 and counter >= 3 and prev_letter != ' ':
                all_combos.append(Match((i, j - counter + 1), counter, 'h', prev_letter))

    # then go vertical
    for j in range(columns):
        counter = 1
        prev_letter = '*'
        for i in range(rows):
            if prev_letter == board[i][j]:
                counter += 1
            else:
                if counter >= 3 and prev_letter != ' ':
                    all_combos.append(Match((i - counter, j), counter, 'v', prev_letter))
                prev_letter = board[i][j]
                counter = 1

            if i == rows - 1 and counter >= 3 and prev_letter != ' ':
                all_combos.append(Match((i - counter + 1, j), counter, 'v', prev_letter))

    # see if they overlap using __contains__ dunder in Match class
    to_purge = set()  # the indexes of combos that have been swallowed
    for i in range(len(all_combos) - 1):
        for j in range(i + 1, len(all_combos)):
            if all_combos[i].att == all_combos[j].att and all_combos[i] in all_combos[j]:
                all_combos[i].swallow(all_combos[j])
                to_purge.add(j)

    for i, j in enumerate(sorted(to_purge)):
        all_combos.pop(j - i)

    # coordinates should never overlap
    return [combo.get_coord() for combo in all_combos]


def fall_all(board: [[str]]):
    for j in range(columns):
        col = [board[rows-i-1][j] for i in range(rows) if board[rows-i-1][j] != ' ']
        for i in range(rows):
            if i < len(col):
                board[rows-i-1][j] = col[i]
            else:
                board[rows-i-1][j] = ' '

def create_skyfalls(board: [[str]]):
    for i in range(rows):
        for j in range(columns):
            if board[i][j] == ' ':
                board[i][j] = random.choices('RBGLDHJP', sf_distribution)[0]

def remove_combos(board: [[str]], combos: [[(int, int)]]):
    #replaces coordinates in matches with blanks
    for combo in combos:
        for coord in combo:
            board[coord[0]][coord[1]] = ' '

def orbs_remaining(board_string):
    board = [[board_string[i * columns + j] for j in range(columns)] for i in range(rows)]
    combo_coords = find_combos(board)
    orbs_used = 0

    while len(combo_coords) > 0:
        orbs_used += sum([len(match) for match in combo_coords])
        remove_combos(board, combo_coords)
        fall_all(board)
        combo_coords = find_combos(board)

    return rows*columns - orbs_used

def combo_sim(board_string: str, x_sf: (int, int), print_stats=True, skyfall=True) -> \
        (str, int, int, int, int):
    '''board size is 0 for 6x5 and 1 for 7x6;
    x_sf is how many skyfalls you want over how many turns'''
    assert len(board_string) == rows * columns

    perm_board = [[board_string[i * columns + j] for j in range(columns)] for i in range(rows)]

    # no skyfall combo count is on cc_list[0]
    cc_list = []  # number of combos per iteration
    guard_breaks = 0 # number of guard break activations

    for i in range(ITERATIONS+1):
        board = copy.deepcopy(perm_board)
        total_combos = 0
        guard_break = False
        combo_coords = find_combos(board)

        while len(combo_coords) > 0:
            total_combos += len(combo_coords)
            remove_combos(board, combo_coords)
            fall_all(board)
            if cc_list: #first one is noSF
                create_skyfalls(board)
            combo_coords = find_combos(board)

        cc_list.append(total_combos)

    base_combo_count = cc_list[0]
    cc_list = cc_list[1:]

    sum_list = [sum(cc_list[i:(i + x_sf[1])]) - x_sf[1] * base_combo_count for i in range(ITERATIONS - x_sf[1])]
    avg, avg_sf, min_c, p = (np.mean(cc_list), np.mean(cc_list) - base_combo_count, min(cc_list),
                             sum(x > x_sf[0] for x in sum_list) / ITERATIONS)

    if print_stats:
        print(f'board was {board_string}')
        print(f'The mean combos is {avg}')
        print(f'The mean number of skyfalls is {avg_sf}')
        print(f'The min number of combos is {min_c}')
        print(f'The observed probability of getting {x_sf[0]} skyfalls over {x_sf[1]} turns is {p}')

    return board_string, avg, avg_sf, min_c, p

def no_combos():
    #returns probability of no combos
    free = 0
    for i in range(ITERATIONS):
        board = gen_random_board()
        if len(find_combos(board)) == 0:
            free+=1

    return free/ITERATIONS


def set_skyfalls(distribution: [int] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 0, 0]):
    '''sets the skyfalls in order of RBGLDHJP'''
    global sf_distribution
    sf_distribution = distribution

def set_to_76():
    global rows
    global columns
    rows = 6
    columns = 7

if __name__ == '__main__':
    #set_skyfalls([1/4, 1/4, 1/4, 1/4, 0, 0, 0, 0]) #luci is quadcolor
    print(no_combos())
    set_skyfalls()
    board_string = "DDDLLLLLLDDDDDDLLLLLLDDDDDDLLL"
    assert orbs_remaining(board_string) == 0
    assert orbs_remaining('bgrgbgbrrbrggrrbrgbbrrbbggggrg') == 13
    combo_sim(board_string, (15, 2), print_stats=True, skyfall=True)
