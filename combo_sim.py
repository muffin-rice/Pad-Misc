import random
import numpy as np 
import copy 
import scipy.stats as stats

columns = 6
rows = 5
ITERATIONS = 100000
sf_distribution = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 0, 0]

class Match: 
    #a match is just gonna be a collection of combos

    def __init__(self, root : (int, int), length : int, direction : str, attribute : str): 
        assert direction in ['h', 'v']
        assert length >= 3
        self.combos = [(root, length, direction)]
        self.att = attribute
    
    def __contains__(self, other): 
        if other.att != self.att: 
            return False 

        #each combo is a tuple in the form (root, length, direction), with the root being in the top left 
        for combo in self.combos: 
            for others in other.combos: 
                #if they're in opposite directions; ie L pattern 
                if combo[2] != others[2]: 
                    if combo[2] == 'h': 
                        #the first clause ensures self is beneath the other combo (but not too far beneath)
                        #the second clause ensures that self is to the left of the other combo 
                        if ((0 <= combo[0][0] - others[0][0] < others[1]) and (0 <= others[0][1] - combo[0][1] < combo[1])): 
                            return True
                    else: 
                        #should just be the opposite, with combo being vertical and other horizontal
                        #the first clause ensures other is beneath self 
                        #second ensures other is to the left of self 
                        if ((0 <= others[0][0] - combo[0][0] < combo[1]) and (0 <= combo[0][1] - others[0][1] < others[1])): 
                            return True
                
                #check for the weird 'snakelike pattern' https://pad.dawnglare.com/?s=rEPwZq0
                else: 
                    if combo[2] == 'h': 
                        #check they're one above/below each other, and then that they fit in the 'box' 
                        if ((-1 <= combo[0][0] - others[0][0] <= 1) and ((0 <= combo[0][1] - others[0][1] < others[1]) or (0 <= others[0][1] - combo[0][1] < combo[1]))): 
                            return True
                    else: 
                        if ((-1 <= combo[0][1] - others[0][1] <= 1) and ((0 <= combo[0][0] - others[0][0] < others[1]) or (0 <= others[0][0] - combo[0][0] < combo[1]))): 
                            return True
        
        return False 

    def get_coord(self): 
        '''returns a list of the coordinates in the combo, starting from the root'''
        coordinate_box = set()
        for combo in self.combos: 
            if combo[2] == 'h': 
                coordinate_box.update(set((combo[0][0], combo[0][1] + i) for i in range(combo[1]))) 
            else: 
                coordinate_box.update(set((combo[0][0] + i, combo[0][1]) for i in range(combo[1]))) 
        
        return list(coordinate_box)
    
    def get_roots(self): 
        return list({combo[0] for combo in self.combos})

    
    def swallow(self, other): 
        self.combos += other.combos


def print_board(board : [[str]]):  # debugging purposes
    for row in board: 
        for orb in row: 
            print(orb, end = '')
        print()

def find_combos(board : [[str]]): 
    '''returns a list of lists of tuples with coordinates of every combo'''
    all_combos = []
    #first go horizontal
    for i, row in enumerate(board): 
        counter = 1 
        prev_letter = '*'
        for j in range(columns): 
            if prev_letter == row[j]: 
                counter+=1 
            else: 
                if counter>=3: 
                    all_combos.append(Match((i, j - counter), counter, 'h', prev_letter))
                prev_letter = row[j] 
                counter = 1
            
            if j == columns-1 and counter >= 3: 
                all_combos.append(Match((i, j - counter + 1), counter, 'h', prev_letter))

    #then go vertical 
    for j in range(columns): 
        counter = 1 
        prev_letter = '*'
        for i in range(rows): 
            if prev_letter == board[i][j]: 
                counter+=1 
            else: 
                if counter>=3: 
                    all_combos.append(Match((i - counter, j), counter, 'v', prev_letter))
                prev_letter = board[i][j]
                counter = 1 
            
            if i == rows-1 and counter>=3: 
                all_combos.append(Match((i - counter + 1, j), counter, 'v', prev_letter ))

    #print('Roots are: ' + str([combo.get_roots() for combo in all_combos]))
    #see if they overlap using __contains__ dunder in Match class 
    to_purge = set() #the indexes of combos that have been swallowed 
    for i in range(len(all_combos) - 1):
        for j in range(i+1, len(all_combos)): 
            if all_combos[i].att == all_combos[j].att and all_combos[i] in all_combos[j]: 
                all_combos[i].swallow(all_combos[j])
                to_purge.add(j)
    
    for i, j in enumerate(sorted(to_purge)): 
        all_combos.pop(j-i)

    #coordinates should never overlap
    return [combo.get_coord() for combo in all_combos]


def fall_all(board : [[str]]): 
    #start from the bottom of screen 
    for j in range(len(board[0])): 
        counter = len(board)
        for i in range(len(board)-1, 0, -1): 
            #counter <= 0 means that the whole column is empty 
            while(board[i][j] == ' ' and counter > 0): 
                for k in range(i, 0, -1): 
                    board[k][j] = board[k-1][j]
                    board[k-1][j] = ' '
                
                counter-=1 
                #print_board(board)
                #input()
            
            counter -= 1 

def remove_combos(board : [[str]], combos : [[(int, int)]], skyfall = True):  # removing combo'd orbs
    for combo in combos:
        for coord in combo:
            board[coord[0]][coord[1]] = ' '  # just using " " to represent the board 

    fall_all(board)

    if skyfall: 
        for i in range(rows): 
            for j in range(columns): 
                if board[i][j] == ' ': 
                    board[i][j] = random.choices('RBGLDHJP', sf_distribution)[0]

def find_nosf_combos(board : [[str]]): 
    all_combos = []
    #first go horizontal
    for i, row in enumerate(board): 
        counter = 1 
        prev_letter = '*'
        for j in range(columns): 
            if row[j] == ' ': 
                continue 
            if prev_letter == row[j]: 
                counter+=1 
            else: 
                if counter>=3: 
                    all_combos.append(Match((i, j - counter), counter, 'h', prev_letter))
                prev_letter = row[j] 
                counter = 1
            if j == columns-1 and counter >= 3: 
                all_combos.append(Match((i, j - counter + 1), counter, 'h', prev_letter))
    for j in range(columns): 
        counter = 1 
        prev_letter = '*'
        for i in range(rows): 
            if (board[i][j] == ' '): 
                continue 
            if prev_letter == board[i][j]: 
                counter+=1 
            else: 
                if counter>=3: 
                    all_combos.append(Match((i - counter, j), counter, 'v', prev_letter))
                prev_letter = board[i][j]
                counter = 1 
            if i == rows-1 and counter>=3: 
                all_combos.append(Match((i - counter + 1, j), counter, 'v', prev_letter ))

    #print('Roots are: ' + str([combo.get_roots() for combo in all_combos]))
    to_purge = set()
    for i in range(len(all_combos) - 1):
        for j in range(i+1, len(all_combos)): 
            if all_combos[i].att == all_combos[j].att and all_combos[i] in all_combos[j]: 
                all_combos[i].swallow(all_combos[j])
                to_purge.add(j)
    for i, j in enumerate(sorted(to_purge)): 
        all_combos.pop(j-i)
    return [combo.get_coord() for combo in all_combos]

def set_skyfalls(distribution : [int]): 
    '''sets the skyfalls in order of RBGLDHJP'''
    global sf_distribution
    sf_distribution = distribution

def calculate_base_combo(board : [[str]]): 
    total_combos = 0 
    combo_coords = find_nosf_combos(board)
    while(len(combo_coords) > 0):
        total_combos += len(combo_coords)
        remove_combos(board, combo_coords, skyfall = False)
        combo_coords = find_nosf_combos(board)
    
    return total_combos

def combo_sim(board_string : str, board_size : (0,1), x_sf : (int, int), print_stats = True, skyfall = True) -> (str, int, int, int, int): 
    '''board size is 0 for 6x5 and 1 for 7x6;
    x_sf is how many skyfalls you want over how many turns''' 
    global rows 
    global columns 
    if board_size == 1: 
        rows = 6
        columns = 7
    
    assert len(board_string) == rows * columns 
    perm_board = [[board_string[i*columns + j] for j in range(columns)] for i in range(rows)]

    #no skyfall combo count  
    base_combo_count = calculate_base_combo(copy.deepcopy(perm_board))

    cc_list = [] #number of combos per iteration 
    guard_break_list = [] 

    if skyfall: 
    
        for i in range(ITERATIONS): 

            #print(f'Iteration {i}')

            board = copy.deepcopy(perm_board)
            #print(board)

            total_combos = 0 
            guard_break = False
            combo_coords = find_combos(board)

            while(len(combo_coords) > 0):
                #print('Combo Coordinates:')
                #print(combo_coords)
                #print('Board:')
                #print_board(board)
                #input()
                total_combos += len(combo_coords)
                remove_combos(board, combo_coords)
                combo_coords = find_combos(board)

            cc_list.append(total_combos)

        sum_list = [sum(cc_list[i:(i+x_sf[1])]) - x_sf[1]*base_combo_count for i in range(ITERATIONS-x_sf[1])]
        avg, avg_sf, min_c, p = (np.mean(cc_list), np.mean(cc_list) - base_combo_count, min(cc_list), sum(x > x_sf[0] for x in sum_list)/ITERATIONS)
        if print_stats: 
            print(f'board was {board_string}')
            print(f'The mean combos is {avg}')
            print(f'The mean number of skyfalls is {avg_sf}')
            print(f'The min number of combos is {min_c}')
            print(f'The observed probability of getting {x_sf[0]} skyfalls over {x_sf[1]} turns is {p}')
        
        return (board_string, avg, avg_sf, min_c, p)
    
    else: 
        print(f'The combo count is {base_combo_count}')
        return (base_combo_count)
    



if __name__ == '__main__': 

    board_string = "DDDLLLLLLDDDDDDLLLLLLDDDDDDLLL"
    #10c cascade: DLDLLDLLDLLDDLLDDLDDLLLDDLDDDL
    #10c inverted: LDLDDLDDLDDLLDDLLDLLDDDLLDLLLD
    #10c: DDDLLLLLLDDDDDDLLLLLLDDDDDDLLL

    set_skyfalls([1/4, 1/4, 1/4, 0, 1/4, 0, 0, 0])
    #normal: [1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 0, 0]
    #xellos: [1/4, 1/4, 1/4, 0, 1/4, 0, 0, 0]
    #off color from lin (ie jeanne enma): [1/4, 1/4, 1/4, 0, 0, 1/4, 0, 0]
    '''
    LDLDDL
    DDLDDL
    LDDLLD
    LLDDDL
    LDLLLD
    '''
    combo_sim(board_string, 0, (15,2), print_stats = True, skyfall = True)
