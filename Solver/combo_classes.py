import copy

rows,columns = 5,6

def get_board_string (board : [[str]]):
    return '\n'.join(''.join(str(orb) for orb in row) for row in board)

def remove_overlapping(all_combos):
    to_purge = set()
    for i in range(len(all_combos) - 1):
        for j in range(i + 1, len(all_combos)):
            if all_combos[i].att == all_combos[j].att and all_combos[i] in all_combos[j]:
                all_combos[i].swallow(all_combos[j])
                to_purge.add(j)

    for i, j in enumerate(sorted(to_purge)):
        all_combos.pop(j - i)

    to_purge = set() #do it twice for the rare ++ formation
    for i in range(len(all_combos) - 1):
        for j in range(i + 1, len(all_combos)):
            if all_combos[i].att == all_combos[j].att and all_combos[i] in all_combos[j]:
                all_combos[i].swallow(all_combos[j])
                to_purge.add(j)

    for i, j in enumerate(sorted(to_purge)):
        all_combos.pop(j - i)

    return all_combos

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

    all_combos = remove_overlapping(all_combos)

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

def remove_combos(board: [[str]], combos: [[(int, int)]]):
    #replaces coordinates in matches with blanks
    for combo in combos:
        for coord in combo:
            board[coord[0]][coord[1]] = ' '

def orbs_remaining(board):
    board = copy.deepcopy(board)
    combo_coords = find_combos(board)
    orbs_used = 0

    while len(combo_coords) > 0:
        orbs_used += sum([len(match) for match in combo_coords])
        remove_combos(board, combo_coords)
        fall_all(board)
        combo_coords = find_combos(board)

    return rows*columns - orbs_used

class Match:
    # a match is just gonna be a collection of combos

    def __init__(self, root: (int, int), length: int, direction: str, attribute: str):
        assert direction in ['h', 'v']
        assert length >= 3
        self.combos = [(root, length, direction)]
        self.att = attribute

    def __contains__(self, other):
        if other.att != self.att:
            return False

        # each combo is a tuple in the form (root, length, direction), with the root being in the top left
        for combo in self.combos:
            for others in other.combos:
                # if they're in opposite directions; ie L pattern
                if combo[2] != others[2]:
                    if combo[2] == 'h':
                        # the first clause ensures self is just beneath the other combo
                        # the second clause ensures that self is to the left of the other combo
                        if ((0 <= combo[0][0] - others[0][0] < others[1]) and (
                                0 <= others[0][1] - combo[0][1] < combo[1])):
                            return True
                    else:
                        # should just be the opposite, with combo being vertical and other horizontal
                        # the first clause ensures other is beneath self
                        # second ensures other is to the left of self
                        if ((0 <= others[0][0] - combo[0][0] < combo[1]) and (
                                0 <= combo[0][1] - others[0][1] < others[1])):
                            return True

                # check for the weird 'snakelike pattern' https://pad.dawnglare.com/?s=rEPwZq0
                else:
                    if combo[2] == 'h':
                        # check they're one above/below each other, and then that they fit in the 'box'
                        if ((-1 <= combo[0][0] - others[0][0] <= 1) and (
                                (0 <= combo[0][1] - others[0][1] < others[1]) or (
                                0 <= others[0][1] - combo[0][1] < combo[1]))):
                            return True
                    else:
                        if ((-1 <= combo[0][1] - others[0][1] <= 1) and (
                                (0 <= combo[0][0] - others[0][0] < others[1]) or (
                                0 <= others[0][0] - combo[0][0] < combo[1]))):
                            return True

        return False

    def get_coord(self):
        """returns a set of the coordinates in the combo, starting from the root"""
        coordinate_box = set()
        for combo in self.combos:
            if combo[2] == 'h':
                coordinate_box.update(set((combo[0][0], combo[0][1] + i) for i in range(combo[1])))
            else:
                coordinate_box.update(set((combo[0][0] + i, combo[0][1]) for i in range(combo[1])))

        return coordinate_box

    def get_roots(self):
        return list({combo[0] for combo in self.combos})

    def swallow(self, other):
        self.combos += other.combos
        self.combos = list(set(self.combos))

    def __str__(self):
        return f'The combos in this match are: {self.combos} with attribute {self.att}'