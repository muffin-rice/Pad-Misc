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
                        # the first clause ensures self is beneath the other combo (but not too far beneath)
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