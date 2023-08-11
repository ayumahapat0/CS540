import copy
import random

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]


    def heuristic(self, state, piece):
        if piece == 'b':
            my_piece = 'b'
            opp_piece = 'r'

        if piece == 'r':
            my_piece = 'r'
            opp_piece = 'b'

        my_count = 0
        my_max = 0
        opp_count = 0
        opp_max = 0

        # horizontals


    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        drop_phase = True   # TODO: detect drop phase

        if not drop_phase:
            # TODO: choose a piece to move and remove it from the board
            # (You may move this condition anywhere, just be sure to handle it)
            #
            # Until this part is implemented and the move list is updated
            # accordingly, the AI will not follow the rules after the drop phase!
            pass

        # select an unoccupied space randomly
        # TODO: implement a minimax algorithm to play better
        move = []
        (row, col) = (random.randint(0,4), random.randint(0,4))
        while not state[row][col] == ' ':
            (row, col) = (random.randint(0,4), random.randint(0,4))

        # ensure the destination (row,col) tuple is at the beginning of the move list
        move.insert(0, (row, col))
        return move


    def succ(self, state, piece):
        self.game_value(state)
        mypiece = piece
        succ = []
        drop_phase = True

        num_B = sum(i.count('b') for i in state)
        num_R = sum(i.count('r') for i in state)

        if num_B >= 4 and num_R >= 4:
            drop_phase = False


        if not drop_phase:
            for i in range(len(state)):
                for j in range(len(state)):
                    if state[i][j] == mypiece:
                        succ.append(self.up(state, i, j))
                        succ.append(self.down(state, i, j))
                        succ.append(self.left(state, i, j))
                        succ.append(self.right(state, i, j))
                        succ.append(self.upleft(state, i, j))
                        succ.append(self.upright(state, i, j))
                        succ.append(self.downright(state, i, j))
                        succ.append(self.downleft(state, i, j))
            return list(succ)

        for i in range(len(state)):
            for j in range(len(state)):
                new_state = copy.deepcopy(state)
                if new_state[i][j] == ' ':
                    new_state[i][j] = mypiece
                    succ.append(new_state)

        return list(succ)

    def up(self, state, row, col):
        new_state = copy.deepcopy(state)
        if row - 1 >= 0 and new_state[row-1][col] ==' ':
            new_state[row-1][col] = new_state[row][col]
            new_state[row][col] = ' '

        return new_state

    def down(self, state, row, col):
        new_state = copy.deepcopy(state)
        if row + 1 < len(state) and new_state[row + 1][col] == ' ':
            new_state[row + 1][col] = new_state[row][col]
            new_state[row][col] = ' '

        return new_state

    def left(self, state, row, col):
        new_state = copy.deepcopy(state)
        if col - 1 >= 0 and new_state[row][col - 1] == ' ':
            new_state[row][col - 1] = new_state[row][col]
            new_state[row][col] = ' '

        return new_state

    def right(self, state, row, col):
        new_state = copy.deepcopy(state)
        if col + 1 < len(state) and new_state[row ][col + 1] == ' ':
            new_state[row][col + 1] = new_state[row][col]
            new_state[row][col] = ' '

        return new_state

    def upleft(self, state, row, col):
        new_state = copy.deepcopy(state)
        if row - 1 >= 0 and col - 1 >= 0 and new_state[row - 1][col - 1] == ' ':
            new_state[row - 1][col - 1] = new_state[row][col]
            new_state[row][col] = ' '

        return new_state

    def upright(self, state, row, col):
        new_state = copy.deepcopy(state)
        if row - 1 >= 0 and col + 1 < len(state) and new_state[row - 1][col + 1] == ' ':
            new_state[row - 1][col + 1] = new_state[row][col]
            new_state[row][col] = ' '

        return new_state

    def downleft(self, state, row, col):
        new_state = copy.deepcopy(state)
        if row + 1 < len(state) and col - 1 >= 0 and new_state[row + 1][col - 1] == ' ':
            new_state[row + 1][col - 1] = new_state[row][col]
            new_state[row][col] = ' '

        return new_state

    def downright(self, state, row, col):
        new_state = copy.deepcopy(state)
        if row + 1 < len(state) and col + 1 < len(state) and new_state[row + 1][col + 1] == ' ':
            new_state[row + 1][col + 1] = new_state[row][col]
            new_state[row][col] = ' '

        return new_state

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # TODO: check \ diagonal wins

        # check main \ diagonal
        for i in range(2):
            if state[i][i] != ' ' and state[i][i] == state[i+1][i+1] == state[i+2][i+2] == state[i+3][i+3]:
                return 1 if state[i][i] == self.my_piece else -1

        # check the two smaller \ diagonals
        if state[1][0] != ' ' and state[1][0] == state[2][1] == state[3][2] == state[4][3]:
            return 1 if state[1][0] == self.my_piece else -1

        if state[0][1] != ' ' and state[0][1] == state[1][2] == state[2][3] == state[3][4]:
            return 1 if state[0][1] == self.my_piece else -1
        # TODO: check / diagonal wins

        # check main / main diagonal
        for i in range(2):
            if state[4-i][i] != ' ' and state[4-i][i] == state[3-i][i+1] == state[2-i][i+2] == state[1-i][i+3]:
                return 1 if state[4-i][i] == self.my_piece else -1

        # check the two smaller / diagonals
        if state[0][3] != ' ' and state[0][3] == state[1][2] == state[2][1] == state[3][0]:
            return 1 if state[0][3] == self.my_piece else -1

        if state[1][4] != ' ' and state[1][4] == state[2][3] == state[3][2] == state[4][1]:
            return 1 if state[1][4] == self.my_piece else -1

        # TODO: check box wins
        for i in range(len(self.board) - 1):
            for j in range(len(self.board) - 1):
                if state[i][j] != ' ' and state[i][j] == state[i][j+1] == state[i+1][j]  == state[i+1][j+1]:
                    return 1 if state[i][j] == self.my_piece else -1

        return 0 # no winner yet

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()