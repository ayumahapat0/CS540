import numpy as np
import random
import math
import copy


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

    def succ(self, state, piece):
        l = list()
        drop_phase = True
        c = 0
        for i in state:
            for j in i:
                if j != " ":
                    c += 1
        if c == 8:
            drop_phase = False
        if not drop_phase:
            for i in range(5):
                for j in range(5):
                    if state[i][j] == piece:
                        if j+1 < 5 and state[i][j+1] == " ":
                            dc = copy.deepcopy(state)
                            dc[i][j+1] = piece
                            dc[i][j] = " "
                            l.append(dc)
                            # adding = ((i, j), (i, j+1))
                        if j - 1 >= 0 and state[i][j-1] == " ":
                            dc = copy.deepcopy(state)
                            dc[i][j-1] = piece
                            dc[i][j] = " "
                            l.append(dc)
                        if i+1 < 5 and state[i+1][j] == " ":
                            dc = copy.deepcopy(state)
                            dc[i+1][j] = piece
                            dc[i][j] = " "
                            l.append(dc)
                        if i - 1 >= 0 and state[i-1][j] == " ":
                            dc = copy.deepcopy(state)
                            dc[i-1][j] = piece
                            dc[i][j] = " "
                            l.append(dc)
                        if i - 1 >= 0 and j - 1 >= 0 and state[i-1][j-1] == " ":
                            dc = copy.deepcopy(state)
                            dc[i-1][j-1] = piece
                            dc[i][j] = " "
                            l.append(dc)
                        if i - 1 >= 0 and j + 1 < 5 and state[i-1][j+1] == " ":
                            dc = copy.deepcopy(state)
                            dc[i-1][j+1] = piece
                            dc[i][j] = " "
                            l.append(dc)
                        if i + 1 < 5 and j + 1 < 5 and state[i+1][j+1] == " ":
                            dc = copy.deepcopy(state)
                            dc[i+1][j+1] = piece
                            dc[i][j] = " "
                            l.append(dc)
                        if i + 1 < 5 and j - 1 >= 0 and state[i+1][j-1] == " ":
                            dc = copy.deepcopy(state)
                            dc[i+1][j-1] = piece
                            dc[i][j] = " "
                            l.append(dc)
        else:
            for i in range(5):
                for j in range(5):
                    if state[i][j] == " ":
                        dc = copy.deepcopy(state)
                        dc[i][j] = piece
                        l.append(dc)
        return l

    def printb(self, succ):
        for i in succ:
            for j in i:
                print(j)
            print()
        pass

    def make_move(self, state):
        # self.printb(state)
        # print(self.heuristic_game_value(state))
        # TeekoPlayer.printb(self, TeekoPlayer.succ(self, state))
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
        # TODO: implement a minimax algorithm to play better
        # c = 0
        # for i in state:
        #     for j in i:
        #         if j != " ":
        #             c += 1
        # if c == 8:
        #     drop_phase = False
        # if not drop_phase:

        move = []
        succ = self.succ(state, self.my_piece)
        best = 10000
        for i in succ:
            if self.heuristic_game_value(i) == 1:
                move = self.find(state, i, self.my_piece)
                break
            eval = self.minimax(i, 0, False)
            if(eval == 1):
                move = self.find(state, i, self.my_piece)
                break
            if eval <= best:
                move = self.find(state, i, self.my_piece)
                best = eval
        return move

    def find(self, state, nstate, piece):
        c = 0
        for i in state:
            for j in i:
                if j != " ":
                    c += 1
        c2 = 0
        for i in nstate:
            for j in i:
                if j != " ":
                    c2 += 1
        if c != c2:
            for i in range(5):
                for j in range(5):
                    if state[i][j] != nstate[i][j]:
                        move = []
                        move.append((i, j))
                        return move
        else:
            for i in range(5):
                for j in range(5):
                    if state[i][j] == " " and nstate[i][j] != " ":
                        if j-1 >= 0 and state[i][j-1] == nstate[i][j] and nstate[i][j-1] == " ":
                            move = []
                            move.append((i,j))
                            move.append((i, j-1))
                            return move
                        if j+1 < 5 and state[i][j+1] == nstate[i][j] and nstate[i][j+1] == " ":
                            move = []
                            move.append((i,j))
                            move.append((i, j+1))
                            return move
                        if i-1 >= 0 and state[i-1][j] == nstate[i][j] and nstate[i-1][j] == " ":
                            move = []
                            move.append((i,j))
                            move.append((i-1, j))
                            return move
                        if i + 1 < 5 and state[i+1][j] == nstate[i][j] and nstate[i+1][j] == " ":
                            move = []
                            move.append((i,j))
                            move.append((i+1, j))
                            return move
                        if i + 1 < 5 and j + 1 < 5 and state[i+1][j+1] == nstate[i][j] and nstate[i+1][j+1] == " ":
                            move = []
                            move.append((i,j))
                            move.append((i+1, j+1))
                            return move
                        if i + 1 < 5 and j - 1 >= 0 and state[i+1][j-1] == nstate[i][j] and nstate[i+1][j-1] == " ":
                            move = []
                            move.append((i,j))
                            move.append((i+1, j-1))
                            return move
                        if j + 1 < 5 and i - 1 >= 0 and state[i-1][j+1] == nstate[i][j] and nstate[i-1][j+1] == " ":
                            move = []
                            move.append((i,j))
                            move.append((i-1, j+1))
                            return move
                        if i - 1 >= 0 and j - 1 >= 0 and state[i-1][j-1] == nstate[i][j] and nstate[i-1][j-1] == " ":
                            move = []
                            move.append((i,j))
                            move.append((i-1, j-1))
                            return move

    def print_board2(self, state):
        for row in range(len(state)):
            line = str(row) + ": "
            for cell in state[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def heuristic_game_value(self, state):
        if self.game_value(state) == 1 or self.game_value(state) == -1:
            return self.game_value(state)
        tc = 0
        for i in range(5):
            for j in range(5):
                if state[i][j] != " ":
                    c = 0
                    for k in range(5):
                        for l in range(5):
                            if state[k][l] != " ":
                                c += max(abs(i-k), abs(j-l))
                    tc += c
        if tc == 0:
            return 0
        if tc/2-22 > 0:
            return (tc/2-22)/70
        return ((tc/2)-22)/22

    def minimax(self, state, depth, aiTurn):
        if depth >= 2 or abs(self.heuristic_game_value(state)) == 1:
            return(self.heuristic_game_value(state))
        if aiTurn:
            maxy = -math.inf
            succ = self.succ(state, self.my_piece)
            for i in succ:
                maxy = max(maxy, self.minimax(i, depth+1, False))
            return maxy
        else:
            mini = math.inf
            piece = "b"
            if(self.my_piece != "r"):
                piece = "r"
            succ = self.succ(state, piece)
            for i in succ:
                mini = min(mini, self.minimax(i, depth+1, True))
            return mini

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
            line = str(row) + ": "
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
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i + 1] == row[i + 2] == row[i + 3]:
                    return 1 if row[i] == self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i + 1][col] == state[i + 2][col] == state[i + 3][col]:
                    return 1 if state[i][col] == self.my_piece else -1


        for i in range(2):
            for j in range(2):
                if state[i][j] == state[i+1][j+1] == state[i+2][j+2] == state[i+3][j+3] and state[i][j] != " ":
                    return 1 if state[i][j] == self.my_piece else -1

        for i in range(2):
            for j in range(4,2,-1):
                if state[i][j] == state[i+1][j-1] == state[i+2][j-2] == state[i+3][j-3] and state[i][j] != " ":
                    return 1 if state[i][j] == self.my_piece else -1

        for i in range(4):
            for j in range(4):
                if state[i][j] == state[i+1][j+1] == state[i+1][j] == state[i][j+1] and state[i][j] != " ":
                    return 1 if state[i][j] == self.my_piece else -1
        return 0  # no winner yet


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
            print(ai.my_piece + " moved at " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0]) - ord("A"))])
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
            print(ai.my_piece + " moved from " + chr(move[1][1] + ord("A")) + str(move[1][0]))
            print("  to " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0]) - ord("A")),
                                      (int(move_from[1]), ord(move_from[0]) - ord("A"))])
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
