# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 3))
        self.height = int(kwargs.get('height', 3))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 3))
        self.players = [1, 2]  # player1 and player2

    # def set_win_con(self, **kwargs):
    #     self.p1_win_con = int(kwargs.get('p1_win_con', 1))
    #     self.p2_win_con = int(kwargs.get('p2_win_con', 2))

    def init_board(self, start_player=0):
        if (start_player == 0):
            self.p1_win_con = 1
            self.p2_win_con = 2
        else:
            self.p1_win_con = 2
            self.p2_win_con = 1
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((5, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
            # print('self.current_player', self.current_player)
            if (self.current_player == 1):
                square_state[4][:, :] = self.p1_win_con
                # square_state[5][:, :] = self.p2_win_con
            else:
                square_state[4][:, :] = self.p2_win_con
                # square_state[5][:, :] = self.p1_win_con
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def check_win(self, condition):
        p1_win_con = self.p1_win_con
        p2_win_con = self.p2_win_con
        # print('p1_win_con', p1_win_con, 'p2_win_con', p2_win_con)
        # print(condition)
        if condition == 3:
            if p1_win_con == 1 and p2_win_con == 2:
                return -1
            elif p1_win_con == 2 and p2_win_con == 1:
                return -1
            elif p1_win_con == 1 or p1_win_con == 2:
                return 1
            elif p2_win_con == 1 or p2_win_con == 2:
                return 2
        if p1_win_con == condition and p2_win_con == condition:
            return 3
        elif p1_win_con == condition:
            return 1
        elif p2_win_con == condition:
            return 2
        else:
            if len(self.availables):
                return 0
            else:
                return -1

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row + 2:
            return 0

        player1_in_row = 0
        player2_in_row = 0

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]
            # print('player', player)#1,2

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                if player == 1:
                    player1_in_row += 1
                else:
                    player2_in_row += 1
                # return self.check_win(player)

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                if player == 1:
                    player1_in_row += 1
                else:
                    player2_in_row += 1
                # return self.check_win(player)

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                if player == 1:
                    player1_in_row += 1
                else:
                    player2_in_row += 1
                # return self.check_win(player)

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                if player == 1:
                    player1_in_row += 1
                else:
                    player2_in_row += 1
                # return self.check_win(player)

        # print('player1_in_row', player1_in_row, 'player2_in_row', player2_in_row)
        if (player1_in_row >= 1 and player2_in_row >= 1):
            return self.check_win(3)
        elif (player1_in_row >= 1):
            return self.check_win(1)
        elif (player2_in_row >= 1):
            return self.check_win(2)
        elif len(self.availables) == 0:
            return self.check_win(-1)

        return self.check_win(0)

    def game_end(self):
        """Check whether the game is ended or not"""
        winner = self.has_a_winner()
        if winner == 1 or winner == 2:
            return True, winner
        elif winner == -1 or winner == 3:
            return True, -1
        else:
            return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height
        fisrt_move_player = 'O'
        second_move_player = 'X'
        # if self.start_player == 0:
        print("Player", player1, "with O".rjust(3), 'win_con', self.board.p1_win_con)
        print("Player", player2, "with X".rjust(3), 'win_con', self.board.p2_win_con)
        # else:
        #     print("Player", player1, "with X".rjust(3), 'win_con', self.board.p1_win_con)
        #     print("Player", player2, "with O".rjust(3), 'win_con', self.board.p2_win_con)
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    # if self.start_player == 0:
                        print('O'.center(8), end='')
                    # else:
                    #     print('X'.center(8), end='')
                elif p == player2:
                    # if self.start_player == 0:
                        print('X'.center(8), end='')
                    # else:
                    #     print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1, **kwargs):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.start_player = start_player
        self.board.init_board(start_player)
        self.board.p1_win_con = int(kwargs.get('p1_win_con', 1))
        self.board.p2_win_con = int(kwargs.get('p2_win_con', 2))
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            print('start_player: ', start_player)
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner], 'Fisrt player is', players[start_player+1])
                        # print(start_player, "Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie", 'Fisrt player is', players[start_player+1])
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3, **kwargs):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        self.board.p1_win_con = int(kwargs.get('p1_win_con', 1))
        self.board.p2_win_con = int(kwargs.get('p2_win_con', 2))
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
