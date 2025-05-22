import numpy as np
import random
import math

# SimpleRuleBasedAgent
#
# The basic idea of this agent is to place a piece adjacent to a previous piece of ours (if X will place an X within 1 space of a previous X)
# Added in one more thing since it wasn't winning enough- if it is 1 turn from winning it will select the winning piece

class SimpleRuleBasedAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.own_moves = []

    def act(self, observation):
        board = observation["board"]
        board_len = len(board)
        board_size = int(math.sqrt(board_len))
        legal_actions = [i for i in range(board_len) if board[i] == 0]

        #determining which player we are
        count_x = np.sum(board == 1)
        count_o = np.sum(board == -1)
        player = 1 if count_x == count_o else -1

        # look for wins
        for move in legal_actions:
            temp_board = board.copy()
            temp_board[move] = player
            if self.check_win(temp_board, board_size, player=player):
                self.own_moves.append(move)
                return move

        # if no wins choose an adjacent move
        move_coords = [(i // board_size, i % board_size) for i in self.own_moves]
        adjacent_candidates = set()
        for r, c in move_coords:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board_size and 0 <= nc < board_size:
                        idx = nr * board_size + nc
                        if board[idx] == 0:
                            adjacent_candidates.add(idx)

        adjacent_legal_moves = list(adjacent_candidates)
        if adjacent_legal_moves:
            chosen = random.choice(adjacent_legal_moves)
        else:
            chosen = random.choice(legal_actions)

        self.own_moves.append(chosen)
        return chosen


    def check_win(self, board, n, player):
        b = np.array(board).reshape((n, n))

        # rows + columns
        for i in range(n):
            if np.sum(b[i, :] == player) == n:
                return True
            if np.sum(b[:, i] == player) == n:
                return True

        # diagonals
        if np.sum(np.diag(b) == player) == n:
            return True
        if np.sum(np.diag(np.fliplr(b)) == player) == n:
            return True

        return False