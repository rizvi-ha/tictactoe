import numpy as np
import random
import math

class ModerateRuleBasedAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        board = observation["board"]
        board_len = len(board)
        board_size = int(math.sqrt(board_len))
        legal_actions = [i for i in range(board_len) if board[i] == 0]

        # determine which player we are
        count_x = np.sum(board == 1)
        count_o = np.sum(board == -1)
        player = 1 if count_x == count_o else -1
        opponent = -player

        # 1. Can we win?
        for move in legal_actions:
            temp_board = board.copy()
            temp_board[move] = player
            if self.check_win(temp_board, board_size, player):
                return move

        # 2. Do we need to block?
        for move in legal_actions:
            temp_board = board.copy()
            temp_board[move] = opponent
            if self.check_win(temp_board, board_size, opponent):
                return move

        # 3. Otherwise, play randomly
        return random.choice(legal_actions)

    def check_win(self, board, n, player):
        b = np.array(board).reshape((n, n))

        # rows and columns
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