import gym
from gym import spaces
import numpy as np
from collections import deque
import time

class VanishingTicTacToeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, board_size=3, disappear_turn=None):
        super().__init__()
        self.n = board_size
        self.disappear_turn = self.n
        self.disappear_turn = self.n if disappear_turn is None else disappear_turn
        self.num_cells = self.n * self.n
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=-1, high=1,
                                shape=(self.num_cells,),
                                dtype=np.int8),
            "history_x": spaces.Box(low=-1, high=self.num_cells-1,
                                    shape=(self.disappear_turn,),
                                    dtype=np.int8),
            "history_o": spaces.Box(low=-1, high=self.num_cells-1,
                                    shape=(self.disappear_turn,),
                                    dtype=np.int8),
            "curr_player": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int8)
        })
        self.action_space = spaces.Discrete(self.num_cells)
        self.reset()

    #turns board + histories into a dict.
    def _pack_obs(self):
        history_x = list(self.move_history_x)
        history_o = list(self.move_history_o)

        history_x += [-1] * (self.disappear_turn - len(history_x))
        history_o += [-1] * (self.disappear_turn - len(history_o))

        return {
            "board":     self.board.copy(),
            "history_x": np.array(history_x, dtype=np.int8),
            "history_o": np.array(history_o, dtype=np.int8),
            "current_player": np.array([self.current_player], dtype=np.int8)
        }


    def reset(self):
        self.board = np.zeros(self.num_cells, dtype=np.int8)
        self.move_history_x = deque()
        self.move_history_o = deque()
        self.current_player = 1
        self.done = False
        return self._pack_obs()

    def step(self, action):
        info = {} #to store move order

        if self.done or self.board[action] != 0:
            info["invalid"]   = True
            print(f"Invalid action: {action} by player {self.current_player}")
            print(f"Current board:\n{self.board.reshape((self.n, self.n))}")
            exit("Invalid action taken, exiting...")

        history = (self.move_history_x if self.current_player == 1 else self.move_history_o)

        if len(history) >= self.disappear_turn:
            oldest_pos = history.popleft()
            self.board[oldest_pos] = 0

        self.board[action] = self.current_player
        history.append(action)

        winner = self.check_winner()
        reward, self.done = 0.0, False

        if winner is not None:
            reward = 1.0 * winner
            self.done = True
        else:
            reward = 0.0
            self.current_player *= -1

        info["history_x"] = list(self.move_history_x)
        info["history_o"] = list(self.move_history_o)

        return self._pack_obs(), reward, self.done, info

    def check_winner(self):
        b = self.board.reshape((self.n, self.n))
        player = self.current_player

        # looks for horizontals
        for i in range(self.n):
            if all(b[i, j] == player for j in range(self.n)):
                return player
            if all(b[j, i] == player for j in range(self.n)):
                return player

        # looks at the two diagonals
        if all(b[i, i] == player for i in range(self.n)):
            return player
        if all(b[i, self.n - 1 - i] == player for i in range(self.n)):
            return player

        return None


    # right now the rendering is in the terminal (just print()) but maybe switch to matplotlib?
    def render(self, mode="human", delay=0.0):
        symbols = {1: " X ", -1: " O ", 0: "   "}
        b = [symbols[val] for val in self.board]
        for i in range(self.n):
            row = b[i * self.n:(i + 1) * self.n]
            print("|".join(row))
            if i < self.n - 1:
                print("---+" * (self.n - 1) + "---")
        print()
        time.sleep(delay)

    def close(self):
        pass 
