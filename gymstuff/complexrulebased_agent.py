import numpy as np
from collections import deque

class ComplexRuleBasedAgent:
    """
    One-ply rule-based policy for *Vanishing* Tic-Tac-Toe on an n×n board.
    Decision order (highest → lowest):
        1. Win immediately.
        2. Block an opponent win.
        3. Create a fork (two winning threats next turn).
        4. Block an opponent fork.
        5. Positional fallback: centre → corner → random.
    The agent infers its own marker (+1 for X, -1 for O) the first time
    `act` is called, so no extra constructor arguments are needed.
    """

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def __init__(self, action_space):
        self.action_space = action_space
        # will be filled on first call
        self.marker        = None   # +1 or -1
        self.n             = None   # board dimension
        self.k             = None   # disappear_turn ( = n by default )
        self.win_lines     = None   # list[tuple[int]]
        self.corners       = None
        self.center_cells  = None
        np.random.seed()            # independent RNG per agent

    def act(self, obs):
        """
        Choose an *integer* action (cell index) given the current observation.
        `obs["board"]` is a flat vector of length n² with values {-1,0,+1}.
        """
        if self.n is None:                       # first ever call → init once
            self._lazy_init(obs)

        board     = obs["board"].copy()
        empty_idx = [i for i, v in enumerate(board) if v == 0]

        # --- decide which side we are playing (only once) ----------------
        if self.marker is None:
            n_x = np.count_nonzero(board == 1)
            n_o = np.count_nonzero(board == -1)
            # if counts are equal → it's X's turn; otherwise O's.
            self.marker = 1 if n_x == n_o else -1
        player   = self.marker
        opponent = -player

        hist_p   = self._clean_hist(obs["history_x"] if player == 1
                                    else obs["history_o"])
        hist_o   = self._clean_hist(obs["history_o"] if player == 1
                                    else obs["history_x"])

        # 1. win now?
        for pos in empty_idx:
            if self._is_win_after(board, hist_p, pos, player):
                return pos

        # 2. block opponent win?
        for pos in empty_idx:
            if self._is_win_after(board, hist_o, pos, opponent):
                return pos

        # 3. create a fork?
        for pos in empty_idx:
            if self._creates_fork(board, hist_p, pos, player):
                return pos

        # 4. block opponent fork?
        for pos in empty_idx:
            if self._creates_fork(board, hist_o, pos, opponent):
                return pos

        # 5. positional fallback ------------------------------------------
        # 5a. any available single centre (odd n) or one of the four centres (even n)?
        centres_open = [c for c in self.center_cells if c in empty_idx]
        if centres_open:
            return np.random.choice(centres_open)

        # 5b. a corner?
        corners_open = [c for c in self.corners if c in empty_idx]
        if corners_open:
            return np.random.choice(corners_open)

        # 5c. anything left
        return np.random.choice(empty_idx)

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_hist(hist_vec):
        """drop sentinel -1s and return a deque (oldest → newest)."""
        return deque(int(x) for x in hist_vec if x != -1)

    # ---- one-off lazy initialisation (depends on board size) ------------
    def _lazy_init(self, obs):
        board_len  = len(obs["board"])
        self.n     = int(round(board_len ** 0.5))
        self.k     = obs["history_x"].shape[0]   # == disappear_turn

        # pre-compute all winning line index tuples
        self.win_lines = []
        n = self.n
        for r in range(n):
            self.win_lines.append(tuple(r * n + c for c in range(n)))  # rows
        for c in range(n):
            self.win_lines.append(tuple(r * n + c for r in range(n)))  # cols
        self.win_lines.append(tuple(i * n + i for i in range(n)))      # diag ↘
        self.win_lines.append(tuple(i * n + (n - 1 - i) for i in range(n)))  # ↙

        # corners and centres for fallback stage
        self.corners = (0, n - 1, (n - 1) * n, n * n - 1)
        if n % 2:                         # odd → one exact centre
            c = (n // 2) * n + (n // 2)
            self.center_cells = (c,)
        else:                             # even → four-cell centre block
            tl = (n // 2 - 1) * n + (n // 2 - 1)
            self.center_cells = (tl, tl + 1, tl + n, tl + n + 1)

    # ---- game-logic primitives -----------------------------------------
    def _simulate(self, board, hist, pos, player):
        """
        Return (board′, hist′) after *player* places at *pos* and, if they
        already have k pieces, their oldest disappears.
        """
        b2 = board.copy()
        h2 = deque(hist)          # shallow copy
        if len(h2) >= self.k:     # oldest vanishes
            old = h2.popleft()
            b2[old] = 0
        b2[pos] = player
        h2.append(pos)
        return b2, h2

    def _check_win(self, board, player):
        for line in self.win_lines:
            if all(board[i] == player for i in line):
                return True
        return False

    def _is_win_after(self, board, hist, pos, player):
        b2, _ = self._simulate(board, hist, pos, player)
        return self._check_win(b2, player)

    def _count_immediate_wins(self, board, hist, player):
        wins = 0
        for pos in (i for i, v in enumerate(board) if v == 0):
            if self._is_win_after(board, hist, pos, player):
                wins += 1
        return wins

    def _creates_fork(self, board, hist, pos, player, thresh=2):
        """
        Fork = move that leaves ≥ `thresh` distinct immediate winning moves.
        """
        b2, h2 = self._simulate(board, hist, pos, player)
        return self._count_immediate_wins(b2, h2, player) >= thresh
