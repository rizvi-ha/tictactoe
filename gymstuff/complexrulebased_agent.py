import numpy as np
from collections import deque

class ComplexRuleBasedAgent:

    def __init__(self, action_space):
        self.action_space   = action_space
        self.marker         = None     # +1 (X) or -1 (O)
        self.n              = None     # board dimension
        self.k              = None     # disappear_turn
        self.win_lines      = None
        self.corners        = None
        self.center_cells   = None
        np.random.seed()

    def act(self, obs):
        if self.n is None:
            self._lazy_init(obs)

        board     = obs["board"].copy()
        empty_idx = [i for i, v in enumerate(board) if v == 0]

        # decide side (+1 or -1) once
        if self.marker is None:
            self.marker = 1 if (board == 1).sum() == (board == -1).sum() else -1
        me, opp = self.marker, -self.marker

        hist_me  = self._clean_hist(obs["history_x"] if me == 1 else
                                    obs["history_o"])
        hist_opp = self._clean_hist(obs["history_o"] if me == 1 else
                                    obs["history_x"])

        win_now     = lambda pos, p=me: self._is_win_after(board, hist_me, pos, p)
        opp_win_now = lambda pos:      self._is_win_after(board, hist_opp, pos, opp)
        fork_me     = lambda pos:      self._creates_fork(board, hist_me, pos, me)
        fork_opp    = lambda pos:      self._creates_fork(board, hist_opp, pos, opp)
        safe        = lambda pos:      self._is_safe_move(board, hist_me, hist_opp,
                                                          pos, me, opp)

        # 1. immediate win
        for pos in empty_idx:
            if win_now(pos):
                return pos

        # 2. block opponent win
        blocks = [pos for pos in empty_idx if opp_win_now(pos)]
        if blocks:
            # prefer a *safe* block, otherwise first block
            for pos in blocks:
                if safe(pos):
                    return pos
            return blocks[0]

        # 3. create fork (must be safe)
        for pos in empty_idx:
            if fork_me(pos) and safe(pos):
                return pos

        # 4. block opponent fork
        forks_to_block = [pos for pos in empty_idx if fork_opp(pos)]
        if forks_to_block:
            for pos in forks_to_block:
                if safe(pos):
                    return pos
            return forks_to_block[0]

        # 5. positional fallback (centre → corner → edge) but keep it safe
        fallback = (
            [c for c in self.center_cells if c in empty_idx] +
            [c for c in self.corners      if c in empty_idx] +
            empty_idx
        )
        for pos in fallback:
            if safe(pos):
                return pos
        # (extremely rare: nothing is safe) → random legal move
        return np.random.choice(empty_idx)

    @staticmethod
    def _clean_hist(vec):
        return deque(int(x) for x in vec if x != -1)

    def _lazy_init(self, obs):
        board_len  = len(obs["board"])
        self.n     = int(round(board_len ** 0.5))
        self.k     = obs["history_x"].shape[0]

        n = self.n
        self.win_lines = (
            [tuple(r * n + c for c in range(n)) for r in range(n)] +        # rows
            [tuple(r * n + c for r in range(n)) for c in range(n)] +        # cols
            [tuple(i * n + i for i in range(n)),                            # diag left to right
             tuple(i * n + (n - 1 - i) for i in range(n))]                  # diag right to left
        )
        self.corners = (0, n - 1, n * (n - 1), n * n - 1)
        if n % 2:
            self.center_cells = ((n // 2) * n + (n // 2),)
        else:
            tl = (n // 2 - 1) * n + (n // 2 - 1)
            self.center_cells = (tl, tl + 1, tl + n, tl + n + 1)


    def _simulate(self, board, hist, pos, player):
        b2 = board.copy()
        h2 = deque(hist)
        if len(h2) >= self.k:
            b2[h2.popleft()] = 0
        b2[pos] = player
        h2.append(pos)
        return b2, h2

    def _check_win(self, board, player):
        return any(all(board[i] == player for i in line) for line in self.win_lines)

    def _is_win_after(self, board, hist, pos, player):
        b2, _ = self._simulate(board, hist, pos, player)
        return self._check_win(b2, player)

    def _count_wins(self, board, hist, player):
        return sum(self._is_win_after(board, hist, p, player)
                   for p, v in enumerate(board) if v == 0)

    def _creates_fork(self, board, hist, pos, player):
        b2, h2 = self._simulate(board, hist, pos, player)
        return self._count_wins(b2, h2, player) >= 2

    def _opponent_can_win_next(self, board, hist_opp, opponent):
        for p, v in enumerate(board):
            if v == 0 and self._is_win_after(board, hist_opp, p, opponent):
                return True
        return False

    def _is_safe_move(self, board, hist_me, hist_opp, pos, me, opp):
        b2, h2me = self._simulate(board, hist_me, pos, me)
        return not self._opponent_can_win_next(b2, hist_opp, opp)
