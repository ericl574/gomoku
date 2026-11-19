import numpy as np


# Does the network see whose turn it is (plane 2 of encode())?
# False = color-blind: plane 2 is always 0, i.e. every position is shown the
# way white saw it. While one color wins almost every game, a color plane lets
# the value head cheat ("black to move = lost") and stops it reading the board.
# The color is still implied by stone counts (equal counts = black to move).
# Set back to True once self-play black/white wins are roughly balanced; the
# input shape does not change, so every checkpoint keeps working.
COLOR_PLANE = False


def _five_windows(size=19):
    """Flat indices of every line of 5 cells: rows, columns, both diagonals."""
    windows = []
    for r in range(size):
        for c in range(size):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                end_r, end_c = r + 4 * dr, c + 4 * dc
                if 0 <= end_r < size and 0 <= end_c < size:
                    windows.append([(r + k * dr) * size + (c + k * dc) for k in range(5)])
    return np.array(windows, dtype=np.int64)  # 1,020 windows on 19x19


FIVE_WINDOWS = _five_windows()


class GomokuState:
    def __init__(self):
        self.size = 19
        self.board = np.zeros(
            (self.size, self.size),
            dtype=np.int8,
        )
        self.current_player = 1
        self.done = False
        self.winner = 0

    def legal_actions(self):
        if self.done:
            return np.array([], dtype=np.int64)

        return np.flatnonzero(self.board == 0)

    def has_immediate_win(self):
        """
        Can the player to move make five (or more) with one stone right now?
        Pure rule check, no strategy: a winning stone always completes some line
        of 5 cells that already holds 4 of the mover's stones and 1 empty cell,
        and filling such a line always makes at least five.
        """
        if self.done:
            return False
        cells = self.board.reshape(-1)[FIVE_WINDOWS]
        own = (cells == self.current_player).sum(axis=1)
        empty = (cells == 0).sum(axis=1)
        return bool(np.any((own == 4) & (empty == 1)))

    def winning_moves(self):
        """Every cell where the player to move makes five right now (sorted)."""
        if self.done:
            return np.array([], dtype=np.int64)
        cells = self.board.reshape(-1)[FIVE_WINDOWS]
        empty = cells == 0
        lines = (empty.sum(axis=1) == 1) & ((cells == self.current_player).sum(axis=1) == 4)
        return np.unique(FIVE_WINDOWS[lines][empty[lines]])

    def loses_in_two(self):
        """
        Is the player to move certainly lost: unable to make five now, while the
        opponent already has two or more different cells that make five (an open
        four, or two fours)? One stone blocks only one of those cells, so the
        opponent wins next move. Pure rule check, like has_immediate_win().
        """
        if self.done:
            return False
        cells = self.board.reshape(-1)[FIVE_WINDOWS]
        empty = cells == 0
        one_empty = empty.sum(axis=1) == 1
        if np.any(one_empty & ((cells == self.current_player).sum(axis=1) == 4)):
            return False
        threats = one_empty & ((cells == -self.current_player).sum(axis=1) == 4)
        if threats.sum() < 2:
            return False
        return np.unique(FIVE_WINDOWS[threats][empty[threats]]).size >= 2

    def step(self, action):
        if self.done:
            raise ValueError("Game is already over")

        if isinstance(action, (bool, np.bool_)) or not isinstance(
            action, (int, np.integer)
        ):
            raise ValueError("Action must be an integer")

        action = int(action)

        if not 0 <= action < self.size * self.size:
            raise ValueError("Action outside board")

        row, col = divmod(action, self.size)

        if self.board[row, col] != 0:
            raise ValueError("Position already occupied")

        player = self.current_player
        self.board[row, col] = player

        if self._is_winner(row, col, player):
            self.done = True
            self.winner = player
        elif not np.any(self.board == 0):
            self.done = True
            self.winner = 0

        self.current_player = -player

    def _is_winner(self, row, col, player):
        directions = [
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1),
        ]

        for dr, dc in directions:
            count = 1

            for direction in (-1, 1):
                r = row + dr * direction
                c = col + dc * direction

                while (
                    0 <= r < self.size
                    and 0 <= c < self.size
                    and self.board[r, c] == player
                ):
                    count += 1
                    r += dr * direction
                    c += dc * direction

            if count >= 5:
                return True

        return False

    def terminal_value(self):
        if not self.done:
            raise ValueError("Game is not over")

        return float(self.winner * self.current_player)

    def clone(self):
        other = GomokuState()
        other.board = self.board.copy()
        other.current_player = self.current_player
        other.done = self.done
        other.winner = self.winner
        return other

    def encode(self):
        own_stones = self.board == self.current_player
        opponent_stones = self.board == -self.current_player
        black_to_move = np.full(
            (self.size, self.size),
            COLOR_PLANE and self.current_player == 1,
        )

        return np.stack(
            [own_stones, opponent_stones, black_to_move],
            axis=0,
        ).astype(np.float32)