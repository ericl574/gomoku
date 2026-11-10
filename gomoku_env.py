
import numpy as np


class GomokuEnv:
    def __init__(self, size=19):
        self.size = size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1
        self.done = False
        return self.board.copy()

    def legal_actions(self):
        return np.flatnonzero(self.board == 0)

    def step(self, action):
        if self.done:
            raise ValueError("Game is already over")

        row, col = divmod(int(action), self.size)

        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError("Action outside board")

        if self.board[row, col] != 0:
            raise ValueError("Position already occupied")

        player = self.current_player
        self.board[row, col] = player

        if self._is_winner(row, col, player):
            self.done = True
            reward = 1.0
        elif not np.any(self.board == 0):
            self.done = True
            reward = 0.0
        else:
            reward = 0.0
            self.current_player = -player

        return self.board.copy(), reward, self.done

    def _is_winner(self, row, col, player):
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            count = 1

            for direction in [-1, 1]:
                r, c = row + dr * direction, col + dc * direction

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
