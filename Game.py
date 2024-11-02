import numpy as np
import random
from typing import List, Tuple

class Game:
    def __init__(self, rows: int, cols: int, num_mines: int):
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.reset_game()
        
    def reset_game(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.revealed = np.zeros((self.rows, self.cols), dtype=bool)
        self.flagged = np.zeros((self.rows, self.cols), dtype=bool)
        self.game_over = False
        self.won = False
        self.first_move = True
        
    def initialize_board(self, first_row: int, first_col: int):
        safe_cells = [(i, j) for i in range(max(0, first_row-1), min(self.rows, first_row+2))
                     for j in range(max(0, first_col-1), min(self.cols, first_col+2))]
        available_cells = [(i, j) for i in range(self.rows) for j in range(self.cols)
                          if (i, j) not in safe_cells]
        
        mine_positions = random.sample(available_cells, self.num_mines)
        for x, y in mine_positions:
            self.board[x][y] = -1
        
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] != -1:
                    self.board[i][j] = self.count_adjacent_mines(i, j)
    
    def count_adjacent_mines(self, row: int, col: int) -> int:
        count = 0
        for i in range(max(0, row-1), min(self.rows, row+2)):
            for j in range(max(0, col-1), min(self.cols, col+2)):
                if self.board[i][j] == -1:
                    count += 1
        return count
    
    def get_adjacent_cells(self, row: int, col: int) -> List[Tuple[int, int]]:
        cells = []
        for i in range(max(0, row-1), min(self.rows, row+2)):
            for j in range(max(0, col-1), min(self.cols, col+2)):
                if (i, j) != (row, col):
                    cells.append((i, j))
        return cells
    
    def reveal(self, row: int, col: int) -> bool:
        if self.revealed[row][col] or self.flagged[row][col]:
            return True
        
        if self.first_move:
            self.initialize_board(row, col)
            self.first_move = False
        
        self.revealed[row][col] = True
        
        if self.board[row][col] == -1:
            self.game_over = True
            return False
        
        if self.board[row][col] == 0:
            for i, j in self.get_adjacent_cells(row, col):
                if not self.revealed[i][j]:
                    self.reveal(i, j)
        
        if np.sum(~self.revealed) == self.num_mines:
            self.won = True
            self.game_over = True
        return True