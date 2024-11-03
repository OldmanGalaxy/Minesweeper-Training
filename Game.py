import numpy as np
import random
from typing import List, Tuple

class Game:
    def __init__(self, rows: int, cols: int, num_mines: int):
        self.rows = rows
        self.cols = cols
        self.num_mines = min(num_mines, rows * cols - 9)
        self.reset_game()
        
    def reset_game(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.revealed = np.zeros((self.rows, self.cols), dtype=bool)
        self.flagged = np.zeros((self.rows, self.cols), dtype=bool)
        self.game_over = False
        self.won = False
        self.first_move = True
        self.flags_placed = 0

    def initialize_board(self, first_row: int, first_col: int):
        safe_cells = {(i, j) for i in range(max(0, first_row-1), min(self.rows, first_row+2))
                     for j in range(max(0, first_col-1), min(self.cols, first_col+2))}
        
        all_cells = {(i, j) for i in range(self.rows) for j in range(self.cols)} - safe_cells
        available_cells = list(all_cells)
        mine_positions = []
        remaining_cells = list(available_cells)
        
        while len(mine_positions) < self.num_mines and remaining_cells:
            candidate = random.choice(remaining_cells)
            remaining_cells.remove(candidate)
            
            self.board[candidate[0]][candidate[1]] = -1
            risky_pattern = False
            
            for i in range(max(0, candidate[0]-2), min(self.rows, candidate[0]+3)):
                for j in range(max(0, candidate[1]-2), min(self.cols, candidate[1]+3)):
                    if (i,j) not in safe_cells:
                        adjacent_count = self.count_adjacent_mines(i, j)
                        neighbors = self.get_adjacent_cells(i, j)
                        unassigned = sum(1 for x, y in neighbors if (x,y) not in safe_cells and self.board[x][y] != -1)
                        if unassigned > 0 and adjacent_count / unassigned >= 0.5:
                            risky_pattern = True
                            break
                if risky_pattern:
                    break
            
            if risky_pattern:
                self.board[candidate[0]][candidate[1]] = 0
                continue
                
            mine_positions.append(candidate)
        
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
    
    def toggle_flag(self, row: int, col: int) -> bool:
        if self.revealed[row][col]:
            return False
            
        if not self.flagged[row][col] and self.flags_placed < self.num_mines:
            self.flagged[row][col] = True
            self.flags_placed += 1
            return True
        elif self.flagged[row][col]:
            self.flagged[row][col] = False
            self.flags_placed -= 1
            return False
        return False
    
    def check_win(self):
        if self.flags_placed != self.num_mines:
            return False
            
        for i in range(self.rows):
            for j in range(self.cols):
                if self.flagged[i][j] and self.board[i][j] != -1:
                    return False
                if not self.flagged[i][j] and self.board[i][j] == -1:
                    return False
                if not self.revealed[i][j] and self.board[i][j] != -1 and not self.flagged[i][j]:
                    return False
                    
        self.won = True
        self.game_over = True
        self.reset_game()
        return True
    
    def reveal(self, row: int, col: int) -> bool:
        if self.revealed[row][col] or self.flagged[row][col]:
            return True
        
        if self.first_move:
            self.initialize_board(row, col)
            self.first_move = False
        
        self.revealed[row][col] = True
        
        if self.board[row][col] == -1:
            self.game_over = True
            self.reset_game()
            return False
        
        if self.board[row][col] == 0:
            for i, j in self.get_adjacent_cells(row, col):
                if not self.revealed[i][j]:
                    self.reveal(i, j)
        
        if np.sum(~self.revealed) == self.num_mines:
            self.won = True
            self.game_over = True
            self.reset_game()
        return True