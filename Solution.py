import numpy as np 
import torch
import torch.nn as nn
import random
from typing import Tuple, Set
from Game import *

class Solution:
    def __init__(self, game: Game, model: nn.Module = None):
        self.game = game
        self.model = model
        self.move_history = []
        self.success_rate = {}
        self.temperature = 0.2
        self.early_game_threshold = 5
        self.moves_made = 0
    
    def get_game_state_features(self) -> np.ndarray:
        revealed_state = np.zeros((self.game.rows, self.game.cols))
        number_state = np.zeros((self.game.rows, self.game.cols))
        
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                revealed_state[i][j] = 1 if self.game.revealed[i][j] else 0
                if self.game.revealed[i][j]:
                    number_state[i][j] = self.game.board[i][j] / 8 if self.game.board[i][j] > 0 else 0
                elif self.game.flagged[i][j]:
                    number_state[i][j] = -1
        
        features = np.stack([revealed_state, number_state])
        return features
    
    def get_border_cells(self) -> Set[Tuple[int, int]]:
        border_cells = set()
        revealed_numbers = set()
        
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                if self.game.revealed[i][j] and self.game.board[i][j] > 0:
                    revealed_numbers.add((i, j))
                    
        for i, j in revealed_numbers:
            adjacent_cells = self.game.get_adjacent_cells(i, j)
            for ni, nj in adjacent_cells:
                if not self.game.revealed[ni][nj] and not self.game.flagged[ni][nj]:
                    border_cells.add((ni, nj))
        return border_cells

    def get_safe_opening_move(self) -> Tuple[int, int]:
        center_row = self.game.rows // 2
        center_col = self.game.cols // 2
        return (center_row, center_col)
    
    def find_safe_moves_basic(self) -> Set[Tuple[int, int]]:
        safe_moves = set()
        must_be_mines = set()
        
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                if self.game.revealed[i][j] and self.game.board[i][j] > 0:
                    adjacent_cells = self.game.get_adjacent_cells(i, j)
                    unrevealed = [(x, y) for x, y in adjacent_cells 
                                if not self.game.revealed[x][y]]
                    flagged = sum(1 for x, y in unrevealed if self.game.flagged[x][y])
                    remaining = [(x, y) for x, y in unrevealed 
                               if not self.game.flagged[x][y]]
                    
                    if flagged == self.game.board[i][j] and remaining:
                        safe_moves.update(remaining)
                    elif len(remaining) + flagged == self.game.board[i][j] and remaining:
                        must_be_mines.update(remaining)
        
        if must_be_mines:
            return safe_moves
        return safe_moves
    
    def find_definite_mines(self) -> Set[Tuple[int, int]]:
        must_be_mines = set()
        
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                if self.game.revealed[i][j] and self.game.board[i][j] > 0:
                    adjacent_cells = self.game.get_adjacent_cells(i, j)
                    unrevealed = [(x, y) for x, y in adjacent_cells 
                                if not self.game.revealed[x][y]]
                    flagged = sum(1 for x, y in unrevealed if self.game.flagged[x][y])
                    remaining = [(x, y) for x, y in unrevealed 
                               if not self.game.flagged[x][y]]
                    
                    if len(remaining) + flagged == self.game.board[i][j]:
                        must_be_mines.update(remaining)
        
        return must_be_mines
    
    def calculate_cell_safety(self, row: int, col: int) -> float:
        if self.game.revealed[row][col] or self.game.flagged[row][col]:
            return -float('inf')
            
        safety_score = 0
        adjacent_cells = self.game.get_adjacent_cells(row, col)
        revealed_neighbors = [(i, j) for i, j in adjacent_cells if self.game.revealed[i][j]]
        
        if not revealed_neighbors:
            return 0.1
            
        for i, j in revealed_neighbors:
            if self.game.board[i][j] > 0:
                unrevealed = [(x, y) for x, y in self.game.get_adjacent_cells(i, j)
                             if not self.game.revealed[x][y] and not self.game.flagged[x][y]]
                if unrevealed:
                    mine_probability = (self.game.board[i][j] - 
                                     sum(1 for x, y in self.game.get_adjacent_cells(i, j)
                                         if self.game.flagged[x][y])) / len(unrevealed)
                    safety_score += (1 - mine_probability) / len(revealed_neighbors)
            else:
                safety_score += 1 / len(revealed_neighbors)
        
        return safety_score
    
    def get_unexplored_cells(self) -> Set[Tuple[int, int]]:
        unexplored = set()
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                if not self.game.revealed[i][j] and not self.game.flagged[i][j]:
                    unexplored.add((i, j))
        return unexplored
    
    def make_move(self) -> Tuple[Tuple[int, int], bool]:
        if self.game.first_move:
            return self.get_safe_opening_move(), False
            
        safe_moves = self.find_safe_moves_basic()
        if safe_moves:
            return random.choice(list(safe_moves)), False
            
        definite_mines = self.find_definite_mines()
        if definite_mines and self.game.flags_placed < self.game.num_mines:
            return random.choice(list(definite_mines)), True
            
        border_cells = self.get_border_cells()
        if not border_cells:
            unexplored = self.get_unexplored_cells()
            if unexplored:
                return max(unexplored, key=lambda pos: self.calculate_cell_safety(*pos)), False
            return (0, 0), False
            
        if self.model is not None:
            state = self.get_game_state_features()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                reveal_pred, flag_pred = self.model(state_tensor)
                reveal_pred = reveal_pred.numpy() if isinstance(reveal_pred, torch.Tensor) else reveal_pred
                flag_pred = flag_pred.numpy() if isinstance(flag_pred, torch.Tensor) else flag_pred
                
                reveal_matrix = reveal_pred.reshape(self.game.rows, self.game.cols)
                flag_matrix = flag_pred.reshape(self.game.rows, self.game.cols)
                
                for i in range(self.game.rows):
                    for j in range(self.game.cols):
                        if self.game.revealed[i][j] or self.game.flagged[i][j]:
                            reveal_matrix[i][j] = -float('inf')
                            flag_matrix[i][j] = -float('inf')
                            
                border_matrix = np.zeros_like(reveal_matrix)
                for i, j in border_cells:
                    safety_score = self.calculate_cell_safety(i, j)
                    border_matrix[i][j] = safety_score * reveal_matrix[i][j]
                
                if np.max(border_matrix) > 0:
                    best_pos = np.unravel_index(border_matrix.argmax(), border_matrix.shape)
                    return best_pos, False
        return max(border_cells, key=lambda pos: self.calculate_cell_safety(*pos)), False