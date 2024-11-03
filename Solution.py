import numpy as np
import torch
import torch.nn as nn
import random
from typing import Tuple, Set, Dict
from Game import *

class Solution:
    def __init__(self, game: Game, model: nn.Module = None):
        self.game = game
        self.model = model
        self.move_history = []
        self.success_rate = {}
    
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
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                if not self.game.revealed[i][j] and not self.game.flagged[i][j]:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < self.game.rows and 
                                0 <= nj < self.game.cols and 
                                self.game.revealed[ni][nj]):
                                border_cells.add((i, j))
                                break
        return border_cells
    
    def make_move(self) -> Tuple[Tuple[int, int], bool]:
        if self.game.first_move:
            return (self.game.rows // 2, self.game.cols // 2), False
        
        safe_moves = self.find_safe_moves_basic()
        if safe_moves:
            return random.choice(list(safe_moves)), False
            
        if self.model is not None:
            state = self.get_game_state_features()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                reveal_pred, flag_pred = self.model(state_tensor)
                
                reveal_pred = reveal_pred.numpy() if isinstance(reveal_pred, torch.Tensor) else reveal_pred
                flag_pred = flag_pred.numpy() if isinstance(flag_pred, torch.Tensor) else flag_pred
                
                reveal_matrix = reveal_pred.reshape(self.game.rows, self.game.cols)
                flag_matrix = flag_pred.reshape(self.game.rows, self.game.cols)
                
                border_cells = self.get_border_cells()
                non_border_mask = np.ones_like(reveal_matrix)
                
                for i in range(self.game.rows):
                    for j in range(self.game.cols):
                        if (i, j) not in border_cells or self.game.revealed[i][j] or self.game.flagged[i][j]:
                            reveal_matrix[i][j] = -1
                            flag_matrix[i][j] = -1
                
                max_reveal = reveal_matrix.max()
                max_flag = flag_matrix.max()
                
                if max_flag > 0.85 and self.game.flags_placed < self.game.num_mines:
                    flag_pos = np.unravel_index(flag_matrix.argmax(), flag_matrix.shape)
                    return flag_pos, True
                    
                if max_reveal > 0.15:
                    reveal_pos = np.unravel_index(reveal_matrix.argmax(), reveal_matrix.shape)
                    return reveal_pos, False
        
        border_cells = list(self.get_border_cells())
        if border_cells:
            return random.choice(border_cells), False
            
        unrevealed = [(i, j) for i in range(self.game.rows)
                     for j in range(self.game.cols)
                     if not self.game.revealed[i][j] and not self.game.flagged[i][j]]
        
        if not unrevealed:
            return (0, 0), False
            
        if len(unrevealed) == self.game.num_mines - self.game.flags_placed:
            return unrevealed[0], True
            
        return random.choice(unrevealed), False
    
    def find_safe_moves_basic(self) -> Set[Tuple[int, int]]:
        safe_moves = set()
        flagged_mines = set()
        
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                if self.game.revealed[i][j] and self.game.board[i][j] > 0:
                    adjacent_cells = self.game.get_adjacent_cells(i, j)
                    unknown_cells = [(x, y) for x, y in adjacent_cells 
                                   if not self.game.revealed[x][y] and not self.game.flagged[x][y]]
                    flagged_adjacent = sum(1 for x, y in adjacent_cells if self.game.flagged[x][y])
                    
                    if flagged_adjacent == self.game.board[i][j]:
                        safe_moves.update(unknown_cells)
                    elif len(unknown_cells) + flagged_adjacent == self.game.board[i][j]:
                        flagged_mines.update(unknown_cells)
        
        for mine in flagged_mines:
            if not self.game.flagged[mine[0]][mine[1]] and self.game.flags_placed < self.game.num_mines:
                self.game.toggle_flag(mine[0], mine[1])
        return safe_moves