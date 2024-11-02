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
        
    def get_game_state_features(self) -> np.ndarray:
        basic_state = np.zeros((self.game.rows, self.game.cols))
        adjacent_unknown = np.zeros((self.game.rows, self.game.cols))
        
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                if self.game.revealed[i][j]:
                    basic_state[i][j] = self.game.board[i][j]
                    unknown_count = 0
                    for adj_i, adj_j in self.game.get_adjacent_cells(i, j):
                        if not self.game.revealed[adj_i][adj_j]:
                            unknown_count += 1
                    adjacent_unknown[i][j] = unknown_count
                else:
                    basic_state[i][j] = -2
        
        features = np.concatenate([basic_state.flatten(), adjacent_unknown.flatten()])
        return features
    
    def find_safe_moves_basic(self) -> Set[Tuple[int, int]]:
        safe_moves = set()
        mine_locations = set()
        
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                if self.game.revealed[i][j] and self.game.board[i][j] > 0:
                    adjacent_cells = self.game.get_adjacent_cells(i, j)
                    unknown_cells = [(x, y) for x, y in adjacent_cells 
                                   if not self.game.revealed[x][y]]
                    if len(unknown_cells) == self.game.board[i][j]:
                        mine_locations.update(unknown_cells)
        
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                if self.game.revealed[i][j] and self.game.board[i][j] > 0:
                    adjacent_cells = self.game.get_adjacent_cells(i, j)
                    unknown_cells = [(x, y) for x, y in adjacent_cells 
                                   if not self.game.revealed[x][y]]
                    mine_count = sum(1 for x, y in unknown_cells if (x, y) in mine_locations)
                    
                    if mine_count == self.game.board[i][j]:
                        safe_moves.update((x, y) for x, y in unknown_cells 
                                        if (x, y) not in mine_locations)
        
        return safe_moves
    
    def make_move(self) -> Tuple[int, int]:
        if self.game.first_move:
            if random.random() < 0.8:
                return (self.game.rows // 2, self.game.cols // 2)
            else:
                corners = [(0, 0), (0, self.game.cols-1), 
                          (self.game.rows-1, 0), (self.game.rows-1, self.game.cols-1)]
                return random.choice(corners)
        
        safe_moves = self.find_safe_moves_basic()
        if safe_moves:
            return random.choice(list(safe_moves))
        
        if self.model is not None:
            state = self.get_game_state_features()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                predictions = self.model(state_tensor)
                predictions = predictions.squeeze(0).numpy()
                predictions = predictions.reshape(self.game.rows, self.game.cols)
                
                for i in range(self.game.rows):
                    for j in range(self.game.cols):
                        if self.game.revealed[i][j] or self.game.flagged[i][j]:
                            predictions[i][j] = -1
                
                max_prob = predictions.max()
                if max_prob > -1:
                    max_indices = np.where(predictions == max_prob)
                    row = max_indices[0][0]
                    col = max_indices[1][0]
                    return (row, col)
        
        unrevealed = [(i, j) for i in range(self.game.rows) 
                     for j in range(self.game.cols)
                     if not self.game.revealed[i][j] and not self.game.flagged[i][j]]
        return random.choice(unrevealed) if unrevealed else (0, 0)
