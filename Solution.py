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
    
    def get_border_cells(self) -> Set[Tuple[int, int]]:
        border_cells = set()
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                if not self.game.revealed[i][j] and not self.game.flagged[i][j]:
                    for adj_i, adj_j in self.game.get_adjacent_cells(i, j):
                        if self.game.revealed[adj_i][adj_j]:
                            border_cells.add((i, j))
                            break
        return border_cells
    
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
                        for x, y in unknown_cells:
                            if not self.game.flagged[x][y] and self.game.flags_placed < self.game.num_mines:
                                self.game.toggle_flag(x, y)
        
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
    
    def calculate_mine_probabilities(self, border_cells: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        probabilities = {}
        for cell in border_cells:
            mine_prob = 0
            adjacent_revealed = []
            for adj_i, adj_j in self.game.get_adjacent_cells(*cell):
                if self.game.revealed[adj_i][adj_j]:
                    adjacent_revealed.append((adj_i, adj_j))
            
            if adjacent_revealed:
                cell_probs = []
                for adj_i, adj_j in adjacent_revealed:
                    remaining_mines = self.game.board[adj_i][adj_j]
                    unknown_neighbors = [(x, y) for x, y in self.game.get_adjacent_cells(adj_i, adj_j)
                                      if not self.game.revealed[x][y] and not self.game.flagged[x][y]]
                    if unknown_neighbors:
                        cell_probs.append(remaining_mines / len(unknown_neighbors))
                mine_prob = min(cell_probs) if cell_probs else 0
            
            probabilities[cell] = mine_prob
        return probabilities
    
    def make_move(self) -> Tuple[Tuple[int, int], bool]:
        if self.game.first_move:
            if random.random() < 0.8:
                return (self.game.rows // 2, self.game.cols // 2), False
            else:
                corners = [(0, 0), (0, self.game.cols-1), 
                          (self.game.rows-1, 0), (self.game.rows-1, self.game.cols-1)]
                return random.choice(corners), False
        
        safe_moves = self.find_safe_moves_basic()
        if safe_moves:
            return random.choice(list(safe_moves)), False
        
        if self.model is not None:
            state = self.get_game_state_features()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                reveal_pred, flag_pred = self.model(state_tensor)
                reveal_pred = reveal_pred.squeeze(0).numpy()
                flag_pred = flag_pred.squeeze(0).numpy()
                
                reveal_matrix = reveal_pred.reshape(self.game.rows, self.game.cols)
                flag_matrix = flag_pred.reshape(self.game.rows, self.game.cols)
                border_cells = self.get_border_cells()
                
                for i in range(self.game.rows):
                    for j in range(self.game.cols):
                        if (i, j) not in border_cells:
                            reveal_matrix[i][j] = -1
                            flag_matrix[i][j] = -1
                        if self.game.revealed[i][j] or self.game.flagged[i][j]:
                            reveal_matrix[i][j] = -1
                            flag_matrix[i][j] = -1
                
                max_reveal = reveal_matrix.max()
                max_flag = flag_matrix.max()
                
                if max_flag > 0.9 and self.game.flags_placed < self.game.num_mines:
                    high_confidence_flags = np.where(flag_matrix > 0.9)
                    for i, j in zip(high_confidence_flags[0], high_confidence_flags[1]):
                        if not self.game.revealed[i][j] and not self.game.flagged[i][j]:
                            self.game.toggle_flag(i, j)
                
                if max_flag > max_reveal and max_flag > 0.7 and self.game.flags_placed < self.game.num_mines:
                    max_indices = np.where(flag_matrix == max_flag)
                    row = max_indices[0][0]
                    col = max_indices[1][0]
                    return (row, col), True
                elif max_reveal > -1:
                    max_indices = np.where(reveal_matrix == max_reveal)
                    row = max_indices[0][0]
                    col = max_indices[1][0]
                    return (row, col), False
        
        border_cells = self.get_border_cells()
        if border_cells:
            probs = self.calculate_mine_probabilities(border_cells)
            safest_cell = min(probs.items(), key=lambda x: x[1])[0]
            return safest_cell, False
        
        unrevealed = [(i, j) for i in range(self.game.rows) 
                     for j in range(self.game.cols)
                     if not self.game.revealed[i][j] and not self.game.flagged[i][j]]
                     
        if len(unrevealed) == self.game.num_mines - self.game.flags_placed:
            return unrevealed[0], True
            
        center_distance = lambda pos: abs(pos[0] - self.game.rows//2) + abs(pos[1] - self.game.cols//2)
        return min(unrevealed, key=center_distance), False