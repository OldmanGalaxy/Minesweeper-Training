import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tkinter as tk
from tkinter import ttk
import random
from typing import List, Tuple, Set, Dict
import time

class MinesweeperNet(nn.Module):
    def __init__(self, input_size: int):
        super(MinesweeperNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class MinesweeperGame:
    def __init__(self, rows: int, cols: int, num_mines: int):
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.reset_game()
        
    def reset_game(self):
        self.board = np.zeros((self.rows, self.cols), dtype = int)
        self.revealed = np.zeros((self.rows, self.cols), dtype = bool)
        self.flagged = np.zeros((self.rows, self.cols), dtype = bool)
        self.game_over = False
        self.won = False
        self.first_move = True
        
    def initialize_board(self, first_row: int, first_col: int):
        safe_cells = [(i, j) for i in range(max(0, first_row - 1), min(self.rows, first_row + 2))
                     for j in range(max(0, first_col - 1), min(self.cols, first_col + 2))]
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
        for i in range(max(0, row - 1), min(self.rows, row + 2)):
            for j in range(max(0, col - 1), min(self.cols, col + 2)):
                if self.board[i][j] == -1:
                    count += 1
        return count
    
    def get_adjacent_cells(self, row: int, col: int) -> List[Tuple[int, int]]:
        cells = []
        for i in range(max(0, row - 1), min(self.rows, row + 2)):
            for j in range(max(0, col - 1), min(self.cols, col + 2)):
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

class MinesweeperSolver:
    def __init__(self, game: MinesweeperGame, model: nn.Module = None):
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
                corners = [(0, 0), (0, self.game.cols - 1), 
                          (self.game.rows - 1, 0), (self.game.rows - 1, self.game.cols - 1)]
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

class MinesweeperUI:
    def __init__(self, game: MinesweeperGame):
        self.game = game
        self.root = tk.Tk()
        self.root.title("Minesweeper AI")
        
        self.info_frame = ttk.Frame(self.root)
        self.info_frame.grid(row = 0, column = 0, columnspan = game.cols, pady = 5)
        
        self.generation_label = ttk.Label(self.info_frame, text="Generation: 1")
        self.generation_label.pack(side = tk.LEFT, padx = 5)
        
        self.winrate_label = ttk.Label(self.info_frame, text = "Win Rate: 0%")
        self.winrate_label.pack(side = tk.LEFT, padx = 5)
        
        self.grid_frame = ttk.Frame(self.root)
        self.grid_frame.grid(row=1, column=0, columnspan=game.cols)
        
        self.buttons = []
        for i in range(game.rows):
            row = []
            for j in range(game.cols):
                btn = ttk.Button(self.grid_frame, width=2)
                btn.grid(row = i, column = j)
                row.append(btn)
            self.buttons.append(row)
        self.update_display()
    
    def update_display(self):
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                if self.game.revealed[i][j]:
                    value = self.game.board[i][j]
                    text = "💣" if value == -1 else str(value) if value > 0 else " "
                    self.buttons[i][j].config(text = text)
                elif self.game.flagged[i][j]:
                    self.buttons[i][j].config(text = "🚩")
                else:
                    self.buttons[i][j].config(text = "")
        self.root.update()
    
    def update_generation(self, generation: int):
        self.generation_label.config(text = f"Generation: {generation}")
        self.root.update()
    
    def update_winrate(self, winrate: float):
        self.winrate_label.config(text = f"Win Rate: {winrate:.1f}%")
        self.root.update()

def train_generation(model: nn.Module, games_data: List[Dict], rows: int, cols: int):
    if not games_data:
        return
    
    states = []
    labels = []
    
    for game_data in games_data:
        states.extend(game_data['states'])
        labels.extend(game_data['labels'])
    
    states_tensor = torch.FloatTensor(states)
    labels_tensor = torch.FloatTensor(labels)
    
    dataset = torch.utils.data.TensorDataset(states_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    
    for epoch in range(3):
        for batch_states, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

def main():
    rows = int(input("Enter number of rows: "))
    cols = int(input("Enter number of columns: "))
    num_mines = int(input("Enter number of mines: "))
    
    input_size = rows * cols
    model = MinesweeperNet(input_size)
    
    num_generations = 10
    games_per_generation = 5
    total_games = 0
    total_wins = 0
    
    game = MinesweeperGame(rows, cols, num_mines)
    ui = MinesweeperUI(game)
    
    try:
        for generation in range(num_generations):
            games_data = []
            generation_wins = 0
            
            print(f"\nGeneration {generation + 1}/{num_generations}")
            ui.update_generation(generation + 1)
            
            for game_num in range(games_per_generation):
                print(f"Playing game {game_num + 1}/{games_per_generation}")
                game.reset_game()
                solver = MinesweeperSolver(game, model)
                game_states = []
                game_labels = []
                
                while not game.game_over:
                    state = solver.get_game_state_features()
                    row, col = solver.make_move()
                    success = game.reveal(row, col)
                    
                    label = np.zeros(input_size)
                    label[row * cols + col] = 1 if success else 0
                    
                    game_states.append(state)
                    game_labels.append(label)
                    
                    ui.update_display()
                    time.sleep(0.1)
                
                if game.won:
                    generation_wins += 1
                    total_wins += 1
                    print("Game Won!")
                else:
                    print("Game Lost!")
                
                total_games += 1
                current_winrate = (total_wins / total_games) * 100
                ui.update_winrate(current_winrate)
                
                if game_states:
                    games_data.append({
                        'states': game_states,
                        'labels': game_labels
                    })
            
            gen_winrate = (generation_wins / games_per_generation) * 100
            print(f"\nGeneration {generation + 1} Results:")
            print(f"Games Won: {generation_wins}/{games_per_generation}")
            print(f"Win Rate: {gen_winrate:.1f}%")
            
            print("\nTraining on generation data...")
            train_generation(model, games_data, rows, cols)
            print("Training complete!")
            
            if generation < num_generations - 1:
                response = input("\nPress Enter to continue to next generation (or 'q' to quit): ")
                if response.lower() == 'q':
                    break
        
        print("\nTraining Complete!")
        print(f"Total Games: {total_games}")
        print(f"Total Wins: {total_wins}")
        print(f"Overall Win Rate: {(total_wins/total_games)*100:.1f}%")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    finally:
        print("\nClosing game...")
        ui.root.destroy()

if __name__ == "__main__":
    main()