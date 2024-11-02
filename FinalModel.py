import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Dict
import time
from Interface import *
from Game import *
from Solution import *
from Network import *

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
    model = Network(input_size)
    
    num_generations = 10
    games_per_generation = 5
    total_games = 0
    total_wins = 0
    
    game = Game(rows, cols, num_mines)
    ui = Interface(game)
    
    try:
        for generation in range(num_generations):
            games_data = []
            generation_wins = 0
            
            print(f"\nGeneration {generation + 1}")
            ui.update_generation(generation + 1)
            
            for game_num in range(games_per_generation):
                print(f"Playing game {game_num + 1}/{games_per_generation}")
                game.reset_game()
                solver = Solution(game, model)
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
        
        while True:
            play_more = input("\nWould you like to play another game with the trained model? (y/n): ")
            if play_more.lower() != 'y':
                break
                
            game.reset_game()
            solver = Solution(game, model)
            
            while not game.game_over:
                row, col = solver.make_move()
                game.reveal(row, col)
                ui.update_display()
                time.sleep(0.2)
            
            print("Game Won!" if game.won else "Game Lost!")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    finally:
        print("\nClosing game...")
        ui.root.destroy()

if __name__ == "__main__":
    main()