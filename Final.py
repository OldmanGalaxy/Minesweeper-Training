import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
import time
import os
import json
from Network import *
from Game import *
from Interface import *
from Solution import *

def train_generation(model: nn.Module, games_data: List[Dict], rows: int, cols: int):
    if not games_data:
        return
    
    states = []
    reveal_labels = []
    flag_labels = []
    
    for game_data in games_data:
        states.extend([np.array(state, dtype=np.float32) for state in game_data['states']])
        for label in game_data['labels']:
            input_size = rows * cols
            reveal_labels.append(np.array(label[:input_size], dtype=np.float32))
            flag_labels.append(np.array(label[input_size:], dtype=np.float32))
    
    states_tensor = torch.from_numpy(np.stack(states))
    reveal_labels_tensor = torch.from_numpy(np.stack(reveal_labels))
    flag_labels_tensor = torch.from_numpy(np.stack(flag_labels))
    
    dataset = torch.utils.data.TensorDataset(
        states_tensor, 
        reveal_labels_tensor,
        flag_labels_tensor
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(3):
        for batch_states, batch_reveal_labels, batch_flag_labels in dataloader:
            optimizer.zero_grad()
            
            reveal_pred, flag_pred = model(batch_states)
            reveal_loss = criterion(reveal_pred, batch_reveal_labels)
            flag_loss = criterion(flag_pred, batch_flag_labels)
            
            total_loss = reveal_loss + flag_loss
            total_loss.backward()
            optimizer.step()
    model.eval()

def load_latest_model(input_size: int, save_dir: str = "checkpoints") -> Tuple[nn.Module, int, Dict]:
    if not os.path.exists(save_dir):
        return MinesweeperNet(input_size), 0, {"total_games": 0, "total_wins": 0}
    
    try:
        with open(os.path.join(save_dir, "latest_generation.txt"), 'r') as f:
            generation = int(f.read().strip())
        
        model = MinesweeperNet(input_size)
        model_path = os.path.join(save_dir, f"model_gen_{generation}.pt")
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        
        stats_path = os.path.join(save_dir, f"stats_gen_{generation}.json")
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        print(f"Loaded model from generation {generation}")
        print(f"Loaded statistics: Total games: {stats['total_games']}, Total wins: {stats['total_wins']}")
        return model, generation, stats
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading model: {e}")
        return MinesweeperNet(input_size), 0, {"total_games": 0, "total_wins": 0}

def save_model_checkpoint(model: nn.Module, generation: int, stats: Dict, save_dir: str = "checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, f"model_gen_{generation}.pt")
    torch.save(model.state_dict(), model_path)
    
    stats_path = os.path.join(save_dir, f"stats_gen_{generation}.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    with open(os.path.join(save_dir, "latest_generation.txt"), 'w') as f:
        f.write(str(generation))
    
    print(f"Saved checkpoint for generation {generation}")
    print(f"Current statistics: Total games: {stats['total_games']}, Total wins: {stats['total_wins']}")

def main():
    rows = int(input("Enter number of rows: "))
    cols = int(input("Enter number of columns: "))
    num_mines = int(input("Enter number of mines: "))
    
    input_size = rows * cols
    model, start_generation, stats = load_latest_model(input_size)
    
    total_games = int(stats.get("total_games", 0))
    total_wins = int(stats.get("total_wins", 0))
    
    initial_winrate = (total_wins / total_games * 100) if total_games > 0 else 0.0
    print(f"Starting with: Total games: {total_games}, Total wins: {total_wins}, Win rate: {initial_winrate:.1f}%")
    
    num_generations = 10
    games_per_generation = 5
    
    game = MinesweeperGame(rows, cols, num_mines)
    ui = MinesweeperUI(game, initial_generation=start_generation + 1, initial_winrate=initial_winrate)
    
    try:
        for generation in range(start_generation, start_generation + num_generations):
            games_data = []
            generation_wins = 0
            
            print(f"\nGeneration {generation + 1}")
            print(f"Starting generation with: Total games: {total_games}, Total wins: {total_wins}")
            ui.update_generation(generation + 1)
            
            for game_num in range(games_per_generation):
                print(f"Playing game {game_num + 1}/{games_per_generation}")
                game.reset_game()
                solver = MinesweeperSolver(game, model)
                game_states = []
                game_labels = []
                
                while not game.game_over:
                    state = solver.get_game_state_features()
                    position, should_flag = solver.make_move()
                    row, col = position
                    
                    if should_flag:
                        game.flagged[row][col] = True
                        flag_label = np.zeros(input_size * 2)
                        flag_label[input_size + row * cols + col] = 1
                        game_labels.append(flag_label)
                    else:
                        success = game.reveal(row, col)
                        reveal_label = np.zeros(input_size * 2)
                        reveal_label[row * cols + col] = 1 if success else 0
                        game_labels.append(reveal_label)
                    
                    game_states.append(state)
                    ui.update_display()
                    time.sleep(0.1)
                
                if game.won:
                    generation_wins += 1
                    total_wins += 1
                    print("Game won!")
                else:
                    print("Game lost!")
                
                total_games += 1
                current_winrate = (total_wins / total_games) * 100
                ui.update_winrate(current_winrate)
                
                if game_states:
                    games_data.append({
                        'states': game_states,
                        'labels': game_labels
                    })
            
            gen_winrate = (generation_wins / games_per_generation) * 100
            overall_winrate = (total_wins / total_games) * 100
            
            print(f"\nGeneration {generation + 1} Results:")
            print(f"Games Won: {generation_wins}/{games_per_generation}")
            print(f"Generation Win Rate: {gen_winrate:.1f}%")
            print(f"Overall Statistics:")
            print(f"Total Games: {total_games}")
            print(f"Total Wins: {total_wins}")
            print(f"Overall Win Rate: {overall_winrate:.1f}%")
            
            print("\nTraining on generation data...")
            train_generation(model, games_data, rows, cols)
            print("Training complete!")
            
            stats = {
                "total_games": total_games,
                "total_wins": total_wins,
                "generation_winrate": gen_winrate,
                "overall_winrate": overall_winrate
            }
            save_model_checkpoint(model, generation + 1, stats)
            
            if generation < start_generation + num_generations - 1:
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
            solver = MinesweeperSolver(game, model)
            
            while not game.game_over:
                row, col = solver.make_move()
                game.reveal(row, col)
                ui.update_display()
                time.sleep(0.2)
            
            print("Game Won!" if game.won else "Game Lost!")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        stats = {
            "total_games": total_games,
            "total_wins": total_wins,
            "overall_winrate": (total_wins/total_games)*100
        }
        save_model_checkpoint(model, generation + 1, stats)
    
    finally:
        print("\nClosing game...")
        ui.root.destroy()
    
    while not game.game_over:
        position, should_flag = solver.make_move()
        row, col = position
        
        if should_flag:
            game.flagged[row][col] = True
        else:
            success = game.reveal(row, col)
        
        ui.update_display()
        time.sleep(0.2)

if __name__ == "__main__":
    main()