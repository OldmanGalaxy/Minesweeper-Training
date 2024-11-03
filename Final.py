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
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    reveal_criterion = nn.BCELoss(reduction='none')
    flag_criterion = nn.BCELoss(reduction='none')
    
    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch_states, batch_reveal_labels, batch_flag_labels in dataloader:
            optimizer.zero_grad()
            reveal_pred, flag_pred = model(batch_states)
            
            reveal_loss = reveal_criterion(reveal_pred, batch_reveal_labels)
            flag_loss = flag_criterion(flag_pred, batch_flag_labels)
            
            reveal_mask = (batch_reveal_labels > 0).float()
            flag_mask = (batch_flag_labels > 0).float()
            
            reveal_loss = (reveal_loss * reveal_mask).mean()
            flag_loss = (flag_loss * flag_mask).mean()
            
            loss = reveal_loss + flag_loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
    
    model.eval()
    return total_loss

def load_latest_model(input_size: int, save_dir: str = "checkpoints") -> Tuple[nn.Module, int, Dict]:
    if not os.path.exists(save_dir):
        return Network(input_size), 0, {"total_games": 0, "total_wins": 0}
    
    try:
        with open(os.path.join(save_dir, "latest_generation.txt"), 'r') as f:
            generation = int(f.read().strip())
        
        model = Network(input_size)
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
        return Network(input_size), 0, {"total_games": 0, "total_wins": 0}

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
    
    num_generations = 100
    games_per_generation = 1
    
    game = Game(rows, cols, num_mines)
    ui = Interface(game, initial_generation=start_generation + 1, initial_winrate=initial_winrate)
    solver = Solution(game, model)
    
    try:
        for generation in range(start_generation, start_generation + num_generations):
            games_data = []
            generation_wins = 0
            game_states = []
            game_labels = []
            
            print(f"\nGeneration {generation + 1}")
            print(f"Starting generation with: Total games: {total_games}, Total wins: {total_wins}")
            ui.update_generation(generation + 1)
            
            for _ in range(games_per_generation):
                game_completed = False
                while not game_completed:
                    state = solver.get_game_state_features()
                    position, should_flag = solver.make_move()
                    row, col = position
                    
                    if should_flag:
                        flag_placed = game.toggle_flag(row, col)
                        flag_label = np.zeros(input_size * 2)
                        flag_label[input_size + row * cols + col] = 1 if flag_placed else 0
                        game_labels.append(flag_label)
                    else:
                        if not game.flagged[row][col]:
                            success = game.reveal(row, col)
                            reveal_label = np.zeros(input_size * 2)
                            reveal_label[row * cols + col] = 1 if success else 0
                            game_labels.append(reveal_label)
                    
                    game_states.append(state)
                    ui.update_display()
                    time.sleep(0.1)
                    
                    if game.game_over:
                        game_completed = True
                        if game.won:
                            generation_wins += 1
                            total_wins += 1
                            print("Game won!")
                        else:
                            print("Game lost!")
                        
                        total_games += 1
                        current_winrate = (total_wins / total_games) * 100
                        ui.update_winrate(current_winrate)
                        
                        game.reset_game()
                        solver = Solution(game, model)
            
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
                position, should_flag = solver.make_move()
                row, col = position
                
                if should_flag:
                    game.toggle_flag(row, col)
                    game.toggle_flag(row, col)
                    game.check_win()
                else:
                    if not game.flagged[row][col]:
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

if __name__ == "__main__":
    main()