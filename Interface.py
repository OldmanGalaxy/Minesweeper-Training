import tkinter as tk
from tkinter import ttk
from Game import *

class Interface:
    def __init__(self, game: Game):
        self.game = game
        self.root = tk.Tk()
        self.root.title("Minesweeper AI")
        
        self.info_frame = ttk.Frame(self.root)
        self.info_frame.grid(row=0, column=0, columnspan=game.cols, pady=5)
        
        self.generation_label = ttk.Label(self.info_frame, text="Generation: 1")
        self.generation_label.pack(side=tk.LEFT, padx=5)
        
        self.winrate_label = ttk.Label(self.info_frame, text="Win Rate: 0%")
        self.winrate_label.pack(side=tk.LEFT, padx=5)
        
        self.grid_frame = ttk.Frame(self.root)
        self.grid_frame.grid(row=1, column=0, columnspan=game.cols)
        
        self.buttons = []
        for i in range(game.rows):
            row = []
            for j in range(game.cols):
                btn = ttk.Button(self.grid_frame, width=2)
                btn.grid(row=i, column=j)
                row.append(btn)
            self.buttons.append(row)
        
        self.update_display()
    
    def update_display(self):
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                if self.game.revealed[i][j]:
                    value = self.game.board[i][j]
                    text = "💣" if value == -1 else str(value) if value > 0 else " "
                    self.buttons[i][j].config(text=text)
                elif self.game.flagged[i][j]:
                    self.buttons[i][j].config(text="🚩")
                else:
                    self.buttons[i][j].config(text="")
        self.root.update()
    
    def update_generation(self, generation: int):
        self.generation_label.config(text=f"Generation: {generation}")
        self.root.update()
    
    def update_winrate(self, winrate: float):
        self.winrate_label.config(text=f"Win Rate: {winrate:.1f}%")
        self.root.update()