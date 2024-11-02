import tkinter as tk
from tkinter import ttk
from Game import *

class Interface:
    def __init__(self, game: Game, initial_generation: int = 1, initial_winrate: float = 0.0):
        self.game = game
        self.root = tk.Tk()
        self.root.title("Minesweeper AI")
        
        self.style = ttk.Style()
        self.style.configure('GameButton.TButton', padding=4, width=3, font=('Arial', 10, 'bold'))
        
        self.number_colors = {
            1: '#0000FF',  
            2: '#008000',  
            3: '#FF0000',  
            4: '#000080',  
            5: '#800000',  
            6: '#008080',  
            7: '#000000',  
            8: '#808080'   
        }
        
        self.info_frame = ttk.Frame(self.root)
        self.info_frame.grid(row=0, column=0, columnspan=game.cols, pady=10, padx=10)
        
        self.generation_label = ttk.Label(self.info_frame, 
                                        text=f"Generation: {initial_generation}",
                                        font=('Arial', 12))
        self.generation_label.pack(side=tk.LEFT, padx=10)
        
        self.winrate_label = ttk.Label(self.info_frame, 
                                      text=f"Win Rate: {initial_winrate:.1f}%",
                                      font=('Arial', 12))
        self.winrate_label.pack(side=tk.LEFT, padx=10)
        
        self.grid_frame = ttk.Frame(self.root, padding=10)
        self.grid_frame.grid(row=1, column=0, columnspan=game.cols)
        
        self.buttons = []
        for i in range(game.rows):
            row = []
            for j in range(game.cols):
                btn = tk.Button(self.grid_frame, 
                              width=2, 
                              height=1,
                              font=('Arial', 10, 'bold'),
                              relief=tk.RAISED,
                              bg='#E0E0E0')
                btn.grid(row=i, column=j, padx=1, pady=1)
                row.append(btn)
            self.buttons.append(row)
        self.update_display()
    
    def update_generation(self, generation: int):
        self.generation_label.config(text=f"Generation: {generation}")
        self.root.update()
    
    def update_winrate(self, winrate: float):
        self.winrate_label.config(text=f"Win Rate: {winrate:.1f}%")
        self.root.update()
    
    def update_display(self):
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                button = self.buttons[i][j]
                
                if self.game.revealed[i][j]:
                    value = self.game.board[i][j]
                    if value == -1:
                        button.config(text="💣", bg='#FF9999')
                    elif value > 0:
                        button.config(text=str(value),
                                    fg=self.number_colors.get(value, 'black'),
                                    bg='#FFFFFF')
                    else:
                        button.config(text=" ", bg='#FFFFFF')
                elif self.game.flagged[i][j]:
                    button.config(text="🚩", bg='#E0E0E0')
                else:
                    button.config(text="", bg='#E0E0E0')
                
                if self.game.game_over:
                    if self.game.won:
                        if not self.game.revealed[i][j]:
                            button.config(bg='#90EE90')
                    elif self.game.board[i][j] == -1 and self.game.revealed[i][j]:
                        button.config(bg='#FF0000')
        self.root.update()