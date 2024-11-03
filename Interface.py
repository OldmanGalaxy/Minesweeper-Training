import tkinter as tk
from tkinter import ttk
from Game import *

class Interface:
    def __init__(self, game: Game, initial_generation: int = 1, initial_winrate: float = 0.0):
        self.game = game
        self.root = tk.Tk()
        self.root.title("Minesweeper AI")
        self.root.configure(bg='#2C3E50')
        
        self.number_colors = {
            1: '#3498DB',
            2: '#2ECC71',
            3: '#E74C3C',
            4: '#9B59B6',
            5: '#D35400',
            6: '#16A085',
            7: '#8E44AD',
            8: '#2C3E50'
        }
        
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.grid(row=0, column=0, sticky='nsew')
        
        self.info_frame = ttk.Frame(main_frame)
        self.info_frame.grid(row=0, column=0, pady=(0, 20), sticky='ew')
        self.info_frame.grid_columnconfigure(0, weight=1)
        self.info_frame.grid_columnconfigure(1, weight=1)
        
        style = ttk.Style()
        style.configure('Stats.TLabel', font=('Helvetica', 12, 'bold'), padding=10)
        
        self.generation_label = ttk.Label(self.info_frame, 
                                        text=f"Generation: {initial_generation}",
                                        style='Stats.TLabel')
        self.generation_label.grid(row=0, column=0, sticky='w')
        
        self.winrate_label = ttk.Label(self.info_frame, 
                                      text=f"Win Rate: {initial_winrate:.1f}%",
                                      style='Stats.TLabel')
        self.winrate_label.grid(row=0, column=1, sticky='e')
        
        self.grid_frame = ttk.Frame(main_frame)
        self.grid_frame.grid(row=1, column=0)
        
        cell_size = min(40, 800 // max(game.rows, game.cols))
        
        self.buttons = []
        for i in range(game.rows):
            row = []
            for j in range(game.cols):
                frame = ttk.Frame(self.grid_frame, padding=1)
                frame.grid(row=i, column=j)
                
                btn = tk.Button(frame, 
                              width=2,
                              height=1,
                              font=('Helvetica', cell_size // 3, 'bold'),
                              relief=tk.RAISED,
                              bg='#34495E',
                              fg='white',
                              activebackground='#2C3E50',
                              borderwidth=2,
                              highlightthickness=1,
                              highlightbackground='#2C3E50',
                              highlightcolor='#2C3E50')
                btn.pack(expand=True, fill='both')
                row.append(btn)
            self.buttons.append(row)
            
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f'+{x}+{y}')
        
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
                        button.config(text="💣", bg='#E74C3C', fg='white')
                    elif value > 0:
                        button.config(text=str(value),
                                    fg=self.number_colors.get(value, 'white'),
                                    bg='#ECF0F1')
                    else:
                        button.config(text=" ", bg='#ECF0F1')
                elif self.game.flagged[i][j]:
                    button.config(text="🚩", 
                                fg='#E74C3C',
                                bg='#34495E')
                else:
                    button.config(text="",
                                bg='#34495E')
                
                if self.game.game_over:
                    if self.game.won:
                        if not self.game.revealed[i][j]:
                            button.config(bg='#27AE60')
                    elif self.game.board[i][j] == -1 and self.game.revealed[i][j]:
                        button.config(bg='#C0392B')
        self.root.update()