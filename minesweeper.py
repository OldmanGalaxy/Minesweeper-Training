import tkinter as tk
from tkinter import ttk
import random
import numpy as np

class MinesweeperGame:
    def __init__(self, rows: int, cols: int, num_mines: int):
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.board = np.zeros((rows, cols), dtype=int)
        self.initialize_board()
        
    def initialize_board(self):
        available_cells = [(i, j) for i in range(self.rows) for j in range(self.cols)]
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

class MinesweeperViewer:
    def __init__(self, rows: int, cols: int, num_mines: int):
        self.game = MinesweeperGame(rows, cols, num_mines)
        
        self.root = tk.Tk()
        self.root.title("Viewer")
        self.grid_frame = ttk.Frame(self.root)
        self.grid_frame.grid(padx=10, pady=10)
        
        self.labels = []
        for i in range(rows):
            row = []
            for j in range(cols):
                frame = ttk.Frame(self.grid_frame, borderwidth = 1, relief = "solid")
                frame.grid(row = i, column = j, padx = 1, pady = 1)
                label = ttk.Label(frame, width = 2, anchor = "center")
                label.grid(padx=2, pady=2)
                row.append(label)
            self.labels.append(row)
        
        self.update_display()
        
        self.new_game_button = ttk.Button(self.root, text = "New Game", command = self.new_game)
        self.new_game_button.grid(pady = 10)
    
    def update_display(self):
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                value = self.game.board[i][j]
                if value == -1:
                    text = "💣"
                elif value == 0:
                    text = " "
                else:
                    text = str(value)
                self.labels[i][j].config(text=text)
    
    def new_game(self):
        self.game = MinesweeperGame(self.game.rows, self.game.cols, self.game.num_mines)
        self.update_display()
    
    def run(self):
        self.root.mainloop()

def main():
    rows = int(input("Enter number of rows: "))
    cols = int(input("Enter number of columns: "))
    num_mines = int(input("Enter number of mines: "))
    
    viewer = MinesweeperViewer(rows, cols, num_mines)
    viewer.run()

if __name__ == "__main__":
    main()