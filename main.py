from SudokuVision import SudokuVision
from DigitRecognizer import DigitRecognizer
from SudokuSolver import SudokuSolver
import cv2 as cv
import numpy as np


solver = SudokuVision("images/sudoku4.jpg", debug=True)
recognizer = DigitRecognizer("trained.keras")

solver.process_board()
digits = solver.extract_cells()

sudoku_matrix = recognizer.predict_board(digits)

print("Sudoku Matrix:")
print(sudoku_matrix)

sudoku_solver = SudokuSolver(sudoku_matrix)
sudoku_solver.solve()

print("Solved Sudoku Matrix:")
print(sudoku_solver.board)

solution = solver.draw_solution(sudoku_matrix, sudoku_solver.board)

