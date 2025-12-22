from SudokuVision import SudokuVision
from DigitRecognizer import DigitRecognizer
import cv2 as cv
import numpy as np


solver = SudokuVision("images/sudoku.jpg", debug=True)
recognizer = DigitRecognizer("trained.keras")

solver.process_board()
digits = solver.extract_cells()

sudoku_matrix = recognizer.predict_board(digits)

print("Sudoku Matrix:")
print(sudoku_matrix)

solution = solver.draw_solution(sudoku_matrix)

