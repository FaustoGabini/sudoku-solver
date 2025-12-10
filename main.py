from SudokuSolver import SudokuSolver
import cv2 as cv
import numpy as np


solver = SudokuSolver("images/sudoku.jpg")

board = solver.extract_board(debug=True)

digits = solver.extract_digits(debug=True)
