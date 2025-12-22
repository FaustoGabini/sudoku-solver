import numpy as np
from tensorflow.keras.models import load_model # <--- Importar esto

class DigitRecognizer: 
  
  def __init__(self, model_path):
    self.model = load_model(model_path)

  def predict_board(self, cell_images): 

    sudoku_matrix = np.zeros((9, 9), dtype=int)

    batch_images = [] # Fotos para la IA
    locations = []    # Ubicaciones de las celdas
    
    for i in range(9): 
      for j in range(9): 
        cell_img = cell_images[i][j]
        
        if cell_img.sum() == 0: 
          # Celda vacía
          sudoku_matrix[i, j] = 0
        else:
          batch_images.append(cell_img)
          locations.append((i, j))

    if batch_images:
      batch_array = np.array(batch_images)
      predictions = self.model.predict(batch_array)

    for idx, (row, col) in enumerate(locations): 
      probs = predictions[idx]
      predicted_digit = np.argmax(probs) + 1  # Dígitos del 1 al 9

      sudoku_matrix[row, col] = predicted_digit

    return sudoku_matrix
      

         