import cv2 as cv
import numpy as np

class SudokuVision: 

  # Constants 
  KERNEL_SIZE = (3, 3)
  SIDE_LEN = 450
  MARGIN = 2
  ALPHA = 0.2

  def __init__(self, img_path, debug): 
    self.img = self._load_image(img_path)
    self.solution_img = self.img.copy()
    self.debug = debug
    self.transform_matrix = None
    self.sudoku_board = None
    self.binary_board = None
    
  def _load_image(self, path):
    img = cv.imread(path)
    if img is None:
        raise ValueError("No se pudo cargar la imagen.")
    return img
  
  def _show_image(self, name, img): 
    cv.imshow(name, img)
    

  def _order_points(self, pts): 
    # Ordena los puntos en sentido horario empezando por arriba-izquierda
    pts = pts.reshape(4, 2)
    center = pts.mean(axis=0)
    angles = []

    for p in pts:
        angle = np.arctan2(p[1] - center[1], p[0] - center[0]) * 180 / np.pi
        angles.append(angle)

    pts_cyclic = pts[np.argsort(angles)]
    return pts_cyclic.astype(np.float32)

  def _transform_perspective(self, rect_src):
     # Transformacion de perspectiva
      rect_dst = np.array([
          [0, 0],
          [self.SIDE_LEN - 1, 0],
          [self.SIDE_LEN - 1, self.SIDE_LEN - 1],
          [0, self.SIDE_LEN - 1]
      ], dtype=np.float32)

      matrix = cv.getPerspectiveTransform(rect_src, rect_dst)

      return matrix
     
  
  def process_board(self):
    
    # Preprocesamiento de la imagen
    img_gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

    # Binarizamos la imagen
    img_binary = cv.adaptiveThreshold(
        img_gray,
        255,
        cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY_INV,
        11,
        20
    )
    # Realizamos la deteccion de contornos
    contours, _ = cv.findContours(
        img_binary,
        mode=cv.RETR_EXTERNAL,
        method=cv.CHAIN_APPROX_SIMPLE
    )

    # Buscamos el contorno mas grande con 4 lados
    contour_candidates = []
    for contour in contours:
      epsilon = 0.1 * cv.arcLength(contour, True)
      approx = cv.approxPolyDP(contour, epsilon, True)

      # Si la aproximacion tiene 4 lados es un candidato a tablero
      if len(approx) == 4:
        contour_candidates.append(approx)

    if not contour_candidates:
        raise ValueError("No se encontró un tablero de Sudoku.")

    grid_contour = sorted(contour_candidates, key=cv.contourArea, reverse=True)[0]

    # Visualizacion: Overlay transparente del contorno detectado
    overlay = self.solution_img.copy()

    # Rellenamos el contorno detectado
    cv.drawContours(overlay, [grid_contour], -1, (0, 0, 0), cv.FILLED)

    # Fusionamos la capa de overlay (oscura) con la imagen original
    self.solution_img = cv.addWeighted(overlay, self.ALPHA, self.solution_img, 1 - self.ALPHA, 0)

    # Dibujamos el borde exterior del contorno
    cv.drawContours(self.solution_img, [grid_contour], -1, (0, 200, 0), 3)
    
    # Transformacion de perspectiva
    # Ordenamos las esquinas (arriba-izq, arriba-der, abajo-der, abajo-izq)
    rect_src = self._order_points(grid_contour)

    self.transform_matrix = self._transform_perspective(rect_src)

    matrix_inv = np.linalg.inv(self.transform_matrix)

    # Dibujamos la grilla del Sudoku
    cell_size = self.SIDE_LEN // 9

    for i in range(1, 9): 
      thickness = 3 if i % 3 == 0 else 2

      # Líneas verticales
      x = cell_size * i
      # Definimos la linea en el espacio plano
      pts_v = np.array([
          [[x, 0]],
          [[x, self.SIDE_LEN]]
      ], dtype=np.float32)
        
      # Líneas horizontales
      y = i * cell_size
      pts_h = np.array([
          [[0, y]],
          [[self.SIDE_LEN, y]]
      ], dtype=np.float32)

      # Proyectamos los puntos de vuelta a la imagen original
      pts_orig_h = cv.perspectiveTransform(pts_h, matrix_inv)
      pts_orig_v = cv.perspectiveTransform(pts_v, matrix_inv)

      # Dibujamos las líneas en la imagen original
      # pts_orig_h[0] = [[x0, y0]]
      p1 = tuple(pts_orig_h[0][0].astype(int))
      p2 = tuple(pts_orig_h[1][0].astype(int))

      cv.line(self.solution_img, p1, p2, (0, 200, 0), thickness)

      p1 = tuple(pts_orig_v[0][0].astype(int))
      p2 = tuple(pts_orig_v[1][0].astype(int))
      cv.line(self.solution_img, p1, p2, (0, 200, 0), thickness)
  
    self.binary_board = cv.warpPerspective(img_binary, self.transform_matrix, (self.SIDE_LEN, self.SIDE_LEN))
    self.sudoku_board = cv.warpPerspective(img_gray, self.transform_matrix, (self.SIDE_LEN, self.SIDE_LEN))

    if self.debug:
      self._show_image("Sudoku Binarizado", self.binary_board)
      self._show_image("Sudoku Transformado", self.sudoku_board)
      self._show_image("Sudoku detectado + grilla", self.solution_img)

  def extract_cells(self): 
    if self.sudoku_board is None:
      raise ValueError("El tablero no ha sido extraído aún.")
    
    cell_size = self.sudoku_board.shape[0] // 9
    digits = []

    for i in range(9):
        row = []
        for j in range(9):
          x_start = j * cell_size
          y_start = i * cell_size
            
          gray_cell = self.sudoku_board[y_start:y_start + cell_size, x_start:x_start + cell_size]
          binary_cell = self.binary_board[y_start:y_start + cell_size, x_start:x_start + cell_size] 
          
          height, width = binary_cell.shape      

          num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary_cell, connectivity=8)

          cell_cx, cell_cy = width // 2, height // 2
          max_dist = 0.25 * min(width, height)

          best_label = None
          min_distance = float('inf') 

          for label in range(1, num_labels):
            x, y, bw, bh, area = stats[label]
            cx, cy = centroids[label]

            # Descartar componentes que tocan bordes
            if (
              x <= self.MARGIN or y <= self.MARGIN or
              x + bw >= width - self.MARGIN or
              y + bh >= height - self.MARGIN
              ):
              continue

            # Descartar componentes muy pequeñas  
            if area < 0.02 * width * height:
              continue

            # Calcular distancia al centro de la celda
            distance = np.sqrt((cx - cell_cx) ** 2 + (cy - cell_cy) ** 2)
            # Descartar si la distancia es mayor que la mínima encontrada
            if distance < min_distance and distance < max_dist:
              min_distance = distance
              best_label = label
                
          if best_label is None: 
            # No se encontró un número válido
            row.append(np.zeros((20, 20), dtype=np.uint8))
          else: 
            x, y, bw, bh, _ = stats[best_label]
            digit_img = gray_cell[y:y+bh, x:x+bw]
            resized_img = cv.resize(digit_img, (20, 20))
            normalized_img = resized_img / 255.0
            processed = np.expand_dims(normalized_img, axis=-1)
            row.append(processed)
            
        digits.append(row)
    
    return digits

  
  def draw_solution(self, sudoku_matrix): 
    if self.transform_matrix is None: 
      raise ValueError("La transformación de perspectiva no ha sido calculada.")
    
    solution_img = np.zeros((self.SIDE_LEN, self.SIDE_LEN, 3), dtype=np.uint8)
    cell_size = self.SIDE_LEN // 9

    # Escribimos los numeros en la imagen negra plana
    for i in range(9): 
      for j in range(9): 
        num = sudoku_matrix[i][j]
        if num == 0: 
          text = str(np.random.randint(1, 10))
          font = cv.FONT_HERSHEY_SIMPLEX
          font_scale = 1
          thickness = 3
          text_size, _ = cv.getTextSize(text, font, font_scale, thickness)
          text_x = j * cell_size + (cell_size - text_size[0]) // 2
          text_y = i * cell_size + (cell_size + text_size[1]) // 2
          cv.putText(solution_img, text, (text_x, text_y), font, font_scale, (0, 200, 0), thickness)
    
    # Invertimos la transformación de perspectiva
    matrix_inv = np.linalg.inv(self.transform_matrix)
    warped_solution = cv.warpPerspective(solution_img, matrix_inv, (self.img.shape[1], self.img.shape[0]))

    # Combinamos la imagen original con la solución
    # Creamos la máscara
    gray_sol = cv.cvtColor(warped_solution, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray_sol, 40, 255, cv.THRESH_BINARY)

    # "En la imagen original, donde la máscara no sea negra, ponemos los píxeles de la solución"
    self.solution_img[mask != 0] = warped_solution[mask != 0]
    if self.debug:
      self._show_image("Sudoku Original", self.img)
      self._show_image("Sudoku Binarizado", self.binary_board)
      self._show_image("Sudoku Transformado", self.sudoku_board)
      self._show_image("Solucion del Sudoku", self.solution_img)   
      cv.waitKey(0)
      cv.destroyAllWindows() 
    

    return self.solution_img