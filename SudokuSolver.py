import cv2 as cv
import numpy as np

class SudokuSolver: 
  def __init__(self, img_path): 

    self.img = self._load_image(img_path)
    self.board = None
    
  def _load_image(self, path): 
    # Carga la imagen del tablero del sudoku
    img = cv.imread(path)

    if img is None:
        raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")
    return img

  def _show(self, name, img, size=(600, 600)): 
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, size[0], size[1])
    cv.imshow(name, img)

  # Ordena los puntos en sentido horario empezando por arriba-izquierda
  def _order_points(self, pts): 
    pts = pts.reshape(4, 2)
    center = pts.mean(axis=0)
    angles = []

    for p in pts:
        angle = np.arctan2(p[1] - center[1], p[0] - center[0]) * 180 / np.pi
        angles.append(angle)

    pts_cyclic = pts[np.argsort(angles)]
    return pts_cyclic.astype(np.float32)

  # Detecta el contorno del tablero y aplica una transformacion de perspectiva
  def extract_board(self, debug = False):
    
    # Preprocesamiento
    img_gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

    # Deteccion de bordes por medio de Canny
    edges = cv.Canny(img_gray, 100, 255, apertureSize=3)
    

    # Dilatacion para cerrar los bordes
    kernel = cv.getStructuringElement(
        shape=cv.MORPH_RECT, ksize=(3, 3)
    )
    mask = cv.morphologyEx(edges, op=cv.MORPH_DILATE, kernel=kernel, iterations=1)

    # Encontrar contornos
    contours, _ = cv.findContours(mask, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

    contour_candidates = []
    for contour in contours:
        epsilon = 0.1 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            contour_candidates.append(approx)
    
    if not contour_candidates:
      raise ValueError("No se encontró un tablero de Sudoku.")
    
    grid_contour = sorted(contour_candidates, key=cv.contourArea, reverse=True)[0]

    rect_src = self._order_points(grid_contour)
    
    # Transformacion de perspectiva
    side_len = 450
    rect_dst = np.array([
        [0, 0],
        [side_len - 1, 0],
        [side_len - 1, side_len - 1],
        [0, side_len - 1]
    ], dtype=np.float32) 
    
    matrix = cv.getPerspectiveTransform(rect_src, rect_dst)
    warped = cv.warpPerspective(img_gray, matrix, (side_len, side_len))

    if debug:
        self._show("Gris", img_gray)
        self._show("Canny Edges", edges)
        self._show("Dilatacion", mask)
        self._show("Sudoku Transformado", warped)
        cv.waitKey(0)
        cv.destroyAllWindows()

    self.board = warped

    return warped
  
  def _clear_border(self, cell):
    h, w = cell.shape[:2]
    margin = 6

    # Margen interno
    cell[0:margin, :] = 0
    cell[h - margin:h, :] = 0
    cell[:, 0:margin] = 0
    cell[:, w - margin:w] = 0

    cnts = cv.findContours(cell.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if len(cnts) == 0:  
        return cell
  
    c = max(cnts, key=cv.contourArea)

    mask = np.zeros(cell.shape, dtype="uint8")
    # Rellenamos de blanco el contorno mas grande
    cv.drawContours(mask, [c], -1, 255, -1)

    percent = cv.countNonZero(mask) / float(cell.shape[0]*cell.shape[1])
    if percent < 0.03:
      return cell * 0

    return cv.bitwise_and(cell, cell, mask=mask)
  
  # Muestra cada una de las celdas transformadas
  def _build_digits_grid(self, digits, cell_size, spacing=5):
    rows, cols = 9, 9

    # Altura y ancho totales de la imagen final
    grid_h = rows * cell_size + (rows + 1) * spacing
    grid_w = cols * cell_size + (cols + 1) * spacing

    # Fondo negro
    grid_img = np.zeros((grid_h, grid_w), dtype="uint8")

    for i in range(rows):
        for j in range(cols):
            cell = digits[i][j]

            # Aseguramos que la celda tenga el mismo tamaño
            if cell.shape != (cell_size, cell_size):
                cell = cv.resize(cell, (cell_size, cell_size))

            # Coordenadas de inserción (con margen interno)
            y_start = i * cell_size + (i + 1) * spacing
            x_start = j * cell_size + (j + 1) * spacing

            grid_img[y_start:y_start + cell_size,
                     x_start:x_start + cell_size] = cell

    return grid_img

  
  # Extrae los digitos del tablero
  def extract_digits(self, debug = False):    
    if self.board is None:
        raise ValueError("El tablero no ha sido extraído aún.")
    
    cell_size = self.board.shape[0] // 9
    digits = []
    for i in range(9):
        row = []
        for j in range(9):
            x_start = j * cell_size
            y_start = i * cell_size
            cell = self.board[y_start:y_start + cell_size, x_start:x_start + cell_size]
            
            # Binarizamos la imagen de la celda
            cell = cv.threshold(cell, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
            
            cell = self._clear_border(cell)
            row.append(cell)
        digits.append(row)

    if debug: 
      grid_img = self._build_digits_grid(digits, cell_size, spacing=5)
      cv.imshow("Sudoku Board", grid_img)
      cv.waitKey(0)
      cv.destroyAllWindows()
     
    return digits