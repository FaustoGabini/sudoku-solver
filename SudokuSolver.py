"""
* Digit (Dígito):
    Los dígitos son los caracteres del '1' al '9'.

* Digit set (Conjunto de dígitos):
    Cuando varios dígitos podrían llenar una casilla, se usa '123' para indicar
    que 1, 2 o 3 son posibles.

* rows (filas):
    Por convención, las 9 filas tienen etiquetas de la 'A' a la 'I' (arriba a abajo).

* columns (columnas):
    Por convención, las 9 columnas tienen etiquetas del '1' al '9' (izquierda a derecha).

* Square (Casilla):
    Una casilla se nombra mediante la concatenación de sus etiquetas de fila y columna
    (p. ej., 'A9' es la casilla superior derecha).
    --> `squares` es una tupla de las 81 casillas.

* Grid (Cuadrícula):
    Una cuadrícula de 81 casillas se representa como un dict de {Square: DigitSet}.
    Ejemplo: {'A9': '123', ...}.

* Boxes (Cajas/Bloques):
    Las 9 cajas son cuadros de 3x3 dentro de la cuadrícula.
    --> `all_boxes` es una lista de las 9 cajas.

* Unit (Unidad):
    Una unidad es una fila, columna o caja; cada unidad es una tupla de 9 casillas.
    --> `all_units` es una lista de las 27 unidades.

* units:
    Diccionario tal que `units[s]` es una tupla de las 3 unidades de las que
    la casilla `s` forma parte.

* Peers (Compañeros):
    Las casillas que comparten una unidad se llaman compañeros.
    --> `peers` es un diccionario tal que `peers[s]` es un set de 20 casillas
    que están en alguna unidad con `s`.

* None:
    Si un rompecabezas no tiene solución, se representa con None.

* Picture (Imagen/Representación):
    Para entrada y salida, se usa una cadena (string) para describir la cuadrícula.

* Solution (Solución):
    Una cuadrícula es válida si cada unidad está llena con los nueve dígitos (uno por casilla,
    sin repeticiones) y coincide con los dígitos originales del rompecabezas.
"""

class SudokuSolver: 

  def __init__(self, board_matrix): 
    self.board = board_matrix.copy()

  def findEmpty(self): 
    for i in range(9): 
      for j in range(9): 
        if self.board[i][j] == 0: 
          return (i, j)  # fila, columna
    return None
  
  def isValid(self, num, pos):
    row, col = pos
    
    # Verificar fila
    for j in range(9): 
      if self.board[row][j] == num:
        return False
      
    # Verificar columna
    for i in range(9): 
      if self.board[i][col] == num:
        return False
      
    # Verificar caja 3x3
    col_block_start = 3 * (col // 3)
    row_block_start = 3 * (row // 3)

    col_block_end = col_block_start + 3
    row_block_end = row_block_start + 3

    for i in range(row_block_start, row_block_end):
      for j in range(col_block_start, col_block_end):
        if self.board[i][j] == num:
          return False
        
    return True
  
  def solve(self): 
    empty = self.findEmpty()
    if not empty: 
      return True  # Resuelto
    
    row, col = empty
    
    for num in range(1, 10): 
      if self.isValid(num, (row, col)): 
        self.board[row][col] = num
        
        if self.solve(): 
          return True
        
        self.board[row][col] = 0  # Backtrack
      
    return False