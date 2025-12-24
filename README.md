# Solucionador de Sudoku con visión con computadora y aprendizaje profundo

## Introducción

Este proyecto en Python utiliza visión por computadora y aprendizaje profundo para resolver puzzles de Sudoku a partir de imágenes reales. El flujo de procesamiento consiste en tomar una imagen de un Sudoku, extraer automáticamente la grilla del tablero, identificar y clasificar los dígitos presentes en cada celda, resolver el puzzle mediante un algoritmo recursivo de backtracking y, finalmente, mostrar la solución superpuesta sobre la imagen original.

Para el procesamiento de imágenes se emplea OpenCV, mientras que el componente de aprendizaje profundo para el reconocimiento de dígitos fue desarrollado utilizando Keras.

El repositorio incluye un conjunto de imágenes de Sudoku de ejemplo, con variaciones en iluminación, perspectiva y fondo, ubicadas en el directorio images/. Asimismo, el sistema permite probar con imágenes propias, siempre que el tablero sea claramente visible.
