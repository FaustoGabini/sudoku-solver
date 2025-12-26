# Solucionador de Sudoku con Visión por Computadora y Aprendizaje Profundo

## Introducción

Este proyecto en **Python** utiliza **visión por computadora** y **aprendizaje profundo** para resolver puzzles de **Sudoku a partir de imágenes reales**. El flujo de procesamiento consiste en tomar una imagen de un Sudoku, extraer automáticamente la grilla del tablero, identificar y clasificar los dígitos presentes en cada celda, resolver el puzzle mediante un algoritmo recursivo de _backtracking_ y, finalmente, mostrar la solución superpuesta sobre la imagen original.

Para el procesamiento de imágenes se emplea **OpenCV**, mientras que el componente de aprendizaje profundo para el reconocimiento de dígitos fue desarrollado utilizando **Keras**.

El repositorio incluye un conjunto de imágenes de Sudoku de ejemplo, con variaciones en iluminación, perspectiva y fondo, ubicadas en el directorio `images/`. Asimismo, el sistema permite probar con imágenes propias, siempre que el tablero sea claramente visible.

## Demo

![Demo del solucionador de sudoku](assets/demo.gif)

## Implementación

El procesamiento de imágenes se realiza utilizando **OpenCV** y comienza con la conversión de la imagen original a escala de grises. A continuación, se aplica un **umbral adaptativo** para obtener una imagen binaria robusta frente a variaciones de iluminación.

Sobre esta imagen binaria se detectan los **contornos externos**, buscando específicamente aquellos cuya aproximación poligonal tenga **cuatro vértices**, ya que representan candidatos a ser el tablero de Sudoku. Entre estos, se selecciona el contorno de **mayor área**, que corresponde al tablero principal.

Una vez detectado el tablero, se ordenan sus esquinas y se calcula una **transformación de perspectiva**, que permite obtener una vista cenital (_bird’s-eye view_) del Sudoku, normalizada a un tamaño fijo de **450 × 450 píxeles**. Esta transformación facilita el procesamiento posterior y la segmentación precisa de las celdas.

A partir de la vista transformada, el tablero se divide en una grilla de **9 × 9 celdas**. Para cada celda se analiza si contiene un dígito válido. En caso afirmativo, el dígito es recortado, redimensionado a **20 × 20 píxeles**, normalizado y preparado como entrada para la red neuronal convolucional.

Para la clasificación de dígitos se entrenó una **red neuronal convolucional (CNN)** utilizando **Keras**. El dataset fue construido de forma personalizada mediante la **API de Google Fonts**, generando más de **10.000 imágenes** de los dígitos del 1 al 9 con distintas tipografías. Esto permite una mejor generalización frente a estilos no manuscritos y fuentes variadas.

Una vez clasificados los dígitos, el tablero se representa como una matriz y se utiliza un **algoritmo recursivo de backtracking** para resolver el Sudoku y obtener una solución válida.

## Instalación

Seguí estos pasos para configurar el proyecto de forma local:

1. Clonar el repositorio:
    ```bash
    git clone https://github.com/rg1990/cv-sudoku-solver.git
    cd cv-sudoku-solver
    ```
2. Crear un entorno virtual (opcional pero recomendado)
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3. Instalar las dependencias necesarias
    ```bash
    pip install -r requirements.txt
    ```

## Notas

-   Se recomienda utilizar imágenes con buena resolución y contraste para mejorar la detección del tablero y los dígitos.

-   El sistema está diseñado para Sudokus clásicos de 9 × 9.
