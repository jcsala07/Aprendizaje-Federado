# Aprendizaje Federado con MNIST
Cecilia Paláu

Juan Carlos Sala

Rodrigo López

Ricardo Vargas

Donnet Hernandez

Yuu Ricardo Akachi

Pablo Monzon
## Descripción

Este proyecto implementa un flujo de **aprendizaje federado** usando la base de datos **MNIST** y un modelo de **TensorFlow**.  
El objetivo es simular un entorno federado donde cada integrante del equipo entrena localmente su modelo sobre una partición equivalente del conjunto de datos.

## Estructura de archivos

- **modelos/**
  Carpeta para almacenar los modelos entrenados individualmente por cada participante y los modelos globales generados después del Aprendizaje Federado.

- **TheModel.py**  
  Archivo que contiene la definición del modelo global en TensorFlow.

- **local_training.ipynb**  
  Notebook que contiene el ciclo de entrenamiento y la evaluación local de los modelos, incluyendo:
  - Curvas de aprendizaje (loss y accuracy)
  - Reporte de clasificación (classification report)
  - **Entrada de datos**: cada integrante debe cargar su partición de datos `mnist_part_n.npz`, donde `n` es el número de su partición asignada.  
    También se carga un conjunto de validación común: `mnist_validation_data.npz`.

- **global_training.ipynb**
  Notebook encargado de combinar los modelos locales para generar el modelo global.  
  Contiene la implementación de tres estrategias de agregación.
  
## Flujo de Trabajo

1. **División de Datos (confidencial)**  
   El conjunto MNIST fue dividido en 6 particiones equivalentes.

2. **Entrenamiento Local**  
   Cada integrante entrena su modelo usando su partición de datos, siguiendo el flujo definido en `local_training.ipynb`.

3. **Cómputo del Modelo Global**  
   Se combinan los modelos individuales en un modelo global usando:
   
   - **FedAvg (Federated Averaging)**  
     Promedia los pesos de todos los modelos locales para actualizar el modelo global.
   
   - **Método 2: Federated Median**  
     En lugar del promedio, se calcula la **mediana** de los pesos de los modelos locales.  
     Esto es más robusto a valores atípicos.

   - **Método 3: Trimmed Mean**
     La media recortada  es una técnica que mejora la resistencia del modelo global frente a valores atípicos.



