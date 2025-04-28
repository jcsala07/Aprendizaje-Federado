# Aprendizaje Federado con MNIST

## Descripción

Este proyecto implementa un flujo de **aprendizaje federado** usando la base de datos **MNIST** y un modelo de **TensorFlow**.  
El objetivo es simular un entorno federado donde cada integrante del equipo entrena localmente su modelo sobre una partición equivalente del conjunto de datos.

## Estructura de archivos

- **'Individual_Training_Models/'**  
  Carpeta para almacenar los modelos entrenados individualmente por cada participante.

- **'TheModel.py'**  
  Archivo que contiene la definición del modelo global en TensorFlow.

- **'local_training.ipynb'**  
  Notebook que contiene el ciclo de entrenamiento y la evaluación local de los modelos, incluyendo:
  - Curvas de aprendizaje (loss y accuracy)
  - Reporte de clasificación (classification report)

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

   - **Método 3: ?**  

