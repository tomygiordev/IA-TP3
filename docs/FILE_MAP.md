# Mapa de archivos

Este documento registra que hace cada archivo relevante del proyecto. La idea es que cualquier integrante pueda abrir el repositorio y entender rapidamente donde tocar cada cosa.

## Raiz

| Archivo o carpeta | Rol |
| --- | --- |
| `README.md` | Presentacion principal para GitHub: objetivo, estructura, instalacion, ejecucion y resultados. |
| `requirements.txt` | Dependencias necesarias para ejecutar el proyecto. |
| `.gitignore` | Evita subir entorno virtual, caches y datos temporales. |
| `src/` | Codigo fuente del entrenamiento, evaluacion y configuracion. |
| `results/` | Evidencia generada por los entrenamientos: metricas y graficos. |
| `docs/` | Documentacion tecnica y explicativa. |
| `docs/materials/` | Materiales originales de catedra y bibliografia. |
| `docs/modules/` | Documentacion puntual de cada modulo Python. |
| `report/` | Guia para que el grupo redacte el informe final. |
| `notebooks/` | Carpeta reservada para notebooks si luego se quiere presentar una version exploratoria. |
| `data/` | Carpeta reservada para datos locales. No se usa para el dataset principal porque Hugging Face lo descarga/cachea. |

## Codigo fuente

| Archivo | Rol |
| --- | --- |
| `src/__init__.py` | Marca `src` como paquete Python. |
| `src/config.py` | Define dataset, semilla, umbral y lista de experimentos. |
| `src/data.py` | Carga `rotten_tomatoes` desde Hugging Face y prepara arrays de texto/etiqueta. |
| `src/model.py` | Define reproducibilidad, vectorizador y MLP Keras Sequential. |
| `src/evaluate.py` | Convierte probabilidades en clases, calcula metricas y guarda graficos. |
| `src/train.py` | Ejecuta todos los experimentos, elige el mejor por F1 de validacion y guarda resultados. |
| `src/smoke_test.py` | Prueba minima para verificar que el entorno funciona sin entrenar todo. |

## Resultados

| Archivo | Rol |
| --- | --- |
| `results/metrics.csv` | Tabla comparativa de todos los experimentos. |
| `results/confusion_matrix.png` | Matriz de confusion del mejor modelo evaluado en test. |
| `results/best_learning_curves.png` | Curvas de entrenamiento y validacion del mejor modelo. |
| `results/learning_curves.png` | Alias historico de las curvas del mejor modelo. |
| `results/overfitting_learning_curves.png` | Curvas especificas de la simulacion de overfitting. |

## Materiales originales

| Archivo | Rol |
| --- | --- |
| `docs/materials/TP - REDES NEURONALES..pdf` | Consigna oficial del trabajo practico. |
| `docs/materials/TEORIA REDES NEURONALES.pdf` | Material teorico de catedra. |
| `docs/materials/clase_redes_neuronales_multicapa_deep_learning_IA_21_mayo_2026 (1).pdf` | Clase sobre MLP y deep learning. |
| `docs/materials/haykin.neural-networks.3ed.2009.pdf` | Bibliografia de redes neuronales. |
| `docs/materials/redes_neuronales_mlp_deep_learning_reimaginado.html` | Material interactivo de teoria. |
