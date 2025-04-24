# 🏷️ TFG: Detección de Vehículos de Emergencia con Deep Learning 🚨

Este repositorio contiene el proyecto de fin de grado centrado en la detección automática de vehículos de emergencia a partir de señales de audio, usando redes neuronales en Python.

---

## 📂 Estructura del proyecto
```
├─── data # Datos con los que entrenar los modelos
│    ├─── csv_files  # Etiquetas en CSV
|    |     ├─── balanced_train_segments.csv
|    |     ├─── class_labels_indices.csv
|    |     ├─── eval_segments.csv
|    |     └─── unbalanced_train_segments.csv
│    ├─── mfcc.pkl
│    ├─── lfcc.pkl
│    └─── chroma.pkl
├─── models # Modelos obtenidos tras entrenar (MODELO_FEATURE_LOSS.keras)
│    ├─── cnn_chroma_0.1107.keras
│    ├─── cnn_lfcc_0.5374.keras
│    ├─── cnn_mfcc_0.2168.keras
│    ├─── ff_chroma_0.3828.keras
│    ├─── ff_mfcc_0.5162.keras
│    ├─── lstm_chroma_0.2633.keras
│    ├─── lstm_lfcc_0.6681.keras
│    ├─── lstm_mfcc_0.3320.keras
│    └─── encoders # Encoders obtenidos para reducción de la dimensionalidad
│         ├─── encoder_chroma.keras
│         ├─── encoder_lfcc.keras
│         └─── encoder_mfcc.keras
└─── src # Archivos para entrenar
     └─── eda.ipynb
          ├─── evaluation.ipynb
          ├─── download_data  # Archivos para la descarga de los audios desde YouTube
          │    ├─── dna.ipynb
          │    └─── dpa.ipynb
          ├─── feature_extraction # Archivos para obtener las características de los audios
          │    ├─── chroma_extraction.ipynb
          │    ├─── lfcc_extraction.ipynb
          │    └─── mfcc_extraction.ipynb
          └─── training # Archivos para entrenar los autoencoders y los modelos de clasificación
               ├─── autoencoders.ipynb
               └─── models.ipynb
```

---

## 🛠️ Requisitos
- Python ≥ 3.11
- Instalar dependencias:
  ```bash
  pip install -r requirements.txt
  ```

---

## 🚀 Uso
1. **Descarga y preprocesado de datos**:
   Los audios ya han sido descargados y procesados. Se muestra el código relevante en:
      - src/download_data/dpa.ipynb
      - src/download_data/dna.ipynb

   Para los positivos y negativos respectivamente. 
   
   Se deberá trabajar con las características ya extraídas por los archivos en src/feature_extraction/.
   Y que se encuentran almacenadas en:
      - data/mfcc.pkl
      - data/lfcc.pkl
      - data/chroma.pkl

   El formato es el siguiente para los tres casos:
   ```python
   feature = {
      'positive': [],
      'pnames': [],
      'negative': [],
      'nnames': []
   }
   ```

   Donde en 'positive' se encuentran los valores de cada muestra positiva y en su mismo índice en 'pnames', el nombre del archivo .wav. Exactamente igual para los datos negativos con 'negative' y 'nnames'.

   Un estudio detallado del conjunto de datos se encuentra en: src/eda.ipynb

2. **Entrenamiento del modelo**:
    
   Los autoencoders se pueden entrenar en: src/model/training/autoencoders

   Los modelos de clasificación se pueden entrenar en: src/model/training/models


3. **Evaluación y métricas**:
   ```bash
   python src/model_training/evaluate.py
   ```

---

*Ángel Morales* – Estudiante de Ingeniería Informática - Universidad de Cádiz
