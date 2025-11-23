# 🏷️ BS Thesis: Emergency Vehicle Detection with Deep Learning 🚨

This repository contains the BS Thesis focused on automatic detection of emergency vehicles from audio signals using deep neural networks in Python.

---

## 📂 Project Structure
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

## 🛠️ Requirements

- Python ≥ 3.11  
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

---

## 🦜Demostration usage
1. **Run backend**
   ```bash
   cd backend
   python api.py
   ```
2. **Run frontend**
   ```bash
   cd frontend
   netlify dev
   ```


## 🚀 Technical details
1. **Data Download & Preprocessing**:
   Audio files have already been downloaded and preprocessed. Relevant code can be found in:
      - `src/download_data/dpa.ipynb`
      - `src/download_data/dna.ipynb`

   For positive and negative audio files respectively. 
   
   You should work with features extracted from: `src/feature_extraction`.
   They are stored in:
      - data/mfcc.pkl
      - data/lfcc.pkl
      - data/chroma.pkl

   Format for all three cases:
   ```python
   feature = {
      'positive': [],
      'pnames': [],
      'negative': [],
      'nnames': []
   }
   ```

   In `positive`, values for positive data is stored, while in `pnames` their .wav names are stored (same index). Same thing for  `negative` and `nnames`.

   A detailed exploratory data analysis is available in: `src/eda.ipynb`.

2. **Model training**:
    
     Autoencoders can be trained in `src/model/training/autoencoders`.  
     You should select the desired technique at the top of the file.

     Classification models can be trained in `src/model/training/models`.  
     You should choose the architecture and technique you wish to use at the top of the file.

3. **Evaluation & Metrics**:
   ```bash
   python src/evaluation.ipynb
   ```
   
   You should choose the architecture and technique you wish to use at the top of the file.

---

*Ángel Morales* – Computer Engineering student - University of Cadiz
