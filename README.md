# IMDB Sentiment Classification with BiLSTM (a type of RNN)

## 📌 What are we making?
This project is a sentiment analysis model that classifies movie reviews from the IMDB dataset as positive or negative. The goal is to build a deep learning model using BiLSTM (Bidirectional Long Short-Term Memory) to capture context from both directions in a sentence, improving classification performance.

<br>

## 🏗️ What architecture do we use?

The model uses a **Sequential architecture** consisting of:

- **Embedding Layer**: Converts tokens into dense vector representations.
- **Bidirectional LSTM Layer**: Learns context from both directions in the text.
- **Dropout Layers**: Helps prevent overfitting.
- **Dense Layer (ReLU)**: Learns complex patterns.
- **Dense Output Layer (Sigmoid)**: Outputs binary prediction (positive/negative).

<br>

## 🧰 What libraries do we use?

The following libraries were used in this project:

- `tensorflow` / `keras` – deep learning framework
- `tensorflow_datasets` – to load IMDB dataset
- `nltk` – for tokenization, stopwords, lemmatization, and POS tagging
- `scikit-learn` – for evaluation metrics (confusion matrix, classification report)
- `matplotlib` and `seaborn` – for plotting training metrics and confusion matrix
- `tqdm` – for displaying progress bars

<br>

## 🚀 How to run our model

### 1. Clone this repository

```bash
git clone https://github.com/rihannh/ML-DeepL-Assignment-1.git
```

### 2. Open the notebook:
- File name: Main.ipynb
- Run it using Jupyter Notebook, Google Colab, or Kaggle Notebook.

### 3. Run all cells from top to bottom.
The notebook includes:
- Data loading
- Preprocessing (stopword removal, lemmatization, tokenization)
- Model building and training
- Evaluation metrics (accuracy, classification report, confusion matrix)
- Visualization of loss and accuracy

### 4. Or load our trained model
```bash
from tensorflow.keras.models import load_model
model = load_model('Result Model/Model.keras')
```

<br>

## 📂 Dataset Reference

- **Dataset**: IMDB Movie Reviews  
- **Source**: Loaded using [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/imdb_reviews)  
- **Size**:  
  - **Training samples**: 25,000  
  - **Testing samples**: 25,000  
- **Labels**:  
  - `0` – Negative  
  - `1` – Positive

<br>

## 📄 Paper / Research Reference

While this project is not directly based on one single paper, it is inspired by several research works on sentiment classification with deep learning:

- Saputra, A., & Tobing, F. A. T. (2024). Sentiment Analysis of IMDB Movie Reviews Using Recurrent Neural Network Algorithm. Ultimatics : Jurnal Teknik Informatika, 16(1), 54–62. https://doi.org/10.31937/ti.v16i1.3610
- Muhammad Dzaki Arkaan Nasir, & Syarif Hidayat. (2024). Analisis Sentimen Ulasan Film Menggunakan Metode BiLSTM. Jurnal Informatika Dan Teknologi Komputer ( J-ICOM), 5(2), 126-132. https://doi.org/10.55377/j-icom.v5i2.8871
- ABD SAINI, M. S. B. (2018). SENTIMENT ANALYSIS ON MOVIE REVIEWS BY RECURRENT NEURAL NETWORKS AND LONG SHORT-TERM MEMORY.

<br>

## 📈 Model Performance

After training a model using a BiLSTM architecture and evaluating its performance on the test data, here's the model's performance:

### 🔍 Classification Report

```Classification Report
              precision    recall  f1-score   support

    Negative       0.89      0.79      0.83     12500
    Positive       0.81      0.90      0.85     12500

    accuracy                           0.84     25000
   macro avg       0.85      0.84      0.84     25000
weighted avg       0.85      0.84      0.84     25000
```
### 📊 Confusion Matrix
![Confusion Matrix](https://github.com/user-attachments/assets/870d5a5e-3407-4202-a8c0-90c26f938a5d)

### 📉 Training & Validation Accuracy/Loss
![image](https://github.com/user-attachments/assets/a9f4f2fa-353a-458f-9110-6ab39908fa27)
