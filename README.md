# Task-2-Muhammad-Idrees

# 📊 KNN Iris Classification Pipeline

A machine learning project demonstrating the **K-Nearest Neighbors (KNN)** algorithm applied to the classic **Iris dataset**. This project follows a structured INPUT → PROCESS → OUTPUT methodology with comprehensive model validation.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Results](#results)
- [Skills Demonstrated](#skills-demonstrated)
- [Author](#author)

---

## 🎯 Project Overview

**Task-2** implements a complete machine learning pipeline using the **KNN classifier** on the Iris dataset. The project demonstrates:

- **Data Input & Preprocessing**: Loading and scaling the Iris dataset
- **Model Training**: K-Nearest Neighbors with K=5
- **Validation & Metrics**: Comprehensive classification evaluation
- **Visualization**: Confusion matrix heatmap for results analysis

This project showcases the full ML workflow from data preparation to model validation.

---

## ✨ Features

✅ **Iris Dataset Integration** – 150 balanced samples with 4 dimensions  
✅ **KNN Classification** – K=5 majority voting classifier  
✅ **Feature Scaling** – StandardScaler for normalized features (mean=0, variance=1)  
✅ **Train/Test Split** – 80/20 stratified split with shuffle  
✅ **Comprehensive Metrics** – Accuracy, precision, recall, F1-score  
✅ **Confusion Matrix Visualization** – Heatmap for error analysis  
✅ **Classification Report** – Per-class performance breakdown  

---

## 📦 Requirements

- Python 3.8+
- scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn

---

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/idrees006/Task-2-Muhammad-Idrees.git
cd Task-2-Muhammad-Idrees

# Install dependencies
pip install scikit-learn numpy pandas matplotlib seaborn
```

---

## 🚀 How to Run

```bash
# Run the project
python project_2.py
```

The script will:
1. Load the Iris dataset (150 samples, 4 features)
2. Split data into 80% training and 20% test sets
3. Apply StandardScaler for feature normalization
4. Train KNN model with K=5
5. Generate predictions on test data
6. Display accuracy, classification report, and confusion matrix
7. Visualize results with a heatmap

---

## 📊 Project Structure

```
project_2.py
├── [1/3] INPUT PHASE
│   ├── Load Iris dataset
│   ├── Train/test split (80/20 stratified)
│   └── Feature scaling (StandardScaler)
├── [2/3] PROCESS PHASE
│   ├── Instantiate KNeighborsClassifier (K=5)
│   ├── Train model on scaled features
│   └── Generate predictions
└── [3/3] OUTPUT VALIDATION PHASE
    ├── Calculate accuracy
    ├── Print classification report
    └── Visualize confusion matrix
```

---

## 🤖 Model Details

| Parameter | Value |
|-----------|-------|
| **Algorithm** | K-Nearest Neighbors |
| **K Value** | 5 |
| **Dataset** | Iris (150 samples, 4 features) |
| **Classes** | 3 (Setosa, Versicolor, Virginica) |
| **Training Set** | 120 samples (80%) |
| **Test Set** | 30 samples (20%) |
| **Scaler** | StandardScaler |

---

## 📈 Expected Results

- High accuracy on Iris dataset (typically > 95%)
- Balanced classification across all three iris species
- Low false positives/negatives in confusion matrix
- Strong F1-scores for each class

---

## 📚 Skills Demonstrated

| Skill | Implementation |
|-------|----------------|
| **Data Loading** | sklearn.datasets.load_iris |
| **Data Preprocessing** | Train/test split, stratification, feature scaling |
| **Model Training** | KNeighborsClassifier with hyperparameter tuning |
| **Model Evaluation** | Accuracy, precision, recall, F1-score |
| **Visualization** | Seaborn heatmap for confusion matrix |
| **ML Pipeline** | Structured INPUT → PROCESS → OUTPUT workflow |

---

## 👤 Author

**Muhammad Idrees**

---

## 📝 License

This project is educational and part of a machine learning course.
