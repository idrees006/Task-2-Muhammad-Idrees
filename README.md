# Task-2-Muhammad-Idrees

# 🧑‍💻 Fairness-Aware ML Pipeline | Face Classification

A fairness-focused machine learning project built with Python (PyTorch) and Google Colab that demonstrates bias mitigation using the **FairFace dataset**. This project is part of the **DecodeLabs AI program**.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Skills Demonstrated](#skills-demonstrated)
- [Author](#author)

---

## 🎯 Project Overview

**Task-2** implements a **fairness-aware ML pipeline** for face classification. It integrates:

- **Pseudo-balancing** to reduce bias across demographic subgroups
- **Semi-supervised learning (FixMatch style)** for improved generalization
- **ResNet18 backbone** for robust feature extraction
- **Fairness metrics** (accuracy + selection rate) to evaluate subgroup performance

This project demonstrates how fairness-aware pipelines can mitigate bias in real-world datasets.

---

## ✨ Features

✅ **FairFace Dataset Integration** – Diverse demographic representation  
✅ **Pseudo-Balancing** – Adjusts training distribution for fairness  
✅ **Semi-Supervised Learning** – Confidence thresholding for unlabeled data  
✅ **ResNet18 Model** – Pretrained backbone for classification  
✅ **Bias Metrics** – Accuracy + fairness evaluation across subgroups  
✅ **Hyperparameter Tuning** – Configurable thresholds and batch sizes  
✅ **Colab Ready** – Runs seamlessly in Google Colab  

---

## 📦 Requirements

- Python 3.8+
- PyTorch & Torchvision
- NumPy, Pandas, Matplotlib
- Google Colab (recommended)

---

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/idrees006/Task-2-Muhammad-Idrees.git
cd Task-2-Muhammad-Idrees

# Install dependencies
pip install torch torchvision numpy pandas matplotlib
```

---

## 🚀 How to Run

1. Open the Jupyter notebook in Google Colab
2. Install required packages
3. Load the FairFace dataset
4. Train the fairness-aware model with pseudo-balancing
5. Evaluate fairness metrics across demographic subgroups

---

## 📊 Skills Demonstrated

| Skill | Implementation |
|-------|----------------|
| **Fairness-Aware ML** | Pseudo-balancing + subgroup metrics |
| **Semi-Supervised Learning** | FixMatch confidence thresholding |
| **Deep Learning** | ResNet18 backbone |
| **Data Processing** | FairFace dataset integration |
| **Evaluation** | Accuracy + fairness metrics |
| **Experimentation** | Hyperparameter tuning in Colab |

---

## ✅ Qualification Criteria Met

✅ FairFace dataset integration  
✅ Pseudo-balancing fairness pipeline  
✅ Semi-supervised learning implementation  
✅ ResNet18 model training  
✅ Subgroup fairness evaluation  
✅ Hyperparameter tuning  

---

## 👤 Author

**Muhammad Idrees**  
DecodeLabs AI Program

---

## 📝 License

This project is part of the DecodeLabs AI program.
