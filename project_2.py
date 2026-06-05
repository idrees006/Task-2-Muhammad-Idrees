import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. INPUT: Load the Iris Benchmark Dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 2. PROCESS: Algorithm & Tuning
from sklearn.neighbors import KNeighborsClassifier

# 3. OUTPUT: Validation Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ==========================================
# STEP 1: INPUT & PREPROCESSING
# ==========================================
print("=== [1/3] INPUT PHASE ===")
# Load the 150 balanced samples with 4 dimensions (Sepal/Petal length & width)
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(print(f"Dataset Loaded Successfully."))
print(f"• Total Samples: {X.shape[0]} (Balanced Across {len(target_names)} Classes)")
print(f"• Dimensions (Features): {feature_names}\n")

# Structural Integrity: The 80/20 Split with Shuffling to remove order bias
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y
)
print(f"• Training Set Size (80%): {X_train.shape[0]} samples")
print(f"• Test Set Size (20%): {X_test.shape[0]} samples\n")

# The Gatekeeper Rule: Feature Scaling (StandardScaler: Mean = 0, Variance = 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✔ Feature Scaling Applied via StandardScaler.\n")


# ==========================================
# STEP 2: PROCESS (TRAINING THE MODEL)
# ==========================================
print("=== [2/3] PROCESS PHASE ===")
# Workflow: Instantiate the frame with K=5 Majority Vote
k_value = 5
model = KNeighborsClassifier(n_neighbors=k_value)
print(f"• Model Instantiated: KNeighborsClassifier(n_neighbors={k_value})")

# Fit: Memorize the map
model.fit(X_train_scaled, y_train)
print("✔ Model Map Memorized (Training Complete).")

# Predict: Apply logic to the unseen validation data
y_pred = model.predict(X_test_scaled)
print("✔ Predictions Generated on Test Data.\n")


# ==========================================
# STEP 3: OUTPUT VALIDATION
# ==========================================
print("=== [3/3] OUTPUT VALIDATION PHASE ===")

# Looking deeper than the "Accuracy Mirage"
accuracy = accuracy_score(y_test, y_pred)
print(f"• Raw Accuracy Score: {accuracy * 100:.2f}%")

print("\n• Strategic Trade-offs (Classification Report):")
print(classification_report(y_test, y_pred, target_names=target_names))

# Generate the Diagnostic Tool: Confusion Matrix Data
cm = confusion_matrix(y_test, y_pred)

# Visualizing the Confusion Matrix for your project submission portfolio
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'Project 2: KNN (K={k_value}) Confusion Matrix Blueprint', fontsize=14)
plt.xlabel('Predicted Label (Output)', fontsize=12)
plt.ylabel('True Label (Input)', fontsize=12)
plt.tight_layout()
plt.show()

print("✔ Pipeline Architecture Completed Successfully. Diagnostic Matrix Rendered.")