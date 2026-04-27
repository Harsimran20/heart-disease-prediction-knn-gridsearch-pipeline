# heart-disease-prediction-knn-gridsearch-pipeline


## 📌 Overview

This project implements a machine learning pipeline to predict the presence of heart disease using the K-Nearest Neighbors (KNN) algorithm. It includes data preprocessing, model training, evaluation, and hyperparameter tuning using Grid Search with cross-validation.

The implementation leverages the scikit-learn framework for building a robust and scalable ML workflow.

---

## 🎯 Objectives

* Perform data preprocessing and feature scaling
* Train a KNN classification model
* Evaluate model performance using multiple metrics
* Optimize hyperparameters using GridSearchCV
* Implement a Pipeline to prevent data leakage

---

## 📂 Dataset

* **Dataset**: Heart Disease Dataset (`heart.csv`)
* **Target Variable**: `target`

  * `0` → No Disease
  * `1` → Presence of Disease

---

## ⚙️ Features

* Data splitting (Train/Test)
* Feature scaling using StandardScaler
* KNN model training
* Model evaluation:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
* Confusion Matrix
* Hyperparameter tuning (`n_neighbors`)
* Pipeline integration with GridSearchCV

---

## 🛠️ Technologies Used

* Python
* scikit-learn
* Pandas

---

## 🚀 Workflow

### 1. Data Preprocessing

* Load dataset using Pandas
* Split features and target
* Train-test split

### 2. Feature Scaling

* Applied using `StandardScaler`

### 3. Model Training

* KNN classifier trained on scaled data

### 4. Model Evaluation

* Accuracy, Precision, Recall, F1 Score
* Confusion Matrix

### 5. Hyperparameter Tuning

* GridSearchCV with 5-fold cross-validation
* Parameter tuned: `n_neighbors`

### 6. Pipeline Implementation

* Combines scaling + model training
* Eliminates data leakage
* Improves reproducibility

---

## 🧪 Sample Code (Pipeline + GridSearch)

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

param_grid = {"knn__n_neighbors": [3, 5, 7, 9]}

grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X_train, y_train)
```

---

## 📊 Evaluation Metrics

* **Accuracy** → Overall correctness
* **Precision** → Correct positive predictions
* **Recall** → Ability to detect positives
* **F1 Score** → Balance between precision and recall

---

## 📈 Example Output

```
Accuracy Score: 92.3 %
Precision Score: 91.5 %
Recall Score: 92.0 %
Best Parameters: {'knn__n_neighbors': 5}
```

---

## ⚠️ Key Learnings

* Pipelines prevent **data leakage**
* GridSearchCV automates **hyperparameter tuning**
* Proper scaling significantly improves KNN performance

---

## 🔮 Future Enhancements

* Compare KNN with other models (SVM, Random Forest)
* Add feature selection techniques
* Visualize performance metrics
* Deploy model using Flask or FastAPI

---
t description
* Resume bullet points
