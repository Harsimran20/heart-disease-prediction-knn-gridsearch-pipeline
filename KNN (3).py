#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics import precision_score, accuracy_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


heart_df = pd.read_csv("heart.csv")


# In[3]:


heart_df.columns
heart_df.info()
heart_df['target'].nunique()
heart_df.head()


# In[4]:


X = heart_df.drop("target", axis=1)
y = heart_df["target"]


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2 , random_state = 42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[6]:


knn_classifier = KNeighborsClassifier(n_neighbors = 5)
knn_classifier.fit(X_train_scaled,y_train)


# In[7]:


y_pred = knn_classifier.predict(X_test_scaled)
print("Accuracy Score:",accuracy_score(y_test,y_pred)*100,"%")
print("Precision Score:",precision_score(y_test,y_pred)*100,"%")
print("Recall Score:",recall_score(y_test,y_pred)*100,"%")


# In[8]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[9]:


from sklearn.metrics import f1_score
print("F1 Score:", f1_score(y_test, y_pred))


# In[28]:


# Cross Validation for hyperparam tuning GridSearch
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Initialize classifier
classifier = KNeighborsClassifier()

# Define parameter grid
param_grid = {"n_neighbors": [3, 5, 7, 9]}

# GridSearch with cross-validation
classifierCV = GridSearchCV(
    estimator=classifier,
    param_grid=param_grid,
    cv=5
)

# Fit model
classifierCV.fit(X_train_scaled, y_train)

# Predictions
y_pred = classifierCV.predict(X_test_scaled)

# Evaluation
print("Accuracy Score:", accuracy_score(y_test, y_pred) * 100, "%")
print("Precision Score:", precision_score(y_test, y_pred) * 100, "%")
print("Recall Score:", recall_score(y_test, y_pred) * 100, "%")
# Results
res = pd.DataFrame(classifierCV.cv_results_)
print(res)
print("---------------------------------------------------------------------------------")
print(res[["param_n_neighbors","mean_test_score"]])
print(classifierCV.best_params_)


# In[46]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Parameter grid
param_grid = {"knn__n_neighbors": [3, 5, 7, 9]}

# GridSearch
classifierCV = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5
)

# Fit (IMPORTANT: use raw data, NOT scaled)
classifierCV.fit(X_train, y_train)

# Predict
y_pred = classifierCV.predict(X_test)

# Evaluation
print("Accuracy Score:", accuracy_score(y_test, y_pred) * 100, "%")
print("Precision Score:", precision_score(y_test, y_pred, average='weighted') * 100, "%")
print("Recall Score:", recall_score(y_test, y_pred, average='weighted') * 100, "%")

# Results
res = pd.DataFrame(classifierCV.cv_results_)
print(res)

print("---------------------------------------------------------------------------------")
print(res[["param_knn__n_neighbors", "mean_test_score"]])

print("Best Parameters:", classifierCV.best_params_)


# In[ ]:




