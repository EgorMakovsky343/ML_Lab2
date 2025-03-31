import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

data = pd.read_csv("D:\ML PROJECT\mean_mode_norm_onehot_with_deleting_train.csv")

# Разделяем данные на признаки (X) и целевую переменную (y)
X = data.drop('CryoSleep', axis=1)
y = data['CryoSleep']

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Логистическая регрессия
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train, y_train)

# Предсказания и оценка
y_pred = log_reg.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Логистическая регрессия:")
print("Точность (Accuracy):", accuracy_log)
print("Полнота (Recall):", recall)
print("Точность (Precision):", precision)
print("F1-мера:", f1)
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))