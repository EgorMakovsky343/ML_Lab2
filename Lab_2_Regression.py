import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix

data = pd.read_csv("D:\ML PROJECT\mean_mode_norm_onehot_with_deleting_train.csv")

# Разделяем данные на признаки (X) и целевую переменную (y)
X = data.drop('Age', axis=1)
y = data['Age']

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Линейная регрессия
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Предсказания
y_pred_lin = lin_reg.predict(X_test)

print("\nМетрики регрессии:")
print(f"MSE: {mean_squared_error(y_test, y_pred_lin):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_lin):.2f}")
print(f"R²: {r2_score(y_test, y_pred_lin):.2f}")