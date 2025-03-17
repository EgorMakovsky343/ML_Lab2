import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

input_file = r"D:/ML PROJECT/test.csv"
df = pd.read_csv(r"D:/ML PROJECT/test.csv", delimiter=",", quotechar='"', escapechar='\\')

columns_to_drop = ['PassengerId', 'Cabin', 'Name']
df = df.drop(columns=columns_to_drop)

print(df.head())
print(df.isnull().sum())

# ОБРАБОТКА ЗНАЧЕНИЙ МЕДИАНОЙ + МОДОЙ
# num_cols = df.select_dtypes(include=np.number).columns
# cat_cols = df.select_dtypes(exclude=np.number).columns
# medians = df[num_cols].median().to_dict()
#
# for col in num_cols:
#     df[col] = df[col].fillna(medians[col])
#
# for col in cat_cols:
#     modes = df[col].mode()
#     if not modes.empty:
#         df[col] = df[col].fillna(modes.iloc[0])
#
# scaler = MinMaxScaler()
# df[num_cols] = scaler.fit_transform(df[num_cols])
#
# one_hot_encoding_cols = df[['HomePlanet', 'Destination']].columns
# df = pd.get_dummies(df, columns=one_hot_encoding_cols, drop_first=True)

# ОБРАБОТКА ЗНАЧЕНИЙ МОДОЙ
# modes = df.mode().iloc[0].to_dict()
# for col in df.columns:
#     df[col] = df[col].fillna(modes[col])
# numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
#
# scaler = MinMaxScaler()
# df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
#
# one_hot_encoding_cols = df[['HomePlanet', 'Destination']].columns
# df = pd.get_dummies(df, columns=one_hot_encoding_cols, drop_first=True)

# ОБРАБОТКА ЗНАЧЕНИЙ СРЕДНИМ ЗНАЧЕНИЕМ + МОДОЙ
# numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
#
# for col in numeric_cols:
#     df[col] = df[col].fillna(df[col].mean())
#
# for col in df.columns:
#     if col not in numeric_cols:
#         df[col] = df[col].fillna(df[col].mode()[0])
#
# scaler = MinMaxScaler()
# df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
#
# one_hot_encoding_cols = df[['HomePlanet', 'Destination']].columns
# df = pd.get_dummies(df, columns=one_hot_encoding_cols, drop_first=True)

print(df.isnull().sum())

output_file = os.path.join(os.path.dirname(input_file), "median_mode_norm_onehot_with_deleting_test.csv")
df.to_csv(output_file, index=False)