import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
data = pd.read_csv(url)

print('data.head()')
print(data.head())

print('data.info()')
print(data.info())

print('data.describe()')
print(data.describe())

print('data.isnull().sum()')
print(data.isnull().sum())

print("data['Class'].value_counts()")
print(data['Class'].value_counts())

fraud = data[data['Class'] == 1]
legit = data[data['Class'] == 0]

print('fraud.shape')
print(fraud.shape)

print('legit.shape')
print(legit.shape)

print('fraud.Amount.describe()')
print(fraud.Amount.describe())

print('legit.Amount.describe()')
print(legit.Amount.describe())

