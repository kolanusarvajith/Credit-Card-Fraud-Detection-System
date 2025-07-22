import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
data = pd.read_csv(url)

X = data.drop('Class', axis=1)
Y = data['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)

rfc = RandomForestClassifier(n_estimators=100, random_state=0)

rfc.fit(X_train, Y_train)


predictions = rfc.predict(X_test)

print('Classification Report')
print(classification_report(Y_test, predictions))

print('Confusion Matrix')
print(confusion_matrix(Y_test, predictions))