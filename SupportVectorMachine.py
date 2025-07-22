import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import preprocessing
import imblearn 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
data = pd.read_csv(url)

data = data.drop("Time", axis=1)

scaler = preprocessing.StandardScaler()

data['std_Amount'] = scaler.fit_transform(data['Amount'].values.reshape (-1,1))

undersample = RandomUnderSampler(sampling_strategy=0.5)

cols = data.columns.tolist()
cols = [c for c in cols if c not in ["Class"]]
target = "Class"

X = data[cols]
Y = data[target]

X_under, Y_under = undersample.fit_resample(X, Y)

test = pd.DataFrame(Y_under, columns = ['Class'])

X_train, X_test, Y_train, Y_test = train_test_split(X_under, Y_under, test_size=0.2, random_state=0)

model = SVC()
model.fit(X_train,Y_train)

model2 = SVC(probability=True, random_state=2)
svm = model2.fit(X_train, Y_train)

Y_pred_svm = model2.predict(X_test)

print("Accuracy SVM:",metrics.accuracy_score(Y_test, Y_pred_svm))
print("Precision SVM:",metrics.precision_score(Y_test, Y_pred_svm))
print("Recall SVM:",metrics.recall_score(Y_test, Y_pred_svm))
print("F1 Score SVM:",metrics.f1_score(Y_test, Y_pred_svm))





