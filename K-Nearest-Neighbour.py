import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
data = pd.read_csv(url)

scaler = StandardScaler()

X = pd.DataFrame(scaler.fit_transform(data.drop(['Class'],axis = 1)))
Y = data['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30,random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,Y_train)

pred = knn.predict(X_test)

print('Confusion Matrix')
print(confusion_matrix(Y_test,pred))

print('Classification Report')
print(classification_report(Y_test,pred))
