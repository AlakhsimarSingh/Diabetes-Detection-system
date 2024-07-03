import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

csv_data = pd.read_csv('diabetes.csv')
X = csv_data.drop( columns = 'Outcome',axis = 1)
Y = csv_data['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,stratify=Y, random_state = 2)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)
f_xtrain = classifier.predict(X_train)
print("Accuracy",accuracy_score(Y_train,f_xtrain))
f_xtest = classifier.predict(X_test)
print("Accuracy of test data is",accuracy_score(Y_test,f_xtest))
input_data = (1,103,30,38,83,43.3,0.183,33)
input_data = np.asarray(input_data)
input_data = input_data.reshape(1,-1)
input_data = sc.transform(input_data)
if(classifier.predict(input_data)==1):
    print("Patient is diabetic")
else:
    print("Patient is not diabetic")


