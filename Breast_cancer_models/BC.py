# In this exercise I followed this link https://towardsdatascience.com/building-a-simple-machine-learning-model-on-breast-cancer-data-eca4b3b99fa3

# Importing the necessary libraries

import numpy as np
import matplotlib as plt
import pandas as pd

############Data get#######################

#Importing the dataset I'll use

cancer_data = pd.read_csv('wdbcdata.csv')

############Data prep and vis#######################

#The actual data wo the label
X=cancer_data.iloc[:,2:].values
#Getting the label (diagnosis)
Y=cancer_data.iloc[:,1].values

print(cancer_data.head())
print('Cancer dataset dimentions : {}' .format(cancer_data.shape))

#Looking at the labels
print(cancer_data['diagnosis'].value_counts())

# Visualising all
#cancer_data.hist(bins=50, figsize=(20,15))

#Are there any nulls?
series_of_CD = cancer_data.count()
print((series_of_CD < cancer_data.shape[0]).any())

#Or

Null_ = cancer_data.isnull().sum()
Na_ = cancer_data.isna().sum()
print((Null_==0).all())
print((Na_==0).all())

#Data is complete

############Categorical data#######################

#Turn the cat. data into numerical values to use with our models

from sklearn.preprocessing import LabelEncoder

label_encoder_Y = LabelEncoder()
Encoded_Y = label_encoder_Y.fit_transform(Y)

############Split the data#######################

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Encoded_Y, test_size=0.25, random_state=0)

############Feature scaling#######################

#Why?...to address differences in magnityude between the different attributes

from sklearn.preprocessing import StandardScaler

Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.fit_transform(X_test)

############Model Selection#######################

# This task calls for a is a supervised / classification model 

from sklearn.metrics import confusion_matrix

#Using Logistic Regression Algorithm to the Training Set
from sklearn.linear_model import LogisticRegression
logregmod =  LogisticRegression(random_state=0).fit(X_train, y_train)
print('Logistic Regression Algorithm accuracy: {:.3f}' .format(logregmod.score(X_test, y_test)))
print(confusion_matrix(y_test, logregmod.predict(X_test)))

#Using K-neighbors Algorithm to the Training Set
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier().fit(X_train, y_train)
print('K-neighbors Algorithm accuracy: {:.3f}' .format(neigh.score(X_test, y_test)))
print(confusion_matrix(y_test, neigh.predict(X_test)))

#Using SVM linear Algorithm to the Training Set
from sklearn.svm import LinearSVC
SVMLinear = LinearSVC(random_state=0).fit(X_train, y_train)
print('SVM linear Algorithm accuracy: {:.3f}' .format(SVMLinear.score(X_test, y_test)))
print(confusion_matrix(y_test, SVMLinear.predict(X_test)))

#Using SVM rbf to the Training Set
from sklearn.svm import SVC
SVM = SVC(random_state=0).fit(X_train, y_train)
print('SVM RBF Algorithm accuracy: {:.3f}' .format(SVM.score(X_test, y_test)))
print(confusion_matrix(y_test, SVM.predict(X_test)))

#Using Naive Bayes to the Training Set
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB().fit(X_train, y_train)
print('Naive Bayes accuracy: {:.3f}' .format(NB.score(X_test, y_test)))
print(confusion_matrix(y_test, NB.predict(X_test)))

#Using Decision tree algorithm to the Training Set
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(random_state=0, criterion = 'entropy').fit(X_train, y_train)
print('Decision Tree Algorithm accuracy: {:.3f}' .format(DTC.score(X_test, y_test)))
print(confusion_matrix(y_test, DTC.predict(X_test)))

#Using Random Forest to the Training Set
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0).fit(X_train, y_train)
print('Random forest accuracy: {:.3f}' .format(RF.score(X_test, y_test)))
print(confusion_matrix(y_test, RF.predict(X_test)))


