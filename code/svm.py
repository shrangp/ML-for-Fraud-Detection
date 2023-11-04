from sklearn import svm # importing SVM model from sklearn
import pandas as pd # importing pandas for data manipulation
import numpy as np # importing numpy for data manipulation
from sklearn.model_selection import train_test_split # importing train_test_split for splitting data into train and test
from sklearn.metrics import accuracy_score # importing accuracy_score for calculating accuracy
from sklearn.metrics import confusion_matrix # importing confusion_matrix for calculating confusion matrix
from sklearn.metrics import classification_report # importing classification_report for calculating classification report
from sklearn.preprocessing import StandardScaler # importing StandardScaler for feature scaling
import matplotlib.pyplot as plt # importing matplotlib.pyplot for plotting



# reading data from csv file
def read_data():
    link = "C:/Users/saara/OneDrive/Desktop/CS 4641/Project/ml_project_team_34/data/CS4641 Machine Learning Resources/undersampled.csv"
    df = pd.read_csv(link)
    if link == "C:/Users/saara/OneDrive/Desktop/CS 4641/Project/ml_project_team_34/data/CS4641 Machine Learning Resources/SMOTEsampled.csv":
        df = df[:200000]
    return df

# splitting data into train and test
def split_data(df):
    X = df.iloc[:,:-1].values # features
    y = df.iloc[:,-1].values # target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=42) # splitting data into train and test
    return X_train, X_test, y_train, y_test

# SVM MODEL

# training SVM linear model
def train_svm_linear(X_train, y_train):
    svm_linear = svm.SVC(kernel='linear') # creating SVM linear model
    svm_linear.fit(X_train, y_train) # training SVM linear model
    return svm_linear

# predicting using SVM linear model
def predict_svm_linear(svm_linear, X_test):
    y_pred = svm_linear.predict(X_test) # predicting using SVM linear model
    return y_pred

# training SVM poly model
def train_svm_poly(X_train, y_train):
    svm_poly = svm.SVC(kernel='poly') # creating SVM poly model
    svm_poly.fit(X_train, y_train) # training SVM poly model
    return svm_poly

# predicting using SVM poly model
def predict_svm_poly(svm_poly, X_test):
    y_pred = svm_poly.predict(X_test) # predicting using SVM linear model
    return y_pred
#get accuracy of SVM linear model
def get_accuracy_svm_linear(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred) # getting accuracy of SVM linear model
    return accuracy  

data = read_data() # reading data from csv file
X_train, X_test, y_train, y_test = split_data(data) # splitting data into train and test
svm_linear = train_svm_linear(X_train, y_train) # training SVM linear model
y_pred = predict_svm_linear(svm_linear, X_test) # predicting using SVM linear model
accuracy = get_accuracy_svm_linear(y_pred, y_test) # getting accuracy of SVM linear model
print("Accuracy of SVM linear model is: ", accuracy) # printing accuracy of SVM linear model
