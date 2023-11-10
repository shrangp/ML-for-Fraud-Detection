from sklearn import svm # importing SVM model from sklearn
import pandas as pd # importing pandas for data manipulation
import numpy as np # importing numpy for data manipulation
from sklearn.model_selection import train_test_split # importing train_test_split for splitting data into train and test
from sklearn.metrics import accuracy_score # importing accuracy_score for calculating accuracy
from sklearn.metrics import confusion_matrix # importing confusion_matrix for calculating confusion matrix
from sklearn.metrics import classification_report # importing classification_report for calculating classification report
from sklearn.preprocessing import StandardScaler # importing StandardScaler for feature scaling
import matplotlib.pyplot as plt # importing matplotlib.pyplot for plotting

undersampled = "CS4641 Machine Learning Resources/undersampled.csv"
smote = "CS4641 Machine Learning Resources/SMOTEsampled.csv"

# reading data from csv file
def read_data(link):
    df = pd.read_csv(link)
    if link == "CS4641 Machine Learning Resources/SMOTEsampled.csv":
        df = df[:200000]
    return df

# splitting data into train and test
def split_data(df):
    X = df.iloc[:,:-1].values # features
    y = df.iloc[:,-1].values # target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) # splitting data into train and test
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
def get_accuracy_svm(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred) # getting accuracy of SVM linear model
    return accuracy

# getting confusion matrix of SVM model
def get_confusion_matrix_svm(y_pred, y_test):
    confusion_matrix_svm = confusion_matrix(y_test, y_pred) # getting confusion matrix of SVM linear model

    # plotting confusion matrix with values
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(confusion_matrix_svm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, confusion_matrix_svm[i, j], ha='center', va='center', color='red')
    plt.show()

def get_classification_report_svm(y_pred, y_test):
    classification_report_svm = classification_report(y_test, y_pred, digits = 6) # getting classification report of SVM linear model
    return classification_report_svm

def accuracy_vs_features_plot(data, model):
    # Plotting accuracy vs number of features. Calculating accuracy for each number of features
    accuracy_list = [] # list to store accuracy
    for i in range(1, 31):
        X_train, X_test, y_train, y_test = split_data(data) # splitting data into train and test
        if model == "svm_linear":
            svm_linear = train_svm_linear(X_train[:, :i], y_train) # training SVM linear model
            y_pred = predict_svm_linear(svm_linear, X_test[:, :i]) # predicting using SVM linear model
        elif model == "svm_poly":
            svm_poly = train_svm_poly(X_train[:, :i], y_train)
            y_pred = predict_svm_poly(svm_poly, X_test[:, :i])
        accuracy = get_accuracy_svm(y_pred, y_test) # getting accuracy of SVM linear model
        accuracy_list.append(accuracy) # appending accuracy to accuracy list
    # plotting accuracy vs number of features
    plt.plot(range(1, 31), accuracy_list,scaley=True)
    plt.xlabel("Number of features")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of features")
    plt.show()



  
def run_svm_linear():
    data = read_data(undersampled) # reading data from csv file
    X_train, X_test, y_train, y_test = split_data(data) # splitting data into train and test
    svm_linear = train_svm_linear(X_train, y_train) # training SVM linear model
    y_pred = predict_svm_linear(svm_linear, X_test) # predicting using SVM linear model
    accuracy = get_accuracy_svm(y_pred, y_test) # getting accuracy of SVM linear model
    print("Accuracy of SVM linear model is: ", accuracy) # printing accuracy of SVM linear model
    print("Classification report of SVM linear model is:\n ", get_classification_report_svm(y_pred, y_test)) # printing classification report of SVM linear model
    get_confusion_matrix_svm(y_pred, y_test) # printing confusion matrix of SVM linear model
    accuracy_vs_features_plot(data, "svm_linear") # plotting accuracy vs number of features

def run_svm_poly():
    data = read_data(undersampled) # reading data from csv file
    X_train, X_test, y_train, y_test = split_data(data) # splitting data into train and test
    svm_poly = train_svm_poly(X_train, y_train) # training SVM poly model
    y_pred = predict_svm_poly(svm_poly, X_test) # predicting using SVM poly model
    accuracy = get_accuracy_svm(y_pred, y_test) # getting accuracy of SVM poly model
    print("Accuracy of SVM poly model is: ", accuracy) # printing accuracy of SVM poly model
    print("Classification report of SVM poly model is: ", get_classification_report_svm(y_pred, y_test)) # printing classification report of SVM poly model
    get_confusion_matrix_svm(y_pred, y_test) # printing confusion matrix of SVM poly model

run_svm_linear()
#run_svm_poly()