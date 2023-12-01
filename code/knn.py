import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
from sklearn.metrics import precision_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, KFold, cross_validate
import gdown
import numpy as np
import time

def evaluate_knn_with_data(file_id, filename, test_size=0.20, n_neighbors=50, fraction=1.0):
    # Download the file using gdown
    #gdown.download(f'https://drive.google.com/uc?id={file_id}', filename, quiet=False)

    # Load the dataset
    df = pd.read_csv(filename)
    df = df.sample(frac=fraction, random_state=0)
    print("Hello World")

    # Assuming the last column is the label
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target variable
    """
    # Split the dataset into training set and test set with given ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize KNN classifier with given number of neighbors
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit the classifier to the data
    knn.fit(X_train, y_train)

    # Predict the test set results
    y_pred = knn.predict(X_test)

    # Evaluate the classifier
    print(f'Confusion Matrix for {filename}:')
    print(confusion_matrix(y_test, y_pred))
    print(f'Classification Report for {filename}:')
    print(classification_report(y_test, y_pred))
    print(f'Accuracy for {filename}: {accuracy_score(y_test, y_pred)}')
    print(f'Recall for {filename}: {recall_score(y_test, y_pred)}')
    print(f'Precision for {filename}: {precision_score(y_test, y_pred)}')
    print(f'Balanced Accuracy for {filename}: {balanced_accuracy_score(y_test, y_pred)}')
    print(f'F1 for {filename}: {f1_score(y_test, y_pred, average="macro")}')
    """
    # Split the dataset into training set and test set with given ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize KNN classifier with given number of neighbors
    

    # Fit the classifier to the data
    start = time.process_time()
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    #cv_scores = cross_val_score(knn, X, y, cv=5)
    end = time.process_time()

    # Predict the test set results
    

    # Evaluate the classifier
    #print(f'Confusion Matrix for {filename}:')
    #print(confusion_matrix(y_test, y_pred))
    #print(f'Classification Report for {filename}:')
    #print(classification_report(y_test, y_pred))
    #print(f'Accuracy for {filename}: {accuracy_score(y_test, y_pred)}')
    #print(f'Recall for {filename}: {recall_score(y_test, y_pred)}')
    #print(f'Precision for {filename}: {precision_score(y_test, y_pred)}')
    #print(f'Balanced Accuracy for {filename}: {balanced_accuracy_score(y_test, y_pred)}')
    #print(f'F1 for {filename}: {f1_score(y_test, y_pred, average="macro")}')
    #print(f'Cross Validation Score for {filename}: {np.mean(cv_scores)}')
    #print(f'Execution time in seconds for {filename}: {end - start}')

    #Return Accuracy, then Execution time in seconds
    return end - start, balanced_accuracy_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred, average="macro")

np.set_printoptions(suppress=True)

top = 20
outputs = np.zeros((top + 1, 5))
for i in range(1, top + 1):
    frac = i * 0.05
    timed, acc, recall, f1 = evaluate_knn_with_data('1Y-qgStE3cO5OUKBl_k2AOQliVL5rUnzq', 'SMOTEsampled.csv', fraction=frac)
    
    outputs[i][0] = frac
    outputs[i][1] = timed
    outputs[i][2] = acc
    outputs[i][3] = recall
    outputs[i][4] = f1
    #outputs[i][5] = cv

labels = ["Fraction of Data", 'Time (Seconds)', 'Balanced Accuracy', 'Recall', 'F1']
outputs = np.array(outputs)

outDf = pd.DataFrame(outputs, columns=labels)
outDf.to_csv("KNN Fraction Data.csv")

#print(evaluate_knn_with_data('1Y-qgStE3cO5OUKBl_k2AOQliVL5rUnzq', 'SMOTEsampled.csv', fraction=0.0003))
# Evaluate for undersampled data
#evaluate_knn_with_data('1Y-qgStE3cO5OUKBl_k2AOQliVL5rUnzq', 'SMOTEsampled.csv', fraction=0.01)
#evaluate_knn_with_data('1YOjzIph-2yk3SZ9pLwHCAZISw5UYHI-X', 'undersampled.csv')

# Evaluate for SMOTE sampled data
#evaluate_knn_with_data('1Y-qgStE3cO5OUKBl_k2AOQliVL5rUnzq', 'SMOTEsampled.csv')
#evaluate_knn_with_data('1nCWQjAsoEB-tB7ZA2sNY3yC_ixxuq5dj', 'standardized.csv')
#evaluate_knn_with_data('1mDFTCo_MThaJNvbz3FzYTRZm3z35TXv7', 'creditcard.csv')
