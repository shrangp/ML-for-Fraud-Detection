import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import seaborn as sns

# Next, we read in the dataset and split it into a training and testing set:

df = pd.read_csv('SMOTEsampled.csv')
df = df.iloc[:280000, :]
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# We then train a RandomForestClassifier on our training data and labels:

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0)
rf.fit(X_train, y_train)

# We then obtain the predictions for our test data:

y_pred = rf.predict(X_test)

# We then build the confusion matrix for our results:

confusion_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Random Forest Classifier')
plt.show()



# We then calculate the accuracy, precision, recall, and F1 score for each confusion matrix:


# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = 2 * precision * recall / (precision + recall)
# print('Accuracy: {:.2f}'.format(accuracy))
# print('Precision: {:.2f}'.format(precision))
# print('Recall: {:.2f}'.format(recall))
# print('F1: {:.2f}'.format(f1))

# # We first create a list of the number of features we want to test:

num_features_list = [5, 10, 15, 20, 25, 30]

# # We then create a list to store the accuracy, precision, recall, and F1 score for each number of features:

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
best_accuracy = 0
best_precision = 0
best_recall = 0
best_f1 = 0
best_num_features = 0

# # We then loop through each number of features and train a model with that number of features:

for num_features in num_features_list:
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0, max_features=num_features)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_precision = precision
        best_recall = recall
        best_f1 = f1
        best_num_features = num_features

# # We then plot the results:

plt.plot(num_features_list, accuracy_list, label='Accuracy')
plt.plot(num_features_list, precision_list, label='Precision')
plt.plot(num_features_list, recall_list, label='Recall')
plt.plot(num_features_list, f1_list, label='F1')
plt.xlabel('Number of Features')
plt.ylabel('Score')
plt.title('Random Forest Scores for Different Number of Features')
plt.legend()
plt.show()

# # Finally, we plot the confusion matrix for the best performing model:

# rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0, max_features=best_num_features)
# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)
# confusion_matrix = confusion_matrix(y_test, y_pred)
# ConfusionMatrixDisplay(confusion_matrix).plot()
