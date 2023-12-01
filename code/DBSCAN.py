# Start of code for DBSCAN.py on undersampled_use.csv dataset

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset into a pandas DataFrame
data = pd.read_csv('undersampled_use.csv')

# Separate features and labels
X = data.iloc[:, 2:-2]  # Features: PCA features (columns 3 to 30)
y = data.iloc[:, -1]  # Labels: Class (column 31)

dbscan = DBSCAN(eps=7, min_samples=2)
clusters = dbscan.fit_predict(X)
# count = 0
# while True:
#     if clusters[count] != -1 and clusters[count] != 0:
#         clusters[count] = 1
#     count += 1
#     if count == len(clusters):
#         break

predicted_fraud = clusters == 1
actual_fraud = y == 1
# count = 0
# x = []
# for i, j in zip(predicted_fraud, actual_fraud):
#     if i != j:
#         x.append(count)
#     count += 1
# for i in x:
#     print(actual_fraud[i], predicted_fraud[i], clusters[i])
confusion_matrix_fraud = confusion_matrix(actual_fraud, predicted_fraud)
print("True Negatives: ", confusion_matrix_fraud[0][0])
print("False Positives: ", confusion_matrix_fraud[0][1])
print("False Negatives: ", confusion_matrix_fraud[1][0])
print("True Positives: ", confusion_matrix_fraud[1][1])
# visualize the confusion matrix

# sns.heatmap(confusion_matrix_fraud, annot=True, cmap='Blues', fmt='g')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix for Random Forest Classifier')
# plt.show()

# print acccuracy, precision, balanced accuracy, recall, and f1 score
tn = confusion_matrix_fraud[0][0]
fp = confusion_matrix_fraud[0][1]
fn = confusion_matrix_fraud[1][0]
tp = confusion_matrix_fraud[1][1]
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
balanced_accuracy = (tp / (tp + fn) + tn / (tn + fp)) / 2
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Balanced Accuracy: ", balanced_accuracy)
print("Recall: ", recall)
print("F1 Score: ", f1_score)

clusters = list(clusters)
plt.figure(figsize=(12, 8))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='plasma')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.title('DBSCAN Clusters')
plt.show()


# calculate silhouette coefficient
from sklearn.metrics import silhouette_score

silhouette_coefficient = silhouette_score(X, clusters)
print("Silhouette Coefficient: ", silhouette_coefficient)

# calculate adjusted rand index
from sklearn.metrics import adjusted_rand_score

adjusted_rand_index = adjusted_rand_score(actual_fraud, predicted_fraud)
print("Adjusted Rand Index: ", adjusted_rand_index)



