import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


# Load the dataset into a pandas DataFrame
data = pd.read_csv('undersampled_use.csv')

# Separate features and labels
X = data.iloc[:, 2:-2]  # Features: PCA features (columns 3 to 30)
y = data.iloc[:, -1]  # Labels: Class (column 31)

# Define the parameter grids for the grid search
min_samples = list(range(2, 52, 5))
eps = list(range(4, 28, 3))

# Visualize the f1 scores for different values of min_samples and eps
f1_scores = np.zeros((len(min_samples), len(eps)))
for i in range(len(min_samples)):
    for j in range(len(eps)):
        dbscan = DBSCAN(eps=eps[j], min_samples=min_samples[i])
        clusters = dbscan.fit_predict(X)
        count = 0
        while True:
            if clusters[count] != -1 and clusters[count] != 0:
                clusters[count] = 1
            count += 1
            if count == len(clusters):
                break
        predicted_fraud = clusters == 1
        f1_scores[i][j] = f1_score(y, predicted_fraud)
        print("min_samples: ", min_samples[i], "eps: ", eps[j], "f1_score: ", f1_scores[i][j])

# print the best f1 score and the corresponding min_samples and eps
max_f1_score = np.max(f1_scores)
print("max_f1_score: ", max_f1_score)
max_f1_score_index = np.where(f1_scores == max_f1_score)
print("min_samples: ", min_samples[max_f1_score_index[0][0]])
print("eps: ", eps[max_f1_score_index[1][0]])

# Plot the f1 scores so it is easier to visualize
plt.figure(figsize=(12, 8))
plt.imshow(f1_scores, cmap='hot', interpolation='nearest')
plt.xticks(np.arange(len(eps)), eps)
plt.yticks(np.arange(len(min_samples)), min_samples)
plt.xlabel('eps')
plt.ylabel('min_samples')
plt.colorbar()
plt.show()




