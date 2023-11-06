import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from collections import Counter

np.set_printoptions(suppress=True)
#I refuse to try to read scientific notation
#I don't print anything but even so

def standardization(dataset):
    """
    Standardizes an inputted dataset.
    Parameters: A dataset without labels (NP, Panda, etc.)
    Returns: Dataset such that mean of all columns is 0 and variance is 1
    Requres: sklearn library
    """
    return StandardScaler().fit_transform(dataset)

def oversampling(dataset, labels, method='SMOTE'):
    oversample = 0
    if method == 'RandomOverSampler':
        oversample = RandomOverSampler(sampling_strategy='minority')
    if method == 'SMOTE':
        oversample = SMOTE()
    else:
        oversample = RandomOverSampler(sampling_strategy='minority')
    newDataset, newLabels = oversample.fit_resample(dataset, labels)
    return newDataset, newLabels

def undersampling(dataset, labels):
    """
    Undersampler using imbalanced-learn's Random Undersampler method

    Parameters: dataset with no labels, labels for the dataset
    Returns: Undersampled dataset, labels for new dataset
    Requires: imbalanced-learn
    """
    rus = RandomUnderSampler(random_state=0)
    newDataset, newLabels = rus.fit_resample(dataset, labels)

    return newDataset, newLabels

def uniques(dataset, labels):
    """
    Calculates how many values in dataset are unique
    Mostly used for validation that repeats have been removed
    """
    Xy = list(set(list(zip([tuple(x) for x in dataset], labels))))
    X = [list(l[0]) for l in Xy]
    y = [l[1] for l in Xy]
    return X, y

def toCSV(dataset, labels, name):
    """
    Documentation
    """
    vals = range(1, 29)
    dataLabels = ['Time']
    for i in vals:
        dataLabels.append('V' + str(i))
    dataLabels.append('Amount')
    dataLabels.append('Class')

    labels = np.reshape(labels, (np.shape(dataset)[0], 1))

    standardizedFull = np.concatenate((dataset, labels), axis=1)

    df = pd.DataFrame(standardizedFull, columns = dataLabels)
    df.to_csv(str(name) + ".csv")