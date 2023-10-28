import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

np.set_printoptions(suppress=True)
#I refuse to try to read scientific notation
#I don't print anything but even so

def standardization(dataset):
    """
    Standardizes an inputted dataset.
    Parameters: A dataset (NP, Panda, etc.)
    Returns: Dataset such that mean of all columns is 0 and variance is 1
    Requres: sklearn library
    """
    return StandardScaler().fit_transform(dataset)

def undersampling(dataset, labels):
    rus = RandomUnderSampler(random_state=0)
    newDataset, newLabels = rus.fit_resample(dataset, labels)

    return newDataset, newLabels