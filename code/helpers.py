import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

np.set_printoptions(suppress=True)
#I refuse to try to read scientific notation

def standardization(dataset):
    return StandardScaler().fit_transform(dataset)

def undersampling(dataset, labels):
    rus = RandomUnderSampler(random_state=0)
    newDataset, newLabels = rus.fit_resample(dataset, labels)

    return newDataset, newLabels