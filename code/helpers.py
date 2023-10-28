import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

np.set_printoptions(suppress=True)
#I refuse to try to read scientific notation

def standardization(dataset):
    return StandardScaler().fit_transform(dataset)