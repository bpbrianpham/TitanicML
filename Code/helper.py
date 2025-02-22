import csv
import tensorflow as tf
import numpy as np
import pandas as pd

'''
"../Data/train.csv"
'''

def load_features(d):
    df = pd.read_csv(d, usecols=(2,4,5,6,7,9))
    df2 = pd.read_csv(d, usecols=(1))
    data = df.as_matrix()
    label = df2.as_matrix()
    return np.matrix(data), np.matrix(label)
    