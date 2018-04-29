# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:59:45 2018
@author: Andrew, Brian, Matthew
"""
#from sklearn.cross_validation import StratifiedKFold
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pdb

import matplotlib.pyplot as plt

def load_data(filepath):
    df = pd.read_csv(filepath)
    
    #replace genders with 0s and 1s
    df.replace("male", 1, inplace=True)
    df.replace("female", 0, inplace=True)
    
    df["Age"] = df[["Age", "Pclass"]].apply(impute_age, axis=1)
    #df["Age"] = normalize(df["Age"])
    df["Fare"] = normalize(df["Fare"])
    #df["SibSp"] = normalize(df["SibSp"])
    
    #turn embarked into 0s and 1s
    embark = pd.get_dummies(df["Embarked"], drop_first=True)
    embark.head()
    df = pd.concat([df, embark], axis = 1)
    df.drop(["Embarked"], axis = 1, inplace = True)
    df.head()
    
    #normalizing the dataset
    scalar = MinMaxScaler(feature_range=(0,1))
    df = scalar.fit_transform(df)
        
    return df

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 39.159930
        elif Pclass == 2:
            return 29.506705
        else:
            return 24.816367
    else:
        return Age


if __name__ == '__main__':
    #fix random seed for reproducibility
    np.random.seed(11)
    
    #pre-processing Titanic training data
    df = load_data("../Data/train.csv")
    data = df.as_matrix(columns=["Pclass", "Sex", "Age", "SibSp", "Fare", "Parch", "Q", "S"] )
    label = df.as_matrix(columns=["Survived"]).astype(float)
    
    '''
    #pre-processing Titanic test data
    df2 = load_data("../Data/test.csv")
    testData = df2.as_matrix(columns=["Pclass", "Sex", "Age", "SibSp", "Fare", "Parch", "Q", "S"] )
    '''
    
    train_size = int(len(data) * 0.75)
    test_size = len(data) - train_size
    trainData, testData = data[0:train_size,:], data[train_size:len(data),:]
    print(len(trainData), len(testData))
    
    
    
    
    
    
    
    
    
    