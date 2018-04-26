# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:31:58 2018

@author: Andrew, Brian, Matthew
"""
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD, rmsprop, Adam
from keras.utils import np_utils
import numpy as np
import pandas as pd
import pdb


K.tensorflow_backend._get_available_gpus()


def load_data(filepath):
    df = pd.read_csv(filepath)
    
    #replace genders with 0s and 1s
    df.replace("male", 1, inplace=True)
    df.replace("female", 0, inplace=True)
    
    df["Age"] = df[["Age", "Pclass"]].apply(impute_age, axis=1)
    
    #turn embarked into 0s and 1s
    embark = pd.get_dummies(df["Embarked"], drop_first=True)
    embark.head()
    df = pd.concat([df, embark], axis = 1)
    df.drop(["Embarked"], axis = 1, inplace = True)
    df.head()
    
    return df

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
    df = load_data("../Data/train.csv")
        
    batch_size = 200
    num_classes = 10
    epochs = 20
    
    trainData = df.as_matrix(columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Q", "S"] )
    trainLabel = df.as_matrix(columns=["Survived"]).astype(float)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    