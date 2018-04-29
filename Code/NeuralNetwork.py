# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 18:18:52 2018

@author: Brian
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:59:45 2018
@author: Andrew, Brian, Matthew
"""
#from sklearn.cross_validation import StratifiedKFold
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, Flatten
from keras.optimizers import SGD, rmsprop, Adam
from keras.utils import np_utils
import numpy as np
import pandas as pd
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
    #pre-processing Titanic training data
    df = load_data("../Data/train.csv")
    trainData = df.as_matrix(columns=["Pclass", "Sex", "Age", "SibSp", "Fare", "Parch", "Q", "S"] )
    trainLabel = df.as_matrix(columns=["Survived"]).astype(float)
    
    #pdb.set_trace()
    model = Sequential()
    model.add(Dense(units=6, kernel_initializer='uniform', activation='relu',input_dim=8))
    model.add(Dropout(0.2))
    model.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=2, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
    
    passenger_cat = np_utils.to_categorical(trainLabel)
    model.fit(trainData, passenger_cat, shuffle=True, epochs=8, steps_per_epoch=891)
    
    #pre-processing Titanic test data
    df2 = load_data("../Data/test.csv")
    testData = df2.as_matrix(columns=["Pclass", "Sex", "Age", "SibSp", "Fare", "Parch", "Q", "S"] )
    
    predictions = model.predict_classes(testData)
    
    '''
    df3 = pd.read_csv("../Data/gender_submission.csv")
    testLabel = df3.as_matrix(columns=["Survived"]).astype(float)
    test_cat = np_utils.to_categorical(testLabel)
    
    #test model
    
    score = model.evaluate(testData, test_cat, verbose=0)
    print('Convolutional Neural Network Test loss:', score[0])
    print('Convolutional Neural Network Test accuracy:', score[1])
    '''