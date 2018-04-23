# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:59:45 2018

@author: Andrew, Brian, Matthew
"""
#from sklearn.cross_validation import StratifiedKFold
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import pandas as pd

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
    df = load_data("../data/train.csv")
    
#    df.drop(["Name", "PassengerId"], axis = 1, inplace=True)
    
    trainData = df.as_matrix(columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Q", "S"] )
    trainLabel = df.as_matrix(columns=["Survived"]).astype(float)
    
    model = Sequential()
    model.add(Dense(2, input_shape=(8,)))
    model.add(Activation("softmax"))
    sgd = SGD()
    
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    passenger_cat = np_utils.to_categorical(trainLabel)
    model.fit(trainData, passenger_cat, epochs=20)

    df2 = load_data("../data/test.csv")
    testData = df2.as_matrix(columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Q", "S"] )
    
    df3 = pd.read_csv("../data/gender_submission.csv")
    testLabel = df3.as_matrix(columns=["Survived"]).astype(float)
    test_cat = np_utils.to_categorical(testLabel)
    
    score = model.evaluate(testData, test_cat, verbose=0)
    print('Logistic Model Test loss:', score[0])
    print('Logistic Model Test accuracy:', score[1])