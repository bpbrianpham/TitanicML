
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
from keras.layers import Dense, Dropout
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
    
    df["Age"] = df[["Age", "Pclass"]].apply(fill_age, axis=1)
    df["Age"] = normalize(df["Age"])
    df["Fare"].fillna(df["Fare"].mean(), inplace =True)
    df["Fare"] = normalize(df["Fare"])
    df["Embarked"].fillna(df["Embarked"].mode(), inplace=True)
    
    #create categories for each cabin letter
    df['Cabin_letter']=df['Cabin'].str[:1]
    df['Cabin_letter A'] = np.where(df['Cabin_letter']=='A',1,0)
    df['Cabin_letter B'] = np.where(df['Cabin_letter']=='B',1,0)
    df['Cabin_letter C'] = np.where(df['Cabin_letter']=='C',1,0)
    df['Cabin_letter D'] = np.where(df['Cabin_letter']=='D',1,0)
    df['Cabin_letter E'] = np.where(df['Cabin_letter']=='E',1,0)
    df['Cabin_letter noCabin'] = np.where(df['Cabin_letter'].isnull(),1,0)
    
    #turn embarked into 0s and 1s
    embark = pd.get_dummies(df['Embarked'],prefix='Embarked ',drop_first=True)
    
    df.drop(["Embarked"], axis = 1, inplace = True)
    df.drop(["Cabin"], axis = 1, inplace = True)
    df.drop(["Cabin_letter"], axis = 1, inplace = True)
    datas=[df,embark]
    df=pd.concat(datas, axis=1)
    
    return df

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def fill_age(cols):
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
    label = df.as_matrix(columns=["Survived"]).astype(float)
    df.drop(["Name", "PassengerId","Ticket","Survived"], axis = 1, inplace=True)
    data = df.as_matrix()
    
    # split into train and test sets
    train_size = int(len(data) * 0.75)
    test_size = len(data) - train_size
    trainData, testData = data[0:train_size,:], data[train_size:len(data),:]
    
    train_size = int(len(label) * 0.75)
    test_size = len(label) - train_size
    trainLabel, testLabel = label[0:train_size,:], label[train_size:len(label),:]
    
    #pdb.set_trace()
    model = Sequential()
    model.add(Dense(units=14, kernel_initializer='uniform', activation='relu',input_dim=14))
    model.add(Dropout(0.1))
    model.add(Dense(units=14, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=2, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=["accuracy"])
    
    passenger_cat = np_utils.to_categorical(trainLabel)
    model.fit(trainData, passenger_cat, shuffle=True, epochs=8, steps_per_epoch=train_size)
    
    test_cat = np_utils.to_categorical(testLabel)
    score = model.evaluate(testData, test_cat, verbose=0)
    print('Neural Network Test loss:', score[0])
    print('Neural Network Test accuracy:', score[1])