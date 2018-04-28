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
from keras.optimizers import SGD, rmsprop, Adam
from keras.utils import np_utils
import numpy as np
import pandas as pd
import pdb
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def load_data(filepath):
    df = pd.read_csv(filepath)
    
    #replace genders with 0s and 1s
    df.replace("male", 1, inplace=True)
    df.replace("female", 0, inplace=True)
    
    df["Age"] = df[["Age", "Pclass"]].apply(impute_age, axis=1)
    #df["Age"] = normalize(df["Age"])
    #df["Fare"] = normalize(df["Fare"])
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
    
    df = load_data("../Data/train.csv")
    
#    df.drop(["Name", "PassengerId"], axis = 1, inplace=True)
    
    trainData = df.as_matrix(columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Q", "S"] )
    trainLabel = df.as_matrix(columns=["Survived"]).astype(float)  
    
    #pdb.set_trace()
    model = Sequential()
    #model.add(Dense(1, input_shape=(8,)))
    #model.add(Dropout(0.25))
    #model.add(Dense(2))
    model.add(Dense(2, activation="softmax", input_shape=(7,)))
    #model.summary()
    #opt = SGD()
    opt = Adam()
    #opt = rmsprop(lr=0.0001, decay=1e-4)
    
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    passenger_cat = np_utils.to_categorical(trainLabel)
    history = model.fit(trainData, passenger_cat, shuffle=True, epochs=8, steps_per_epoch=891)
    
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    '''plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.title('Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    '''
    df2 = load_data("../Data/test.csv")
    testData = df2.as_matrix(columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Q", "S"] )
    
    df3 = pd.read_csv("../Data/gender_submission.csv")
    testLabel = df3.as_matrix(columns=["Survived"]).astype(float)
    test_cat = np_utils.to_categorical(testLabel)
    
    score = model.evaluate(testData, test_cat, verbose=0)
    print('Logistic Model Test loss:', score[0])
    print('Logistic Model Test accuracy:', score[1])
    
    #SVG(model_to_dot(model).create(prog='dot', format='svg'))
    