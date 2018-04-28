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
    df["Age"] = normalize(df["Age"])
    #df["Fare"] = normalize(df["Fare"])
    df["SibSp"] = normalize(df["SibSp"])
    df['Cabin_letter']=df['Cabin'].str[:1]
    df['Cabin_letter A'] = np.where(df['Cabin_letter']=='A',1,0)
    df['Cabin_letter B'] = np.where(df['Cabin_letter']=='B',1,0)
    df['Cabin_letter C'] = np.where(df['Cabin_letter']=='C',1,0)
    df['Cabin_letter D'] = np.where(df['Cabin_letter']=='D',1,0)
    df['Cabin_letter E'] = np.where(df['Cabin_letter']=='E',1,0)
    df['Cabin_letter noCabin'] = np.where(df['Cabin_letter'].isnull(),1,0)
    #Cabin = pd.get_dummies(df["Cabin"], drop_first=True)
    #Cabin_letter = pd.get_dummies(df["Cabin_letter"], drop_first=True)
    #pdb.set_trace()
    
    #turn embarked into 0s and 1s
    embark = pd.get_dummies(df['Embarked'],prefix='Embarked ',drop_first=True)
    #cabin = pd.get_dummies(df['Cabin'],prefix='cabin ',drop_first=True)
    #cabin_letter= pd.get_dummies(df['Cabin_letter'],prefix='cabin_letter ',drop_first=True)
    
    df.drop(["Embarked"], axis = 1, inplace = True)
    df.drop(["Cabin"], axis = 1, inplace = True)
    df.drop(["Cabin_letter"], axis = 1, inplace = True)
    #pdb.set_trace()
    datas=[df,embark]
    df=pd.concat(datas, axis=1)
    #embark.head()
    #df = embark
    
    #df.head()
    
    '''Cabin.head()
    df = pd.concat([df, Cabin], axis = 1)
    df.drop(["Cabin"], axis = 1, inplace = True)
    df.head()
    
    Cabin_letter.head()
    df = pd.concat([df, Cabin_letter], axis = 1)
    df.head()
    '''
   # df = df[~df.index.duplicated()]
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
    #pdb.set_trace()
    df = load_data("../Data/train.csv")
    
    
    
    trainLabel = df.as_matrix(columns=["Survived"]).astype(float)
    df.drop(["Name", "PassengerId","Fare","Ticket","Survived"], axis = 1, inplace=True)
    trainData = df.as_matrix()
    '''
    logreg=LogisticRegression()
    logreg.fit(trainData,trainLabel)
    
    df2 = load_data("../Data/test.csv")
    testData = df2.as_matrix(columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Q", "S"] )
    #pdb.set_trace()
    
    df3 = pd.read_csv("../Data/gender_submission.csv")
    testLabel = df3.as_matrix(columns=["Survived"]).astype(float)
    test_cat = np_utils.to_categorical(testLabel)
    
    prediction=logreg.predict(testData)
    
    score=logreg.score(testData,test_cat)
    print(score)
    '''
    
    
    
    model = Sequential()
    #model.add(Dense(1, input_shape=(8,)))
    #model.add(Dropout(0.25))
    #model.add(Dense(2))
    model.add(Dense(2, activation="softmax", input_shape=(13,)))
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
    df2.drop(["Name", "PassengerId","Fare","Ticket"], axis = 1, inplace=True)
    testData = df2.as_matrix()
    
    df3 = pd.read_csv("../Data/gender_submission.csv")
    testLabel = df3.as_matrix(columns=["Survived"]).astype(float)
    test_cat = np_utils.to_categorical(testLabel)
    
    score = model.evaluate(testData, test_cat, verbose=0)
    print('Logistic Model Test loss:', score[0])
    print('Logistic Model Test accuracy:', score[1])
    
    #SVG(model_to_dot(model).create(prog='dot', format='svg'))
    