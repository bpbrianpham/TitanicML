# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:59:45 2018

@author: Andrew, Brian, Matthew
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import pandas as pd
import pdb

def load_data(filepath):
    df = pd.read_csv(filepath)
    
    #replace genders with 0s and 1s
    df.replace("male", 1, inplace=True)
    df.replace("female", 0, inplace=True)
    
    df["Age"] = df[["Age", "Pclass"]].apply(impute_age, axis=1)
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

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def survival_convert(predict):
    for i in range(len(predict)):
        if predict[i] > 0.5:
            predict[i] = 1
        else:
            predict[i] = 0
    return predict

def accuracy(predict, label):
    if len(predict) == len(label):    
        correct = 0
        for i in range(len(predict)):
            if predict[i] == label[i]:
                correct = correct + 1        
        percent_accurate = correct / len(predict)
        return percent_accurate
    else:
        raise ValueError("Incorrect input: input shapes do not fit.")

if __name__ == '__main__':
    #pdb.set_trace()
    #load data
    # normalize the dataset    
    df = load_data("../Data/train.csv")
    
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #df = scaler.fit_transform(df)
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
    
    # reshape into X=t and Y=t+1
    look_back = 14    
    
    # reshape input to be [samples, time steps, features]
    trainData = np.reshape(trainData, (trainData.shape[0], 1, trainData.shape[1]))
    testData = np.reshape(testData, (testData.shape[0], 1, testData.shape[1]))
    
    # create and fit the LSTM network
    batch_size = 1
    model = Sequential()
    model.add(LSTM(14, batch_input_shape=(batch_size, 1, look_back), stateful=True, return_sequences=True))
    model.add(LSTM(14, batch_input_shape=(batch_size, 1, look_back), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(15):
        model.fit(trainData, trainLabel, epochs=1, batch_size=batch_size, verbose=2, shuffle=True)
        model.reset_states()
     
    # make predictions
    trainPredict = model.predict(trainData, batch_size=batch_size)
    model.reset_states()
    testPredict = model.predict(testData,  batch_size=batch_size)
    
    #print accuracy
         
    trainNewPredict = survival_convert(trainPredict)
    testNewPredict = survival_convert(testPredict)
    
    print("Train Accuracy", accuracy(trainNewPredict, trainLabel))
    print("Test Accuracy", accuracy(testNewPredict, testLabel))

    #preprocess kaggle test data
    df2 =  load_data("../Data/test.csv")
    df2.drop(["Name", "PassengerId","Ticket"], axis = 1, inplace=True)
    Testing_data = df2.as_matrix()
    Testing_data = np.reshape(Testing_data, (Testing_data.shape[0], 1, Testing_data.shape[1]))
    
    #predict kaggle test data
    model.reset_states()
    testing_data_predict = model.predict(Testing_data, batch_size=batch_size)
    
    model.summary()
        
   
    #Jack and Rose Predictions
    JackRose = load_data("../Data/titanic.csv")
    
    JRlabel = JackRose.as_matrix(columns=["Survived"]).astype(float)
    JackRose.drop(["Name", "PassengerId","Ticket","Survived"], axis = 1, inplace=True)
    JRData = JackRose.as_matrix()
    JRData = np.reshape(JRData, (JRData.shape[0], 1, JRData.shape[1]))
    JRPredict = model.predict(JRData, batch_size=batch_size)
