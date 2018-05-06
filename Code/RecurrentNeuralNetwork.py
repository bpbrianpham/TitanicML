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

def load_data(filepath):
    df = pd.read_csv(filepath)
    
    #replace genders with 0s and 1s
    df.replace("male", 1, inplace=True)
    df.replace("female", 0, inplace=True)
    
    df["Age"] = df[["Age", "Pclass"]].apply(impute_age, axis=1)
    #df["Age"] = normalize(df["Age"])
    df["Fare"] = normalize(df["Fare"])
    #df["SibSp"] = normalize(df["SibSp"])
    df["Age"] = normalize(df["Age"])
    df.drop(["Name"], axis = 1, inplace = True)
    #df.drop(["Ticket"], axis = 1, inplace = True)
    #df.drop(["Cabin"], axis = 1, inplace = True)
    
    
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
    
    #load data
    # normalize the dataset    
    df = load_data("../Data/train.csv")
    
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #df = scaler.fit_transform(df)
    
    data = df.as_matrix(columns=["Pclass", "Sex", "Age", "SibSp", "Fare", "Parch", "Q", "S"] )
    label = df.as_matrix(columns=["Survived"]).astype(float)
    
    # split into train and test sets
    train_size = int(len(data) * 0.67)
    test_size = len(data) - train_size
    trainData, testData = data[0:train_size,:], data[train_size:len(data),:]
    
    train_size = int(len(label) * 0.67)
    test_size = len(label) - train_size
    trainLabel, testLabel = label[0:train_size,:], label[train_size:len(label),:]
    
    # reshape into X=t and Y=t+1
    look_back = 8    
    
    # reshape input to be [samples, time steps, features]
    trainData = np.reshape(trainData, (trainData.shape[0], 1, trainData.shape[1]))
    testData = np.reshape(testData, (testData.shape[0], 1, testData.shape[1]))
    
    # create and fit the LSTM network
    batch_size = 1
    model = Sequential()
    model.add(LSTM(32, batch_input_shape=(batch_size, 1, look_back), stateful=True, return_sequences=True))
    model.add(LSTM(32, batch_input_shape=(batch_size, 1, look_back), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(10):
        model.fit(trainData, trainLabel, epochs=1, batch_size=batch_size, verbose=2, shuffle=True)
        model.reset_states()
     
    # make predictions
    trainPredict = model.predict(trainData, batch_size=batch_size)
    model.reset_states()
    testPredict = model.predict(testData,  batch_size=batch_size)
    model.reset_states()
    

    
    #print accuracy
         
    trainPredict = survival_convert(trainPredict)
    testPredict = survival_convert(testPredict)
    
    print("Train Accuracy", accuracy(trainPredict, trainLabel))
    print("Test Accuracy", accuracy(testPredict, testLabel))
    
    
    #predict actual test
    #preprocess tester
    df2 =  load_data("../Data/test.csv")
    Testing_data = df2.as_matrix(columns=["Pclass", "Sex", "Age", "SibSp", "Fare", "Parch", "Q", "S"] )
    #Testing_label = df2.as_matrix(columns=["Survived"]).astype(float)
    Testing_data = np.reshape(Testing_data, (Testing_data.shape[0], 1, Testing_data.shape[1]))
    testing_data_predict = model.predict(Testing_data, batch_size=batch_size)

        
    # invert predictions
    
    '''
    trainPredict = scaler.inverse_transform(trainPredict)
    trainLabel = scaler.inverse_transform([trainLabel])
    testPredict = scaler.inverse_transform(testPredict)
    testLabel = scaler.inverse_transform([testLabel])
    '''
    
    # calculate root mean squared error
    
    # shift train predictions for plotting
    
    # shift test predictions for plotting
    
    # plot baseline and predictions

