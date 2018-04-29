# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:59:45 2018

@author: Andrew, Brian, Matthew
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM
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

if __name__ == '__main__':
    scaler = MinMaxScaler(feature_range=(0, 1))
    #load data
    # normalize the dataset    
    df = load_data("../Data/train.csv")
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
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainData, trainLabel, epochs=10, batch_size=1, verbose=2)
     
    # make predictions
    trainPredict = model.predict(trainData)
    testPredict = model.predict(testData)
    
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainLabel = scaler.inverse_transform([trainLabel])
    testPredict = scaler.inverse_transform(testPredict)
    testLabel = scaler.inverse_transform([testLabel])
    
    # calculate root mean squared error
    
    # shift train predictions for plotting
    
    # shift test predictions for plotting
    
    # plot baseline and predictions

