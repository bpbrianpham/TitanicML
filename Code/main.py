# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:59:45 2018

@author: Matthew
"""
from helper import *

def logistic_regression(data, label, max_iter, learning_rate):
#	The logistic regression classifier function.
#
#	Args:
#	data: train data with shape (1561, 3), which means 1561 samples and 
#		  each sample has 3 features.(1, symmetry, average internsity)
#	label: train data's label with shape (1561,1). 
#		   1 for digit number 1 and -1 for digit number 5.
#	max_iter: max iteration numbers
#	learning_rate: learning rate for weight update
#	
#	Returns:
#		w: the seperater with shape (3, 1). You must initilize it with w = np.zeros((d,1))
    feature_count = data.shape[1];
    w = np.zeros(feature_count);
    gradient = 0;
    
    for i in range(max_iter):
        for j in range(len(label)):
            gradient += (data[j]*label[j])/(1 + np.exp(label[j]*np.transpose(w)*data[j]));
        gradient = -gradient/len(label);
        w = w - learning_rate*gradient;
    return w;

def accuracy(x, y, w):
    
#   This function is used to compute accuracy of a logsitic regression model.
#    
#    Args:
#    x: input data with shape (n, d), where n represents total data samples and d represents
#        total feature numbers of a certain data sample.
#    y: corresponding label of x with shape(n, 1), where n represents total data samples.
#    w: the seperator learnt from logistic regression function with shape (d, 1),
#        where d represents total feature numbers of a certain data sample.
#
#    Return 
#        accuracy: total percents of correctly classified samples. Set the threshold as 0.5,
#        which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
#    print(x);
    
    accuracy = 0;
    correctlyClassified = 0;
    
    #uses sigmoid function to predict outcome of data, then compare to labels
    for i in range(len(y)):
        prediction = 1 / (1 + np.exp(-1 * y[i] * np.dot(np.transpose(w), x[i])));
        
        #classify based on threshold
        if (prediction > 0.50):
            classification = 1;
        else:
            classification = -1;
        
        #if labeled correctly, +1
        if (classification == y[i]):
            correctlyClassified += 1;
    
    accuracy = correctlyClassified/len(y)*100;
    return accuracy;

#Use for testing the training and testing processes of a model
def train_test_a_model(modelname, train_data, train_label, test_data, test_label, max_iter, learning_rate):
    print(modelname+" testing...")
    # max iteration test cases 
    for i, m_iter in enumerate(max_iter):
        w = logistic_regression(train_data, train_label, m_iter, learning_rate[1])
        Ain, Aout = accuracy(train_data, train_label, w), accuracy(test_data, test_label, w)
        print("max iteration testcase%d: Train accuracy: %f, Test accuracy: %f"%(i, Ain, Aout))
    # learning rate test cases
    for i, l_rate in enumerate(learning_rate):
        w = logistic_regression(train_data, train_label, max_iter[3], l_rate)
        Ain, Aout = accuracy(train_data, train_label, w), accuracy(test_data, test_label, w)
        print("learning rate testcase%d: Train accuracy: %f, Test accuracy: %f"%(i, Ain, Aout))
    print(modelname+" test done.")
    
def test_logistic_regression():
    max_iter = [100, 200, 500,1000]
    learning_rate = [0.1, 0.2, 0.5]
    traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
    train_data,train_label = load_features(traindataloc)
    test_data, test_label = load_features(testdataloc)
    train_test_a_model("logistic regression", train_data, train_label, test_data, test_label, max_iter, learning_rate)
    try:
        train_test_a_model("logistic regression", train_data, train_label, test_data, test_label, max_iter, learning_rate)
    except:
        print("Please finish logistic_regression() and cross_entropy_error() functions \n\
				before you run the test_logistic_regression() function.\n")

if __name__ == '__main__':
    trainData = read_data("../Data/train.csv")
    print(trainData['PassengerID'])