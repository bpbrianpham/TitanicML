# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:59:45 2018

@author: Andrew, Brian, Matthew
"""
import numpy as np
import tensorflow as tf
import pandas as pd

feature_sets_train = pd.read_csv('../Data/train.csv')
# TODO: Use both datasets to make the embeddings (vocab_to_int map)
feature_sets_test = pd.read_csv('../Data/test.csv')
feature_sets_train_tests = pd.concat([feature_sets_train, feature_sets_test])
feature_sets = feature_sets_train

passengers = [' '.join(map(str,passenger[[2,3,4,5,8,9,10,11]])) for passenger in feature_sets.values]
passengers_test = [' '.join(map(str,passenger[[1,2,3,4,7,8,9,10]])) for passenger in feature_sets_test.values]

survived = [passenger[1] for passenger in feature_sets.values]
feature_sets = passengers
feature_sets_test = passengers_test
labels =  survived

#from string import punctuation
#all_text = ''.join([c for c in feature_sets if c not in punctuation])
#feature_sets = all_text.split(',')

passengers = [' '.join(map(str,passenger[[0,1,2,3,4,5,7,8,9,11]])) for passenger in feature_sets_train_tests.values]

all_text = ' '.join(passengers)
words = all_text.split()



