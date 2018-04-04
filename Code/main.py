# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 17:42:13 2018

@author: Brian
"""

import csv
import tensorflow as tf
import numpy as np
import pandas as pd

'''
with open("../Data/train.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        print(' '.join(row))
        
'''        
df = pd.read_csv("../Data/train.csv")
print(df)
'''
#data = np.loadtxt(open("../Data/train.csv", "rb"), delimiter=",", skiprows=1)

def load_data(dataloc):
	data = np.loadtxt(dataloc, unpack='true')
	#data = np.transpose(data, (1,0))
	return data	

data = load_data("../Data/train.csv")
#print(data)
'''
