import tensorflow as tf
import numpy as np
import pandas as pd

def read_data(datapath):
    df = pd.read_csv(datapath)
    return df
'''
#data = np.loadtxt(open("../Data/train.csv", "rb"), delimiter=",", skiprows=1)

def load_data(dataloc):
	data = np.loadtxt(dataloc, unpack='true')
	#data = np.transpose(data, (1,0))
	return data	

data = load_data("../Data/train.csv")
#print(data)
'''
