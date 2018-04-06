import csv
import tensorflow as tf
import numpy as np
import pandas as pd

       
df = pd.read_csv("../Data/train.csv", usecols=(0,1,2,4,5,6,7,9))
print(df)
print(type(df))
Titanic_Training_Data = df.as_matrix()
print(Titanic_Training_Data)

