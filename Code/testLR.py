# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 20:57:29 2018

@author: Brian
"""

import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns # easy visualization

df_train = pd.read_csv('../Data/train.csv')
df_test = pd.read_csv('../Data/test.csv')
df_full = pd.concat([df_train, df_test], axis = 0, ignore_index=True)

passengerId_test = df_test['PassengerId']
df_full.head()

df_full.describe()
