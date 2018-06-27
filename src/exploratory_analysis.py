# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 22:24:11 2018

@author: kr
"""
import pandas as pd

# exploratory analysis of value_prediction datasets

df_train = pd.read_csv('../resource/train.csv')

# indentifying columns and rows that have zero values only
ind_zero_row = (df_train == 0).all(axis=1)
ind_zero_col = (df_train == 0).all(axis=0)

# cleaning columns that have zero values only 
df_train_col_clean = df_train.iloc[:, (~ind_zero_col).tolist()]
df_train_col_clean.iloc[:3,:5]

from sklearn import linear_model

clf = linear_model.ElasticNet(alpha = 0.1, l1_ratio = 0.7)
clf.fit(df_train_col_clean.iloc[:,2:], df_train['target'])

res = clf.predict(df_train_col_clean.iloc[:,2:])
#
#%matplotlib
#import matplotlib.pyplot as plt
#
#plt.spy(df_train_col_clean.iloc[:,2:])

# find duplicate columns
ind_dup = ~df_train_col_clean.iloc[:,2:].T.duplicated()

df_train_dup_col_del = df_train_col_clean.iloc[:,ind_dup.tolist()]
