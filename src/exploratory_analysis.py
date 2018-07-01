# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 22:24:11 2018

@author: kr
"""
import pandas as pd

# exploratory analysis of value_prediction datasets

df_train = pd.read_csv('../resource/train.csv')
df_test = pd.read_csv('../resource/small_test_data.csv')

class PrepareDataForModel:
    def __init__(self, data, index_col = None, target_col = None):
        self.data = data.copy()
        self.df_train = None
        self.df_test = None
        self._index = None
        self._col = None
        self.target_col = target_col
        if index_col is not None:
            self.index_col = index_col
            self.data.set_index(index_col, inplace = True)
        else:
            self.data.reset_index(drop = True, inplace = True)
        #if target_col is not None:
        self.target_col = target_col
        self.inp_id = self.ind_col(self.data, target_col)

    def clean_row(self, val):
        X = self.data.iloc[:,self.inp_id]
        if pd.isnull(val):
            ind = ~X.isnull().all(1)
            self.df_train = self.data[ind]
        else:
            ind = ~(X==val).all(1)
            self.df_train = self.data[ind]
        self._index = self.df_train.index
        self._col = self.ind_col(self.df_train, self.target_col)
        
    def clean_col(self, val):
        X = self.data.iloc[:,self.inp_id]
        X = X.T
        if pd.isnull(val):
            ind = ~X.isnull().all(1)
            self.df_train = self.data.iloc[:,ind.tolist()]
        else:
            ind = ~(X==val).all(1)
            self.df_train = self.data.iloc[:,ind.tolist()]
        self._col = self.ind_col(self.df_train, self.target_col)
        self._index = self.df_train.index
        
    def clean_test_data(self, data):
        if self.index_col in data.columns:
            data.set_index(self.index_col, inplace = True)
        else:
            data.reset_index(drop = True)
        self.df_test = data[self._col]

    @staticmethod
    def ind_col(data, target_col):
        return ~data.columns.str.contains(target_col)


# indentifying columns and rows that have zero values only
ind_zero_row = (df_train == 0).all(axis=1)
ind_zero_col = (df_train == 0).all(axis=0)

# cleaning columns that have zero values only 
df_train_col_clean = df_train.iloc[:, (~ind_zero_col).tolist()]
df_train_col_clean.iloc[:3,:5]

# find duplicate columns
ind_dup = ~df_train_col_clean.iloc[:,2:].T.duplicated()
df_train_dup_col_del = df_train_col_clean.iloc[:,ind_dup.tolist()]

from sklearn import linear_model

clf = linear_model.ElasticNet(alpha = 1, l1_ratio = 0.7)
clf.fit(df_train_dup_col_del.iloc[:,2:], df_train_dup_col_del['target'])

res = clf.predict(df_train_col_clean.iloc[:,2:])
clf.score(df_train_dup_col_del.iloc[:,2:],df_train_dup_col_del['target']) 
###############################################################################
# testing 2
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, SparsePCA
from sklearn import metrics
from sklearn.model_selection import train_test_split

dd = df_train_dup_col_del.copy()
df_train, df_test = train_test_split(dd, test_size = 0.30,random_state = 42) 

pca = SparsePCA(random_state=0)
#pca.n_components = 500
lm = linear_model.ElasticNet(alpha =1, l1_ratio = 0.7)
model = Pipeline(steps=[('pca',pca), ('lm',lm)])

model.fit(df_test.iloc[:,2:100], df_test['target'])
result = model.predict(df_train.iloc[:,2:])
model.score(df_train.iloc[:,2:],df_train['target']) 


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor(random_state=0)
clf.fit(df_test.iloc[:,2:], df_test.target)

res = clf.predict(df_test.iloc[:,2:])
clf.score(df_test.iloc[:,2:],df_test['target']) 


#cross_val_score(clf, df_train.iloc[:,2:], df_train.target, cv=10)
#%matplotlib
#import matplotlib.pyplot as plt
#
#plt.spy(df_train_col_clean.iloc[:,2:])

############################### Loading test data ############################
df_test_data = pd.read_csv('../resource/test.csv')

class CleaningData:
    def __init__(self, data, index_col = None, target_col = None):
        self.data = data
        if index_col is not None:
            self.index_col = index_col
            self.data.set_index(index_col)
        if target_col is not None:
            self.target_col = target_col
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    