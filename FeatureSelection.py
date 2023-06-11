#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 19:00:04 2021

@author: pranjalgarg
"""
import pandas as pd
import numpy as np
from scipy import stats
import statistics
import matplotlib.pyplot as plt
import torch
import seaborn as sns


from sklearn import linear_model,model_selection
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.linear_model import LinearRegression,Ridge,LogisticRegression,RidgeClassifier
from sklearn import tree

# Import your necessary dependencies
from sklearn.feature_selection import RFE

from toolbox_02450 import rlr_validate, train_neural_net, draw_neural_net


from sklearn.linear_model import Ridge

#Regression code


data = pd.read_csv("CHD.csv")
data2=pd.read_csv("CHD.csv")


#Fix data by: replacing family history values with binary int and delete useless row
data.famhist = data.famhist.str.replace('Present','1') 
data.famhist = data.famhist.str.replace('Absent','0')
data.famhist = data.famhist.astype(int)

#delete useless row
del data['row.names']




#Converting data into a matrix and getting classNames, attribute names, the shape of the matrix etc 
raw_data = data.values  

cols = range(0, 10) 
X = raw_data[:, cols]
attributeNames = np.asarray(data.columns[cols])
y = X[:,3]

famhistory=np.reshape(X[:,4], (-1, 1))
chdcolumn= np.reshape(X[:,9], (-1, 1))
#Xtemp= np.hstack((X[:,:3],X[:,5:]))

#One-out-of K encoding


label_encoder_famhist = LabelEncoder()
integer_encoded_famhist = label_encoder_famhist.fit_transform(famhistory)
onehot_encoder_famhist = OneHotEncoder(sparse=False)
integer_encoded_famhist = integer_encoded_famhist.reshape(len(integer_encoded_famhist), 1)
onehot_encoded_famhist = onehot_encoder_famhist.fit_transform(integer_encoded_famhist)
#print(onehot_encoded)\
    


label_encoder_chd = LabelEncoder()
integer_encoded_chd = label_encoder_chd.fit_transform(chdcolumn)
onehot_encoder_chd = OneHotEncoder(sparse=False)
integer_encoded_chd = integer_encoded_chd.reshape(len(integer_encoded_chd), 1)
onehot_encoded_chd = onehot_encoder_chd.fit_transform(integer_encoded_chd)


#X= np.hstack((X[:,:3],X[:,4:]))

C = 2
Xtempregression= np.hstack((X[:,:3],X[:,5:9]))
Xtempclassification= np.hstack((X[:,:4],X[:,5:9]))
yclassification = X[:,9]


# Normalize data

Xtempregression = stats.zscore(Xtempregression)
Xregression= np.hstack((Xtempregression[:,:3],famhistory,Xtempregression[:,3:],chdcolumn))


#Normalize classification data
Xtempclassification = stats.zscore(Xtempclassification)
Xclassification= np.hstack((Xtempclassification[:,:4],famhistory,Xtempclassification[:,4:]))

#X = stats.zscore(X)
#Xclassification = stats.zscore(Xclassification)

Xregression= np.hstack((Xtempregression[:,:3],famhistory,Xtempregression[:,3:],chdcolumn))


#Limited attributes for regression

Xregression= np.hstack((Xregression[:,:3],Xregression[:,7:9]))






N, M = Xregression.shape


## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)


GenError = np.empty(2)

k=0
i=0


model_linear_classification = linear_model.Ridge(alpha=10)
model_linear_classification.fit(Xclassification, yclassification)
y_est_Classification = model_linear_classification.predict(Xclassification)
plt.figure()
plt.title('Predicted value of CHD against actual values')
plt.scatter(yclassification, y_est_Classification, marker='o');
plt.xlabel("CHD True");
plt.ylabel("CHD predicted ");   
plt.show()

plt.figure()
plt.scatter(yclassification, y_est_Classification, alpha=0.1,label='Decision Boundary')
plt.xlabel("CHD True");
plt.ylabel("CHD predicted ");  
plt.show()

modelgraph= linear_model.Ridge(alpha=10)
modelgraph.fit(Xregression , y)
y_graph_est = modelgraph.predict(Xregression)

plt.figure()
plt.title('Predicted value of adiposity against actual values')
plt.scatter(y, y_graph_est, marker='o');
plt.xlabel("Adiposity True");
plt.ylabel("Adiposity predicted ");   
plt.show() 
