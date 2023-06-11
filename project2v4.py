import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.linalg as linalg
from scipy.linalg import svd
import statistics
from scipy import stats
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from toolbox_02450 import train_neural_net, draw_neural_net
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import model_selection, tree
from numpy import array
from numpy import argmax
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error, r2_score

from toolbox_02450 import rlr_validate

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)

from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("CHD.csv")

#Fix data by: replacing family history values with binary int and delete useless row
data.famhist = data.famhist.str.replace('Present','1') 
data.famhist = data.famhist.str.replace('Absent','0')
data.famhist = data.famhist.astype(int)

#delete useless row
del data['row.names']


#Converting data into a matrix and getting classNames, attribute names, the shape of the matrix etc 
raw_data = data.values  
cols = range(0, 9) 
X = raw_data[:, cols]
attributeNames = np.asarray(data.columns[cols])
y = raw_data[:,-1]
classNames = np.unique(y)
classDict = dict(zip(classNames,range(len(classNames))))
N, M = X.shape
C = len(classNames)


#one out of k encoding for family history

values = array(data.loc[:,['famhist']].values)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
#one out of k encoding for family history and CHD

values2 = array(y)

label_encoder2 = LabelEncoder()
integer_encoded2 = label_encoder2.fit_transform(values2)
onehot_encoder2 = OneHotEncoder(sparse=False)
integer_encoded2 = integer_encoded2.reshape(len(integer_encoded2), 1)
onehot_encoded2 = onehot_encoder2.fit_transform(integer_encoded2)
    
Xtransformed= np.hstack((X[:,:4], onehot_encoded, X[:,5:]))
Ytransformed= onehot_encoded2
attributeNamesNew = np.hstack((attributeNames[:4], "FamHistPresent", "FamHistAbsent", attributeNames[5:]))
 
# Xtransformed= X
# Ytransformed= y

Xstandardized = StandardScaler().fit_transform(Xtransformed)
Ystandardized= StandardScaler().fit_transform(Ytransformed)

#prepare input and output for regression - adiposity
X_adiposity_regression= np.hstack((Xstandardized[:,:3], Ystandardized))
Y_adiposity_regression= Xstandardized[:,3]


## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV_outer = model_selection.KFold(K, shuffle=True)
cv_inner = model_selection.KFold(5, shuffle=True)


k=1

#Artifiical Neural network
# Parameters for neural network classifier
params_ANN =  [1,2,3,4]
params_lambda = np.power(10.,range(-5,9))
n_hidden_units = 2      # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000
M=5
errors = [] # make a list for storing generalizaition error in each loop
# print('x size: ' + str(y_train.size))

#Y_adiposity_ANN = np.expand_dims(Y_adiposity_regression, axis=1)

#double layer cross validation: https://towardsdatascience.com/nested-cross-validation-hyperparameter-optimization-and-model-selection-5885d84acda
# X_train, X_test, y_train, y_test = train_test_split(X_adiposity_regression, Y_adiposity_regression, random_state=101, test_size = 0.2)
X_train = X_adiposity_regression
y_train = Y_adiposity_regression

for (train_idx_outer, test_idx_outer) in CV_outer.split(X_train):
    
    # # extract training and test set for current CV fold
    # train_data, val_data = X_train[train_idx], X_train[val_idx]
    # train_target, val_target = y_train[train_idx], y_train[val_idx]
    X_train_outer, X_test_outer = X_train[train_idx_outer], X[test_idx_outer]
    y_train_outer, y_test_outer = y_train[train_idx_outer], y_train[test_idx_outer]

    # baseline_mean= statistics.mean(y_train_outer)
    # Arr_length=  len(y_test_outer)
    # baseline_regression= np.full((Arr_length,1), baseline_mean);
    # baseline_mse[i] = mean_squared_error(y_test_outer, baseline_regression)

    #Baseline based on mean 
    adiposity_mean= statistics.mean(y_train_outer)
    Arr_length= len(y_test_outer)
    baseline_regression_adiposity= np.full((Arr_length,1), adiposity_mean);
    baseline_mean_error = mean_squared_error(y_test_outer, baseline_regression_adiposity)
    
    #Ridge regression 
    model_ridge = Ridge(random_state = 11)
    params_ridge = {'alpha': np.power(10.,range(-5,9))}

    #THis is K-fold for Ridge regression to choose best lambda parameter
    
    # gd_search = GridSearchCV(model_ridge, params_ridge, cv = cv_inner, return_train_score = False)
    # gd_search.fit(train_data, train_target)
    
    E_gen_ANN = np.zeros(len(params_ANN)) 
    h_saved = 0   
    TotalLambdaMse = np.zeros(len(params_lambda))
    

    #This is K-fold for ANN to choose best h parameter
    for (train_idx_inner, val_idx_inner) in cv_inner.split(X_train_outer):
        
        
        # Extract training and test set for current CV fold, convert to tensors
        # train_target_ANN = np.expand_dims(y_train_inner, axis=1)
        # val_target_ANN = np.expand_dims(val_target, axis=1)
        # X_train_ANN = torch.Tensor(X_train_inner)
        # y_train_ANN = torch.Tensor(train_target_ANN)
        # X_test_ANN = torch.Tensor(val_data)
        # y_test_ANN = torch.Tensor(val_target_ANN)
        
        #data for ridge:
        X_train_inner, X_test_inner = X_train_outer[train_idx_inner], X_train_outer[val_idx_inner]
        y_train_inner, y_test_inner = y_train_outer[train_idx_inner], y_train_outer[val_idx_inner]

        for l in range(len( params_lambda)):
    
            model = linear_model.Ridge(alpha=params_lambda[l])
            model.fit(X_train_inner , y_train_inner)
            y_est_inner = model.predict(X_test_inner)
            MseLinear=  mean_squared_error(y_test_inner, y_est_inner)
            TotalLambdaMse[l]=  TotalLambdaMse[l] + MseLinear
            
 
        
        #here ANN is going through all possble h:
        # for i in range(len(params_ANN)):
        #     h = params_ANN[i]
        #     # Define the model
        #     model = lambda: torch.nn.Sequential(
        #                         torch.nn.Linear(M, h), #M features to n_hidden_units
        #                         torch.nn.Tanh(),   # 1st transfer function,
        #                         torch.nn.Linear(h, 1), # n_hidden_units to 1 output neuron
        #                         # no final tranfer function, i.e. "linear output"
        #                         )
        #     loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

        #     net, final_loss, learning_curve = train_neural_net(model,
        #                                                         loss_fn,
        #                                                         X=X_train_ANN,
        #                                                         y=y_train_ANN,
        #                                                         n_replicates=n_replicates,
        #                                                         max_iter=max_iter)
   
        # # Determine estimated class labels for test set
        #     y_test_est = net(X_test_ANN)
        
        # # Determine errors and errors
        #     se = (y_test_est.float()-y_test_ANN.float())**2 # squared error
        #     mse = (sum(se).type(torch.float)/len(y_test_ANN)).data.numpy() #mean
        #     errors.append(mse) # store error rate for current CV fold 
        #     E_gen_ANN[i] = E_gen_ANN[i] + mse #np.sqrt(np.mean(errors))
        
        #np.sqrt(E_gen_ANN)
        #np.true_divide(E_gen_ANN, 4)
        # print('generalization errors: ')
        # print(E_gen_ANN)
        # h_saved = params_ANN[np.argmin(E_gen_ANN)]
        
        # print('h optimal: ' + str(h_saved))
        error_index = h_saved - 1;
    #After summing the errors previosly, we have to now devide it by number of params passed:
    for j in range(len(E_gen_ANN)):
         E_gen_ANN[j] = E_gen_ANN[j]/len(params_ANN)

    # print(df_ridge)
    # print('Accumulated generalization errors: ')
    # print(E_gen_ANN)
    print(' ')
    print('-----------K fold: ' + str(k) + '------------')
    print(' ')
    print('Baseline Mean squared error: '+ 
      str(baseline_mean_error))
    print('Ridge best score (differnet from RMSE;/): ' + str(TotalLambdaMse[np.argmin(TotalLambdaMse)]))
    print('Ridge best Lamba: ' + str(params_lambda[np.argmin(TotalLambdaMse)]))
    print('ANN optimal h:  ' + str(h_saved))
    print('ANN Estimated generalization error, RMSE: ' + str(E_gen_ANN[error_index]) ) #np.argmin(E_gen_ANN)
    #print('Ridge best score: ' + str(gd_search_ANN.best_score_))
   # print('Ridge best Lamba: ' + str(gd_search_ANN.best_params_))
    print(' ')
    print('-----------K fold: ' + str(k) + '------------')
    print(' ')
    k=k+1
    
  #https://medium.com/analytics-vidhya/using-the-corrected-paired-students-t-test-for-comparing-the-performance-of-machine-learning-dc6529eaa97f  

print('end')