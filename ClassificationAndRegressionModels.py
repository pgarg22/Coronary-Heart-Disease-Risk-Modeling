#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 23:18:03 2021

@author: Pranjal Garg, Piotr Saffrani, Avi Raj
"""


import pandas as pd
import numpy as np
from scipy import stats
import statistics
import matplotlib.pyplot as plt
import torch


from sklearn import linear_model,model_selection
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.linear_model import LinearRegression,Ridge,LogisticRegression,RidgeClassifier
from sklearn import tree

# Import your necessary dependencies
from sklearn.feature_selection import RFE

from toolbox_02450 import rlr_validate, train_neural_net, draw_neural_net


from sklearn.linear_model import Ridge


def dependent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = np.mean(data1), np.mean(data2)
	# number of paired samples
	n = len(data1)
	# sum squared difference between observations
	d1 = sum([(data1[i]-data2[i])**2 for i in range(n)])
	# sum difference between observations
	d2 = sum([data1[i]-data2[i] for i in range(n)])
	# standard deviation of the difference between means
	sd = np.sqrt((d1 - (d2**2 / n)) / (n - 1))
	# standard error of the difference between the means
	sed = sd / np.sqrt(n)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = n - 1
	# calculate the critical value
	cv = stats.t.ppf(1.0 - alpha, df)
	# calculate the p-value
	p = (1.0 - stats.t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p


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



#Regression code


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


#Limited attributes for regression


Xregression= np.hstack((Xregression[:,:3],Xregression[:,7:9]))






N, M = Xregression.shape



## Normalize and compute PCA (change to True to experiment with PCA preprocessing)
do_pca_preprocessing = False
if do_pca_preprocessing:
    Y = stats.zscore(Xregression,0)
    U,S,V = np.linalg.svd(Y,full_matrices=False)
    V = V.T
    #Components to be included as features
    k_pca = 3
    X = Xregression @ V[:,:k_pca]
    N, M = Xregression.shape





#Intitalizing settings

K1 = 10
K2 = 10
baseline_mse=np.empty(K1)
linear_mse=np.empty(K1)
ann_mse=np.empty(K1)


params_ANN =  [1,2,3,4]    # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000
argANNmin= np.empty(K1)
hnumber=len(params_ANN)  

arglambdamin= np.empty(K1)
i=0
lambdas = np.power(10.,range(-5,5))
lamdanumber=len(lambdas)  



## Crossvalidation
# Create crossvalidation partition for evaluation
CV_outer = model_selection.KFold(K1)
CV_outer.get_n_splits(Xregression)

#double layer cross validation: https://towardsdatascience.com/nested-cross-validation-hyperparameter-optimization-and-model-selection-5885d84acda
for (train_index_outer, test_index_outer) in CV_outer.split(Xregression):
    X_train_outer, X_test_outer = Xregression[train_index_outer], Xregression[test_index_outer]
    y_train_outer, y_test_outer = y[train_index_outer], y[test_index_outer]
    
    CV_inner = model_selection.KFold(K2)
    CV_inner.get_n_splits(X_train_outer)
    
    
    TotalLambdaMse=np.zeros(lamdanumber)
    TotalhMse=np.zeros(hnumber)
    
    for (train_index_inner, test_index_inner) in CV_inner.split(X_train_outer):
        X_train_inner, X_test_inner = X_train_outer[train_index_inner], X_train_outer[test_index_inner]
        y_train_inner, y_test_inner = y_train_outer[train_index_inner], y_train_outer[test_index_inner]
        
       # Extract training and test set for current CV fold, convert to tensor
        train_target_ANN = np.expand_dims(y_train_outer[train_index_inner], axis=1)
        val_target_ANN = np.expand_dims(y_train_outer[test_index_inner], axis=1)
        X_train_ANN = torch.Tensor(X_train_outer[train_index_inner,:])
       # y_train_ANN = torch.Tensor(y_train_outer[train_index_inner])
        X_test_ANN = torch.Tensor(X_train_outer[test_index_inner,:])
        #y_test_ANN = torch.Tensor(y_train_outer[test_index_inner])
        
        y_train_ANN = torch.Tensor(train_target_ANN)
        y_test_ANN = torch.Tensor(val_target_ANN)

        
        j=0
        for l in lambdas:
    
            modellinear = linear_model.Ridge(alpha=l)
            modellinear.fit(X_train_inner , y_train_inner)
            y_est_inner = modellinear.predict(X_test_inner)
            MseLinear=  mean_squared_error(y_test_inner, y_est_inner)
            TotalLambdaMse[j]=  TotalLambdaMse[j] + MseLinear
            j=j+1
            
        j2=0    
        for ht in range(len(params_ANN)):
            h = params_ANN[ht]
            # Define the model
            modelANN = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, h), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(h, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
            
            # Extract training and test set for current CV fold, convert to tensors
            net, final_loss, learning_curve = train_neural_net(modelANN,
                                                                loss_fn,
                                                                X=X_train_ANN,
                                                                y=y_train_ANN,
                                                                n_replicates=n_replicates,
                                                                max_iter=max_iter)
            # Determine estimated class labels for test set
            y_ann_est = net(X_test_ANN)
            se = (y_ann_est.float()-y_test_ANN.float())**2 # squared error
            mseANN = (sum(se).type(torch.float)/len(y_test_ANN)).data.numpy() #mean
            TotalhMse[j2]=  TotalhMse[j2] + mseANN
            j2=j2+1
   
             
   
     #ANN
    argANN= np.argmin(TotalhMse)
    argANNmin[i]=params_ANN[argANN]
    h2= int(argANNmin[i])
    
    modelANN = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, h2), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(h2, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    
    train_target_ANN_outer = np.expand_dims(y[train_index_outer], axis=1)
    val_target_ANN_outer = np.expand_dims(y[test_index_outer], axis=1)
    X_train_ANN_outer = torch.Tensor(Xregression[train_index_outer,:])
       # y_train_ANN = torch.Tensor(y_train_outer[train_index_inner])
    X_test_ANN_outer = torch.Tensor(Xregression[test_index_outer,:])
        #y_test_ANN = torch.Tensor(y_train_outer[test_index_inner])
        
    y_train_ANN_outer = torch.Tensor(train_target_ANN_outer)
    y_test_ANN_outer = torch.Tensor(val_target_ANN_outer)
            
            # Extract training and test set for current CV fold, convert to tensors
    net, final_loss, learning_curve = train_neural_net(modelANN,
                                                                loss_fn,
                                                                X=X_train_ANN_outer,
                                                                y=y_train_ANN_outer,
                                                                n_replicates=n_replicates,
                                                                max_iter=max_iter)
            # Determine estimated class labels for test set
    y_ann_est_outer = net(X_test_ANN_outer)
    se_outer = (y_ann_est_outer.float()-y_test_ANN_outer.float())**2 # squared error
    ann_mse[i] = (sum(se_outer).type(torch.float)/len(y_test_ANN_outer)).data.numpy() #mean
    
    
            
   
    
    #TotalLambdaMse = [xrandom / lamdanumber for xrandom in TotalLambdaMse] 
    #print (TotalLambdaMse)
    
    
    #Linear Regression
    arglambda= np.argmin(TotalLambdaMse)
    arglambdamin[i]=lambdas[arglambda]
    
    model_linear_outer = linear_model.Ridge(alpha=arglambdamin[i])
    model_linear_outer.fit(X_train_outer , y_train_outer)
    y_est_outer = model_linear_outer.predict(X_test_outer)
    linear_mse[i]=  mean_squared_error(y_test_outer, y_est_outer)
    
    
    
    #Baseline based on mean
    baseline_mean= statistics.mean(y_train_outer)
    Arr_length=  len(y_test_outer)
    baseline_regression= np.full((Arr_length,1), baseline_mean);
    baseline_mse[i] = mean_squared_error(y_test_outer, baseline_regression)
    sum2=0
    
    
    i=i+1

    
    
    
    
    
for t2 in range(K1):
         ann_mse[t2] =  ann_mse[t2]/hnumber    
    
    
    
print ("baseline", baseline_mse)
print ("Lambdas", arglambdamin)
print ("linear", linear_mse)
print ("h", argANNmin)
print ("ANN", ann_mse)


comparison_ANN_baseline = np.zeros(K1)
comparison_ridge_baseline = np.zeros(K1)
comparison_ANN_ridge= np.zeros(K1)

for i in range(K1):
    comparison_ANN_baseline[i] = ann_mse[i] - baseline_mse[i]
    comparison_ridge_baseline[i] = linear_mse[i] - baseline_mse[i]
    comparison_ANN_ridge[i] = ann_mse[i] - linear_mse[i]
    
confidence_ANN_baseline = stats.t.interval(0.95, comparison_ANN_baseline.size-1, loc=np.mean(comparison_ANN_baseline), scale=stats.sem(comparison_ANN_baseline))
confidence_ridge_baseline = stats.t.interval(0.95, comparison_ridge_baseline.size-1, loc=np.mean(comparison_ridge_baseline), scale=stats.sem(comparison_ridge_baseline))
confidence_ANN_ridge = stats.t.interval(0.95, comparison_ANN_ridge.size-1, loc=np.mean(comparison_ANN_ridge), scale=stats.sem(comparison_ANN_ridge))

alpha = 0.05
t_stat, df, cv, p = dependent_ttest(ann_mse, baseline_mse, alpha)
print('Pvalue ANN vs baseline:  t=%.3f, df=%d, cv=%.3f, p=%.9f' % (t_stat, df, cv, p))
print('confidence interval: ' + str(confidence_ANN_baseline))

t_stat, df, cv, p = dependent_ttest(linear_mse, baseline_mse, alpha)
print(' Pvalue ridge vs baseline: t=%.3f, df=%d, cv=%.9f, p=%.9f' % (t_stat, df, cv, p))
print('confidence interval: ' + str(confidence_ridge_baseline))

#stat, p = ttest_ind(ANN_errors, Ridge_errors)

t_stat, df, cv, p = dependent_ttest(ann_mse, linear_mse, alpha)
print('ANN vs ridge: t=%.3f, df=%d, cv=%.3f, p=%.9f' % (t_stat, df, cv, p))
print('confidence interval: ' + str(confidence_ANN_ridge))

#One level cross validation for lambda graph 

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambdas2 = np.power(10.,range(-6,3))

#lambdascores = np.zeros((8, 2))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
GenError = np.empty(9)

k=0
i=0

for l in lambdas2:
    totalerror=0
    
   # opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y, lambdas, 10)

    model = linear_model.Ridge(alpha=l)
    kf = model_selection.KFold(n_splits=10)
    kf.get_n_splits(Xregression)
    for train_index,test_index in kf.split(Xregression):
        X_train, X_test = Xregression[train_index], Xregression[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train , y_train)
        y_est = model.predict(X_test)
        residual = y_est-y_test
        Error=  mean_squared_error(y_test, y_est)
        totalerror=totalerror+Error        
    GenError[i]= totalerror/10
        
    i=i+1

arggenmin= np.argmin(GenError)
lambdamin=lambdas2[arggenmin]
modelgraph= linear_model.Ridge(alpha=lambdamin)
modelgraph.fit(Xregression , y)
y_graph_est = modelgraph.predict(Xregression)

plt.figure()
plt.title('Predicted value of adiposity against actual values')
plt.scatter(y, y_graph_est, marker='o');
plt.xlabel("Adiposity True");
plt.ylabel("Adiposity predicted ");   
plt.show() 
    
plt.figure()
plt.title('Generalization Error as a function of lambda')
plt.plot(lambdas2, GenError,scalex=True)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("lambda");
plt.ylabel("Generalization Error ");   
plt.show()           







#Intitalizing settings

K1 = 10
K2 = 10


BaselineScore=np.empty(K1)
LinearRegressionScore=np.empty(K1)
DecisionTreeScore=np.empty(K1)


baseline_classification_mse=np.empty(K1)
linear_classification_mse=np.empty(K1)
DTC_classiciation_mse=np.empty(K1)
baseline_classification_score=np.empty(K1)
linear_classification_score=np.empty(K1)
DTC_classiciation_score=np.empty(K1)

arglambdaminClassification= np.empty(K1)

lambdasclassification = np.power(10.,range(0,60))
lamdanumberclassfication=len(lambdasclassification)  

arglambdaminClassification= np.empty(K1)

samplessclassification = np.arange(start=2, stop=60, step=1)
samplesnumber=len(samplessclassification)  

arglsampleminClassification= np.empty(K1)


i=0


## Crossvalidation
# Create crossvalidation partition for evaluation
CV_outer = model_selection.KFold(K1)
CV_outer.get_n_splits(Xclassification)

#double layer cross validation: https://towardsdatascience.com/nested-cross-validation-hyperparameter-optimization-and-model-selection-5885d84acda
for (train_index_outer, test_index_outer) in CV_outer.split(Xclassification):
    X_train_outer, X_test_outer = Xclassification[train_index_outer], Xclassification[test_index_outer]
    y_train_outer, y_test_outer = yclassification[train_index_outer], yclassification[test_index_outer]
    
     
    CV_inner = model_selection.KFold(K2)
    CV_inner.get_n_splits(X_train_outer)
    
    
    ClassificationTotalLambdaMse=np.zeros(lamdanumberclassfication)
    ClassificationTotalDTCMse=np.zeros(samplesnumber)
    ClassificationTotalLambdaScore=np.zeros(lamdanumberclassfication)
    ClassificationTotalDTCScore=np.zeros(samplesnumber)
    
    for (train_index_inner, test_index_inner) in CV_inner.split(X_train_outer):
        X_train_inner, X_test_inner = X_train_outer[train_index_inner], X_train_outer[test_index_inner]
        y_train_inner, y_test_inner = y_train_outer[train_index_inner], y_train_outer[test_index_inner]
        
        
        
        j=0
        for l in lambdasclassification:
    
            modellinearclassification = linear_model.Ridge(alpha=l)
            modellinearclassification.fit(X_train_inner , y_train_inner)
            y_est_inner = modellinearclassification.predict(X_test_inner)
            MseLinearClassification=  mean_squared_error(y_test_inner, y_est_inner)
            LinearClassificationScore= modellinearclassification.score(X_test_inner,y_test_inner)
            ClassificationTotalLambdaMse[j]=  ClassificationTotalLambdaMse[j] + MseLinearClassification
            ClassificationTotalLambdaScore[j]=  ClassificationTotalLambdaScore[j] + LinearClassificationScore
        
            j=j+1
        
        
        j=0
        for s in samplessclassification:
            DTC = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=s)
            DTC.fit(X_train_inner, y_train_inner)
            y_DTC_est_inner = DTC.predict(X_test_inner)
            DTC_mse=  mean_squared_error(y_test_inner, y_DTC_est_inner)
            DecisionTreeScore= DTC.score(X_test_inner,y_test_inner)
            ClassificationTotalDTCMse[j]=  ClassificationTotalDTCMse[j] + DTC_mse
            ClassificationTotalDTCScore[j]=  ClassificationTotalDTCScore[j] + DecisionTreeScore
        
            j=j+1
        
   
        
    #Baseline based on mode
    
    baseline_mode= statistics.mode(y_train_outer)
    Arr_length=  len(y_test_outer)
    baseline_classification= np.full((Arr_length,1), baseline_mode);
    baseline_classification_mse[i] = mean_squared_error(y_test_outer, baseline_classification)
    sum2=0
    
    #Linear Regression Classification
    
    arglambdaClassification= np.argmin(ClassificationTotalLambdaMse)
    arglambdaminClassification[i]=lambdasclassification[arglambdaClassification]
    
    model_linear_outer_classification = linear_model.Ridge(alpha=arglambdaminClassification[i])
    model_linear_outer_classification.fit(X_train_outer , y_train_outer)
    y_est_outer_Classification = model_linear_outer_classification.predict(X_test_outer)
    linear_classification_mse[i]=  mean_squared_error(y_test_outer, y_est_outer_Classification)
    linear_classification_score[i]=model_linear_outer_classification.score(X_test_outer,y_test_outer)
    
    
    #Decision Tree classification
    
    argsampleClassification= np.argmin(ClassificationTotalDTCScore)
    arglsampleminClassification[i]=samplessclassification[argsampleClassification]
    minsample= int(arglsampleminClassification[i])
    model_DTC_outer_classification = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=minsample)
    model_DTC_outer_classification.fit(X_train_outer , y_train_outer)
    y_est_outer_DTC = model_DTC_outer_classification.predict(X_test_outer)
    DTC_classiciation_mse[i]=  mean_squared_error(y_test_outer, y_est_outer_DTC)
    DTC_classification_score= model_DTC_outer_classification.score(X_test_outer,y_test_outer)

    i=i+1





print ("baseline", baseline_classification_mse)
print ("Lambdas", arglambdaminClassification)
print ("linear", linear_classification_mse)
print ("`Sample", arglsampleminClassification)
print ("DTC", DTC_classiciation_mse)




comparison_DTC_baseline = np.zeros(K1)
compraison_linear_baseline = np.zeros(K1)
comparison_DTC_linear = np.zeros(K1)
for i in range(K1):
    comparison_DTC_baseline[i] = DTC_classiciation_mse[i] - baseline_classification_mse[i]
    compraison_linear_baseline[i] = linear_classification_mse[i] - baseline_classification_mse[i]
    comparison_DTC_linear[i] = DTC_classiciation_mse[i] - linear_classification_mse[i]
    
confidence_DTC_baseline = stats.t.interval(0.95, comparison_DTC_baseline.size-1, loc=np.mean(comparison_DTC_baseline), scale=stats.sem(comparison_DTC_baseline))
confidence_linear_baseline = stats.t.interval(0.95, compraison_linear_baseline.size-1, loc=np.mean(compraison_linear_baseline), scale=stats.sem(compraison_linear_baseline))
confidence_DTC_linear = stats.t.interval(0.95, comparison_DTC_linear.size-1, loc=np.mean(comparison_DTC_linear), scale=stats.sem(comparison_DTC_linear))



alpha = 0.05
t_stat, df, cv, p = dependent_ttest(DTC_classiciation_mse, baseline_classification_mse, alpha)
print('Pvalue DecisionTree vs baseline:  t=%.3f, df=%d, cv=%.3f, p=%.9f' % (t_stat, df, cv, p))
print('confidence interval: ' + str(confidence_DTC_baseline))


t_stat, df, cv, p = dependent_ttest(linear_classification_mse, baseline_classification_mse, alpha)
print(' Pvalue ridge vs baseline: t=%.3f, df=%d, cv=%.3f, p=%.9f' % (t_stat, df, cv, p))
print('confidence interval: ' + str(confidence_linear_baseline))

#stat, p = ttest_ind(ANN_errors, Ridge_errors)

t_stat, df, cv, p = dependent_ttest(DTC_classiciation_mse, linear_classification_mse, alpha)
print('DecisionTree vs ridge: t=%.3f, df=%d, cv=%.3f, p=%.9f' % (t_stat, df, cv, p))
print('confidence interval: ' + str(confidence_DTC_linear))



model_linear_classification = linear_model.Ridge(alpha=10)
model_linear_classification.fit(Xclassification, yclassification)
y_est_Classification = model_linear_classification.predict(Xclassification)
plt.figure()
plt.title('Predicted value of CHD against actual values')
plt.scatter(yclassification, y_est_Classification, marker='o');
plt.xlabel("CHD True");
plt.ylabel("CHD predicted ");   
plt.show()

