import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression,Ridge,LogisticRegression,RidgeClassifier
from sklearn import tree



data = pd.read_csv(r"C:\Users\rajav\IdeaProjects\Coronary Heart Disease\CHD.csv")

temp=data.loc[:,"sbp":"age"]
temp=data.drop(["famhist","row.names"],axis=1)
name=np.array(list(temp))
scalar=StandardScaler()

data_scalar=scalar.fit_transform(data[["sbp","tobacco","ldl","adiposity","typea","obesity","alcohol","age"]])
data_scalar=pd.DataFrame(data_scalar, columns =name[:8])

data.famhist = data.famhist.str.replace('Present','1') 
data.famhist = data.famhist.str.replace('Absent','0')
data.famhist = data.famhist.astype(int)

data_scalar=pd.concat([data_scalar,data.famhist], axis=1)
data_scalar=data_scalar[list(data)[1:10]]

n_alpha=60
splits=10
samples=60

kf = KFold(n_splits=splits)
score_final=pd.DataFrame()
score_ridge_array=[]
G_error_array=[]
baseline_MSE_array=[]
DTC_score_values=pd.DataFrame()
DTC_MSE_values=pd.DataFrame()
score_final_Ridge=pd.DataFrame()
MSE_final_Ridge=pd.DataFrame()


for train_index, test_index in kf.split(data_scalar,data.chd):
    X_train= data_scalar.iloc[train_index]
    X_test= data_scalar.iloc[test_index]
    Y_train=data.chd[train_index]
    Y_test=data.chd[test_index]
    score_ridge_array=[]
    MSE_ridge_array=[]
    Y_pred=[Y_train.mean()]*len(Y_test)
    MSE_baseline=mean_squared_error(Y_test, Y_pred)
    baseline_MSE_array.append(MSE_baseline)
    MSE_DTC_array=[]
    score_DTC_array=[]
    
    for i in range(2,samples):
        DTC = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=i)
        DTC.fit(X_train, Y_train)
        score_DTC=DTC.score(X_test, Y_test)
        #print(score_DTC)
        y_pred=DTC.predict(X_test)
        MSE=mean_squared_error(Y_test, y_pred)
        MSE_DTC_array.append(MSE)
        score_DTC_array.append(score_DTC)
    #print(score_DTC_array)
    score_DTC_array = pd.Series(score_DTC_array)
    MSE_DTC_array = pd.Series(MSE_DTC_array) 
    #print(MSE_DTC_array)
    DTC_score_values=pd.concat([DTC_score_values,score_DTC_array],axis=1)
    DTC_MSE_values=pd.concat([DTC_MSE_values,MSE_DTC_array],axis=1)
    
    for i in range(n_alpha):
        ridge=RidgeClassifier(alpha=np.power(10,i))
        ridge.fit(X_train, Y_train)
        score_ridge=ridge.score(X_test, Y_test)
        score_ridge_array.append(score_ridge)
        y_pred=ridge.predict(X_test)
        MSE=MSE=mean_squared_error(Y_test, y_pred)
        #print(MSE)
        MSE_ridge_array.append(MSE)
       
        #n_score.append(len(score_ridge_array))#Used for plotting
    score_ridge_array = pd.Series(score_ridge_array)
    score_final_Ridge=pd.concat([score_final_Ridge,score_ridge_array],axis=1)
    MSE_ridge_array=pd.Series(MSE_ridge_array)
    MSE_final_Ridge=pd.concat([MSE_final_Ridge,MSE_ridge_array],axis=1)
    
        
        
        
a1=[]
for i in range(2,samples):
    b=("sample={}".format(i))
    a1.append(b)
  

a2=[]
for i in range(splits):
    b=("{}th split".format(i))
    a2.append(b)    
DTC_score_values.columns=a2 
DTC_score_values=DTC_score_values.T
DTC_score_values.columns=a1


DTC_MSE_values.columns=a2 
DTC_MSE_values=DTC_MSE_values.T
DTC_MSE_values.columns=a1

G_error_array_Decision_Tree=pd.Series(dtype=float)
for i in range(samples-2):
    #print(i)
    G_error_array_Decision_Tree["sample={}".format(i+2)]=DTC_MSE_values.iloc[:,i].mean()

a1=[]
for i in range(n_alpha):
    b=("Lambda={}".format(i))
    a1.append(b)
  

a2=[]
for i in range(splits):
    b=("{}th split".format(i))
    a2.append(b)    
score_final_Ridge.columns=a2 
score_final_Ridge=score_final_Ridge.T
score_final_Ridge.columns=a1
MSE_final_Ridge.columns=a2 
MSE_final_Ridge=MSE_final_Ridge.T
MSE_final_Ridge.columns=a1

G_error_array_Ridge=pd.Series(dtype=float)
for i in range(n_alpha):
    #print(i)
    G_error_array_Ridge["Lamda={}".format(i)]=MSE_final_Ridge.iloc[:,i].mean()
