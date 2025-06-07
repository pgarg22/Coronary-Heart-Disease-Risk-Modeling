# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:33:49 2021

@author:  pranjalgarg, piotrsaffrani
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.linalg as linalg
from scipy.linalg import svd
import statistics
from scipy import stats
import seaborn as snsco

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
cols = range(0, 9) 
X = raw_data[:, cols]
attributeNames = np.asarray(data.columns[cols])
y = raw_data[:,-1]
classNames = np.unique(y)
classDict = dict(zip(classNames,range(len(classNames))))
N, M = X.shape
C = len(classNames)


#making sure we are only handling numeric data (not strings)
data_numerics_only = data.select_dtypes(include=np.number)


#display basic statistics
print('display basic data statistics. ')
for col in data_numerics_only:
    print(col)
    pd_to_list = list(data_numerics_only[col])
    print("min: ", round(data_numerics_only[col].min(), 2),
          "; max:", round(data_numerics_only[col].max(), 2), 
          "; Standard deviation:", round(data_numerics_only[col].std(),2),
          "; mean:", round(data_numerics_only[col].mean(), 2),
          "; Q1:", round(data_numerics_only[col].quantile(0.25),2),
          "; median:", round(data_numerics_only[col].median(), 2),
          "; Q3:", round(data_numerics_only[col].quantile(0.75), 2)
          )
    

nbins=[8,7,12,9,2,8,6,8,6]

#Histograms along with Ben Shapiro Normality test results
f2 = plt.figure(figsize=(20,10))
u = np.floor(np.sqrt(M)); 
v = np.ceil(float(M)/u)
for i in range(M):
    ax = f2.add_subplot(u,v,i+1)
    mins= np.min(X[:,i]) 
    maxs = np.max(X[:,i]) 
    mean = statistics.mean(X[:,i]) 
    sd = statistics.stdev(X[:,i]) 
    
    ax.hist(X[:,i],bins= nbins[i], color='blue', density=True)
    
    x = np.linspace(mins, maxs, 462)
    pdf = stats.norm.pdf(x,mean,sd)
    ax.plot(x,pdf,'.',color='red')

    plt.xlabel(attributeNames[i])
    p= stats.shapiro(X[:,i])
    if (p.pvalue>0.05):
     print (attributeNames[i],"Is Gaussian,","ShapiroResult: statistic=", p.statistic,",pvalue=", p.pvalue)
    else:
     print (attributeNames[i],"Is Not Gaussian,", "ShapiroResult: statistic=", p.statistic,",pvalue=", p.pvalue)



#Box Plot  Standardized
Xstandardized = StandardScaler().fit_transform(X)
cols2 = range(0, 9) 
Xbox= Xstandardized[:, cols2]
plt.figure()
plt.boxplot(Xbox)
plt.xticks(range(1,10),attributeNames)


#Correlation heat map
correlation_mat = data.corr()
plt.figure()
sns.heatmap(correlation_mat, annot = True)
corr_pairs = correlation_mat.unstack()


#Perform PCA from this website: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
print('Perform PCA. ')
#First standarize
features = data.columns.to_numpy()
features = np.delete(features, 9) #delete chd column name to separate target from features
x = data.loc[:, features].values #Separating out the features
y = data.loc[:,['chd']].values # Separating out the target
x = StandardScaler().fit_transform(x) # Standardizing the features

#Then perform PCA
pca = PCA(n_components=9)
principalComponents = pca.fit_transform(x)
principal_data = pd.DataFrame(data = principalComponents, 
                                columns = ['principal component 1', 'pricipal comonent 2', 'pricipal comonent 3', '4', '5' ,'6', '7' , '8', '9'])
final_data = pd.concat([principal_data, data[['chd']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['r', 'g' ]
for target, color in zip(targets,colors):
    indicesToKeep = final_data['chd'] == target
    ax.scatter(final_data.loc[indicesToKeep, 'principal component 1']
               , final_data.loc[indicesToKeep, 'pricipal comonent 2']
               , c = color
               , s = 50)
ax.legend(['CHD absent', 'CHD present'])
ax.grid()
plt.show()
print('sum of each PCA variances:', sum(pca.explained_variance_ratio_) *100,'%')
print('variance of each PCA component [%]', pca.explained_variance_ratio_)
rho = pca.explained_variance_ratio_
threshold = 0.9

# Plot cumulative variance
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

#Plot PCA Component coefficients - direction vectors as a hotmap
N,M = x.shape
U,S,Vh = svd(x,full_matrices=False)
V=Vh.T

PCA = ["PCA 1", "PCA 2","PCA 3","PCA 4","PCA 5","PCA 6","PCA 7","PCA 8","PCA 9" ]
fig, ax = plt.subplots()
im = ax.imshow(V, cmap='hot', interpolation='nearest')

# We want to show all ticks...
ax.set_xticks(np.arange(len(features)))
ax.set_yticks(np.arange(len(PCA)))
# ... and label them with the respective list entries
ax.set_xticklabels(features)
ax.set_yticklabels(PCA)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
          rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(PCA)):
    for j in range(len(features)):
        text = ax.text(j, i, round(V[i, j], 2),
                        ha="center", va="center", color="g", size=7, fontdict=None)

ax.set_title("PCA Component Coefficients")
fig.tight_layout()
plt.colorbar(im)
plt.show()

