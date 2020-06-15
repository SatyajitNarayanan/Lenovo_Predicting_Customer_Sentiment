# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 18:55:19 2018

@author: Satyajit Narayanan
"""
#%%
# Importing required Packages
import pandas as pd
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances 
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.model_selection import train_test_split  
from sklearn import metrics  



# Importing Data

# Importing Commercial Sentiment Data
CommSentiClean_v2 = pd.read_excel(r'C:\Users\Satyajit Narayanan\Desktop\560 Project\Cleaned Data\CommSentiClean_v2.xlsx')

# Importing Commercial EOS Data
CommEOS = pd.read_excel(r'C:\Users\Satyajit Narayanan\Desktop\560 Project\Regression\CommEOS.xlsx')

# Importing Commercial pNPS Data
CommpNPS = pd.read_excel(r'C:\Users\Satyajit Narayanan\Desktop\560 Project\Regression\CommpNPS.xlsx')




# Preprocessing 

# Creating Taxonomy-Sentiment Column
CommSentiClean_v2['TaxSent'] = CommSentiClean_v2[['TaxLevel2', 'Sentiment']].apply(lambda x: ''.join(x), axis=1)

# Creating Month-Series Column 

CommSentiClean_v2['SeriesMonth'] = CommSentiClean_v2[['Series', 'Month']].astype(str).sum(axis=1)


# Getting count of Taxonomy Sentiment combination on Series Month level
CommSentiTaxMonth = pd.crosstab(CommSentiClean_v2.SeriesMonth, CommSentiClean_v2.TaxSent)


# Joining EOS data with Taxonomy Sentiment data
CommX = pd.merge(CommSentiTaxMonth, CommEOS, how='left', left_index=True, right_on='SeriesMonth')


CommX = CommX.loc[CommX['Series'].isin(['E SERIES','L SERIES','M SERIES','N SERIES - CHROME','P SERIES WORKSTATION','T SERIES','THINKPAD 13', 'X SERIES','X1 SERIES','YOGA'])]


CommX['Epoch5']=CommX['Epoch']+5


Comm = pd.merge(CommX, CommpNPS, how='inner', left_on=['Series','Epoch5'], right_on=['Series','Epoch'])

CommF= Comm.copy()
CommF = CommF.drop([ 'SeriesMonth_x', 'SeriesMonth_y','Epoch_x', 'Epoch5', 'Epoch_y', 'Month_y'], axis=1)


Comm = Comm.drop(['Series', 'SeriesMonth_x', 'SeriesMonth_y','Epoch_x', 'Epoch5', 'Epoch_y', 'Month_x', 'Month_y'], axis=1)

CommXs = Comm.iloc[:,:-1]
CommY = Comm.iloc[:,-1]



# Principal Component Analysis
    
X = CommXs
y = CommY



# Scikit-learn PCA
pca = PCA()

# Scale and transform data to get Principal Components

X_reduced = pca.fit_transform(scale(X))

# Variance (% cumulative) explained by the principal components

np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

# Standardizing

X_std = StandardScaler().fit_transform(X)



# 1 - Eigendecomposition - Computing Eigenvectors and Eigenvalues


mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)


cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# 2 - Selecting Principal Components

for ev in eig_vecs.T:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')



# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])



# Explained Variance (Cummulative)
    
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.plot(cum_var_exp)



print(f'{cum_var_exp[49].real:f}')



# Choosing first 80 components out of 188 (explain 98% of variance)

matrix_w = np.hstack((eig_pairs[0][1].reshape(110,1),
eig_pairs[1][1].reshape(110,1),
eig_pairs[2][1].reshape(110,1),
eig_pairs[3][1].reshape(110,1),
eig_pairs[4][1].reshape(110,1),
eig_pairs[5][1].reshape(110,1),
eig_pairs[6][1].reshape(110,1),
eig_pairs[7][1].reshape(110,1),
eig_pairs[8][1].reshape(110,1),
eig_pairs[9][1].reshape(110,1),
eig_pairs[10][1].reshape(110,1),
eig_pairs[11][1].reshape(110,1),
eig_pairs[12][1].reshape(110,1),
eig_pairs[13][1].reshape(110,1),
eig_pairs[14][1].reshape(110,1),
eig_pairs[15][1].reshape(110,1),
eig_pairs[16][1].reshape(110,1),
eig_pairs[17][1].reshape(110,1),
eig_pairs[18][1].reshape(110,1),
eig_pairs[19][1].reshape(110,1),
eig_pairs[20][1].reshape(110,1),
eig_pairs[21][1].reshape(110,1),
eig_pairs[22][1].reshape(110,1),
eig_pairs[23][1].reshape(110,1),
eig_pairs[24][1].reshape(110,1),
eig_pairs[25][1].reshape(110,1),
eig_pairs[26][1].reshape(110,1),
eig_pairs[27][1].reshape(110,1),
eig_pairs[28][1].reshape(110,1),
eig_pairs[29][1].reshape(110,1),
eig_pairs[30][1].reshape(110,1),
eig_pairs[31][1].reshape(110,1),
eig_pairs[32][1].reshape(110,1),
eig_pairs[33][1].reshape(110,1),
eig_pairs[34][1].reshape(110,1),
eig_pairs[35][1].reshape(110,1),
eig_pairs[36][1].reshape(110,1),
eig_pairs[37][1].reshape(110,1),
eig_pairs[38][1].reshape(110,1),
eig_pairs[39][1].reshape(110,1),
eig_pairs[40][1].reshape(110,1),
eig_pairs[41][1].reshape(110,1),
eig_pairs[42][1].reshape(110,1),
eig_pairs[43][1].reshape(110,1),
eig_pairs[44][1].reshape(110,1),
eig_pairs[45][1].reshape(110,1),
eig_pairs[46][1].reshape(110,1),
eig_pairs[47][1].reshape(110,1),
eig_pairs[48][1].reshape(110,1),
eig_pairs[49][1].reshape(110,1)
))


matrix_wdf = pd.DataFrame(matrix_w.real)


# Multiplying values alpha values of components with Xs
Y = X_std.dot(matrix_wdf)
Ydf = pd.DataFrame(Y)


# Joining 

CommReg = pd.merge(Ydf, pd.DataFrame(y), how='left', left_index=True, right_index=True)


# Regession
regX = CommReg.iloc[:, :-1].values  
regY = CommReg.iloc[:, -1].values


regX = np.take(regX,[0,1,3,5,9,14,15,24,25,27,28,29,30,32,33,35,36,40,44,46,48,49], axis=1)


# Splitting this data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(regX, regY, test_size=0.1, random_state=0)  

# Training the algorithm
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

# Making Prediction
y_pred = regressor.predict(X_test)  


X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())

Compdf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(Compdf)


# Evaluate

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  





coeff_df = pd.DataFrame(regressor.coef_, pd.DataFrame(regX).columns, columns=['Coefficient'])  

Beta = pd.DataFrame(regressor.intercept_,range(1),columns=['Coefficient']).append(coeff_df)


cols = [0,1,3,5,9,14,15,24,25,27,28,29,30,32,33,35,36,40,44,46,48,49]
matrix_fill = matrix_wdf.take(matrix_wdf.columns[cols],axis=1)
Ydf_fill = Ydf.take(Ydf.columns[cols],axis=1)



# Writing results


CommFinal = pd.ExcelWriter('CommFinal.xlsx')
CommF.to_excel(CommFinal,'Final Data')
matrix_fill.to_excel(CommFinal,'alpha values')
Ydf_fill.to_excel(CommFinal,'Component values')
Beta.to_excel(CommFinal,'Beta values')
CommFinal.save()



matrix_pca_comm = pd.ExcelWriter('matrix_pca_comm.xlsx')
matrix_wdf.to_excel(matrix_pca_comm,'matrix_w')
Ydf.to_excel(matrix_pca_comm,'Y')
matrix_pca_comm.save()


CorrComm2 = pd.crosstab(CommSentiClean_v2.Month, CommSentiClean_v2.TaxLevel).corr().abs()
CorrComm2S = CorrComm2.unstack()
CorrComm2SO = CorrComm2S.sort_values(kind="quicksort")
print(CorrComm2SO)

CorrComm2SOlog = pd.ExcelWriter('CorrComm2SO.xlsx')
CorrComm2.to_excel(CorrComm2SOlog,'CorrComm2SO')
CorrComm2SOlog.save()






























#%%
# Consumers Regression Model


# Importing Consumer Sentiment Data
ConsuSentiClean_v2 = pd.read_excel(r'C:\Users\Satyajit Narayanan\Desktop\560 Project\Cleaned Data\ConsuSentiClean_v2.xlsx')


# Importing Consumer pNPS Data
ConsupNPS = pd.read_excel(r'C:\Users\Satyajit Narayanan\Desktop\560 Project\Regression\ConsupNPS.xlsx')


# Importing Consumer EOS Data
ConsuEOS = pd.read_excel(r'C:\Users\Satyajit Narayanan\Desktop\560 Project\Regression\ConsuEOS.xlsx')



#%%
'''
# Getting correlation of Taxonomies at a month level
CorrConsu2 = pd.crosstab(ConsuSentiClean_v2.Month, CommSentiClean_v2.TaxLevel).corr().abs()

CorrComm2SOlog = pd.ExcelWriter('CorrConsu2.xlsx')
CorrConsu2.to_excel(CorrComm2SOlog,'CorrConsu2')
CorrComm2SOlog.save()
'''


#%%
# Creating Taxonomy-Sentiment Column
ConsuSentiClean_v2['TaxSent'] = ConsuSentiClean_v2[['TaxLevel2', 'Sentiment']].apply(lambda x: ''.join(x), axis=1)

# Creating Month-Series Column 

ConsuSentiClean_v2['SeriesMonth'] = ConsuSentiClean_v2[['Series', 'Month']].astype(str).sum(axis=1)



# Getting count of Taxonomy Sentiment combination on Series Month level
ConsuSentiTaxMonth = pd.crosstab(ConsuSentiClean_v2.SeriesMonth, ConsuSentiClean_v2.TaxSent)



ConsuX = pd.merge(ConsuSentiTaxMonth, ConsuEOS, how='left', left_index=True, right_on='SeriesMonth')


ConsuX = ConsuX.loc[ConsuX['Series'].isin(['A SERIES',
'IDEAPAD 100 SERIES','IDEAPAD 300 SERIES','IDEAPAD 500 SERIES','IDEAPAD 700 SERIES',
'LEGION (Y SERIES)','MIIX SERIES','YOGA'
])]




#%%

ConsuX['Epoch5']=ConsuX['Epoch']+5


Consu = pd.merge(ConsuX, ConsupNPS, how='inner', left_on=['Series','Epoch5'], right_on=['Series','Epoch'])


ConsuF= Consu.copy()
ConsuF = ConsuF.drop([ 'SeriesMonth_x', 'SeriesMonth_y','Epoch_x', 'Epoch5', 'Epoch_y', 'Month_y'], axis=1)


Consu = Consu.drop(['Series', 'SeriesMonth_x', 'SeriesMonth_y','Epoch_x', 'Epoch5', 'Epoch_y', 'Month_x', 'Month_y'], axis=1)

ConsuXs = Consu.iloc[:,:-1]
ConsuY = Consu.iloc[:,-1]



# Principal Component Analysis
    
X = ConsuXs
y = ConsuY



# Scikit-learn PCA
pca = PCA()

# Scale and transform data to get Principal Components

X_reduced = pca.fit_transform(scale(X))

# Variance (% cumulative) explained by the principal components

np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

# Standardizing

X_std = StandardScaler().fit_transform(X)



# 1 - Eigendecomposition - Computing Eigenvectors and Eigenvalues


mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)


cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# 2 - Selecting Principal Components

for ev in eig_vecs.T:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')



# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])



# Explained Variance (Cummulative)
    
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.plot(cum_var_exp)



print(f'{cum_var_exp[40].real:f}')


#%%
# Choosing first 80 components out of 188 (explain 98% of variance)

matrix_w = np.hstack((
eig_pairs[0][1].reshape(138,1),
eig_pairs[1][1].reshape(138,1),
eig_pairs[2][1].reshape(138,1),
eig_pairs[3][1].reshape(138,1),
eig_pairs[4][1].reshape(138,1),
eig_pairs[5][1].reshape(138,1),
eig_pairs[6][1].reshape(138,1),
eig_pairs[7][1].reshape(138,1),
eig_pairs[8][1].reshape(138,1),
eig_pairs[9][1].reshape(138,1),
eig_pairs[10][1].reshape(138,1),
eig_pairs[11][1].reshape(138,1),
eig_pairs[12][1].reshape(138,1),
eig_pairs[13][1].reshape(138,1),
eig_pairs[14][1].reshape(138,1),
eig_pairs[15][1].reshape(138,1),
eig_pairs[16][1].reshape(138,1),
eig_pairs[17][1].reshape(138,1),
eig_pairs[18][1].reshape(138,1),
eig_pairs[19][1].reshape(138,1),
eig_pairs[20][1].reshape(138,1),
eig_pairs[21][1].reshape(138,1),
eig_pairs[22][1].reshape(138,1),
eig_pairs[23][1].reshape(138,1),
eig_pairs[24][1].reshape(138,1),
eig_pairs[25][1].reshape(138,1),
eig_pairs[26][1].reshape(138,1),
eig_pairs[27][1].reshape(138,1),
eig_pairs[28][1].reshape(138,1),
eig_pairs[29][1].reshape(138,1),
eig_pairs[30][1].reshape(138,1),
eig_pairs[31][1].reshape(138,1),
eig_pairs[32][1].reshape(138,1),
eig_pairs[33][1].reshape(138,1),
eig_pairs[34][1].reshape(138,1),
eig_pairs[35][1].reshape(138,1),
eig_pairs[36][1].reshape(138,1),
eig_pairs[37][1].reshape(138,1),
eig_pairs[38][1].reshape(138,1),
eig_pairs[39][1].reshape(138,1),
eig_pairs[40][1].reshape(138,1)
))


matrix_wdf = pd.DataFrame(matrix_w.real)


#%%
# Multiplying values alpha values of components with Xs
Y = X_std.dot(matrix_wdf)
Ydf = pd.DataFrame(Y)


# Joining 

ConsuReg = pd.merge(Ydf, pd.DataFrame(y), how='left', left_index=True, right_index=True)




# Regession
regX = ConsuReg.iloc[:, :-1].values  
regY = ConsuReg.iloc[:, -1].values


regX = np.delete(regX,[6,8,11,17,18,23,26,34,37,40], axis=1)



# Splitting this data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(regX, regY, test_size=0.1, random_state=0)  

# Training the algorithm
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 


# Making Prediction
y_pred = regressor.predict(X_test)  


X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())



Compdf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(Compdf)


# Evaluate

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  



#%%

coeff_df = pd.DataFrame(regressor.coef_, pd.DataFrame(regX).columns, columns=['Coefficient'])  


#%%
Beta = pd.DataFrame(regressor.intercept_,range(1),columns=['Coefficient']).append(coeff_df)

#%%
cols = [6,8,11,17,18,23,26,34,37,40]
matrix_wdf.drop(matrix_wdf.columns[cols],axis=1,inplace=True)
#%%
Ydf.drop(Ydf.columns[cols],axis=1,inplace=True)



#%%


ConsuFinal = pd.ExcelWriter('ConsuFinal.xlsx')
ConsuF.to_excel(ConsuFinal,'Final Data')
matrix_wdf.to_excel(ConsuFinal,'alpha values')
Ydf.to_excel(ConsuFinal,'Component values')
Beta.to_excel(ConsuFinal,'Beta values')
ConsuFinal.save()

#%%




