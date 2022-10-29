### input
import os
os.getcwd()
#### change your current working directory
os.chdir('/Users/zhenjiaodu/Desktop/python_notebook/1.QSAR_STAT/data')
import pandas as pd
import numpy as np

features= pd.read_csv('aa_553_washed_features.csv',header=0,index_col=0)
features.shape
features = features.T # transpose the matrix
features.shape
# the correlation coeffient is only based on the columns
#   therefore, we need to transpose the matrix first and
#               then claculate the correlation coeffient between different
Pearson_corr = features.corr() # results is a pandas.core.frame.DataFrame
Pearson_corr.shape
#Pearson_corr.to_csv('Prearson_correlation.csv')

553 rows × 553 columns
#  get column name out
cols = list(Pearson_corr.columns)
selected_cols =  list(Pearson_corr.columns)

for i in range(553):
    for j in range(553):
        if abs(Pearson_corr[cols[i]][j]) > 0.95: # set the coefficient for selection
            if i != j: # if not at the diagonal
                if cols[j] in selected_cols: # if still not be removed
                    selected_cols.remove(cols[j])

len(selected_cols)
print(selected_cols)

X = features[selected_cols]

X.shape
type(X)
df = X.T
df.to_csv('aa_pearson_0.95_342_features.csv')
