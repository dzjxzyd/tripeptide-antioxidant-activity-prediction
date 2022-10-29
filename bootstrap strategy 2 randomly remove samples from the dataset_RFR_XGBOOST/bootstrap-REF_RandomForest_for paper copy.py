### input
import os
os.getcwd()
#### change your current working directory
os.chdir('/Users/zhenjiaodu/Desktop/python_notebook/1.QSAR_STAT/data')
import pandas as pd
import numpy as np
raw_ABTS= pd.read_excel('ABTS.xlsx') # read data

peptide_name = raw_ABTS['peptide_name'].to_numpy() # transform the peptide name into a numpy array
peptide_activty = raw_ABTS['activity'].to_numpy()# transform the peptide activity into a numpy array
#raw_ABTS
features= pd.read_csv('aa_553_washed_features.csv',header=0,index_col=0)
features= pd.read_csv('aa_pearson_0.95_342_features.csv',header=0,index_col=0)
### transform peptide_name into a vector for model development
# creat a feature_for_model using the features to encode the peptide_name
sample_size = len(peptide_name) # get the peptide sampel size
peptide_length = len(peptide_name[0])
feature_dimension = np.shape(features)[0] # get dimension of the features
# create matirc for feature extraction
feature_for_model = np.empty([sample_size, peptide_length*feature_dimension])
np.shape(feature_for_model) # confirm the feature_for_model matrix dimenstion

for i in range(len(peptide_name)):
    name = peptide_name[i] # extract the peptide name; maybe a tripeptide
    try:
        first_aa = features[name[0]].to_numpy()
        second_aa = features[name[1]].to_numpy()
        third_aa = features[name[2]].to_numpy()
    except KeyError:
        pass
    # combine them together
    feature_for_model[i]= np.concatenate((first_aa,second_aa,third_aa), axis=0)
raw_ABTS['activity'].shape

X= feature_for_model
y = peptide_activty
X
import pandas as pd

from sklearn.preprocessing import StandardScaler
sc = StandardScaler() #
X= sc.fit_transform(X)

from dataset_split import dataset_split
X_train, X_test, y_train,  y_test = dataset_split(X,y)
X_train.shape


X=pd.DataFrame(X)

### wrapper method, much accurate than filter medthod
### more computationally expensive
# Backward Elimination, Forward Selection, Bidirectional Elimination and RFE

#ii. RFE (Recursive Feature Elimination)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
##### directly get the expected feature number
# choose a model
#note the model must have the attributes of
# set the parameters
#

##### directly get the expected feature number
# choose a model
#note the model must have the attributes of
# set the parameters
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X,y)
model.score(X,y)#0.9840465475972281
model.score(X_rfe,y)
model.fit(X_rfe,y)
# use the optimal feature number to get the final set of features given by RFE method
cols = list(X.columns)
model = RandomForestRegressor()
#Initializing RFE model
rfe = RFE(model, n_features_to_select=15)  # use the optimal feature number
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)
model
#Fitting the data to model
model.fit(X_rfe,y)
model.score(X_rfe,y)#0.9832851295478349

temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
#Int64Index([33, 83, 319, 341, 473, 497, 787, 793, 805, 814, 975, 976, 978, 991,
#            1011],
#           dtype='int64')
## extract the features
a=[]
name = []
# get the extracted feature number
for i in range(15):
    a.append(selected_features_rfe[i])
# extract the features name features
for i in range(342):
    name.append(features.index[i])
# matching the feature number and the feature name
sum = []
for i in range(15):
    if a[i]<342:
        sum.append(name[a[i]])
    if 341<a[i]<684:
        sum.append(name[a[i]-342])
    if a[i]>683:
        sum.append(name[a[i]-684])
print(sum)
#['CHOP780215', 'ISOY800108', 'GEOR030105', 'KARS160122', 'OOBM850103', 'QIAN880102', 'LIFS790103', 'MCMT640101', 'NAKH920102', 'OOBM850102', 'PARS000101', 'PARS000102', 'FODM020101', 'MITS020101', 'DIGM050101']

pd.DataFrame(sum).to_csv('RFE_RF_NAME.csv')
pd.DataFrame(model.feature_importances_).to_csv('RFE_RF_fearue_importance.csv')



#
#

### connected them
X_new = X_rfe
X_new.shape
# dateset seperation, randomly division
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
X_shuffle, y_shuffle = shuffle(X_new, y,random_state=1)
X_important_train, X_important_test, y_train, y_test = train_test_split(X_shuffle, y_shuffle, test_size=0.25,random_state = 1)


#### evaluation start
####
####
#### set a list for the data collection
col_R_train = []
col_MSE_train = []

col_R_cos = []
col_MSE_cos= []

col_R_test = []
col_MSE_test=[]
#### test begin
#### 1. xgbtree
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
##### cross validation
rf = xgb.XGBRegressor( max_depth=2,n_estimators=200,reg_alpha= 1,  reg_lambda = 80)
model = rf
model.fit(X_important_train, y_train)
train_score = model.score(X_important_train, y_train)
train_score
y_train_pre = model.predict(X_important_train)
train_MSE= mean_squared_error(y_train, y_train_pre)
train_MSE
test_score= model.score(X_important_test, y_test)
test_score
y_test_pre = model.predict(X_important_test)
test_MSE=mean_squared_error(y_test, y_test_pre)
test_MSE
y_cross_val_pred = []
y_cross_val_test=[]
#model = xgb.XGBRegressor()
for i in range(X_important_train.shape[0]):
    rest_X_train = np.delete(X_important_train,i,axis=0)
    rest_y_train = np.delete(y_train,i,axis=0)
    # jsut need to transform the np.array shape
    # the shape of directly extracted X_important_train[i] is (3,) not (1,3)
    #   which can not be used for prediction by the rf.model
    out_X = np.empty(([1,15]))
    out_X[0][0] = X_important_train[i][0]
    out_X[0][1] = X_important_train[i][1]
    out_X[0][2] = X_important_train[i][2]
    out_X[0][3] = X_important_train[i][3]
    out_X[0][4] = X_important_train[i][4]
    out_X[0][5] = X_important_train[i][5]
    out_X[0][6] = X_important_train[i][6]
    out_X[0][7] = X_important_train[i][7]
    out_X[0][8] = X_important_train[i][8]
    out_X[0][9] = X_important_train[i][9]
    out_X[0][10] = X_important_train[i][10]
    out_X[0][11] = X_important_train[i][11]
    out_X[0][12] = X_important_train[i][12]
    out_X[0][13] = X_important_train[i][13]
    out_X[0][14] = X_important_train[i][14]
    model.fit(rest_X_train, rest_y_train)
    results = model.predict(out_X)
    y_cross_val_pred.append(results[0])
    y_cross_val_test.append(y_train[i])
from sklearn.metrics import r2_score, mean_squared_error
r2_cross_val=r2_score(y_cross_val_test, y_cross_val_pred)
r2_cross_val
mse_cross_val=mean_squared_error(y_cross_val_test, y_cross_val_pred)
mse_cross_val

##
##assign the values
##
col_R_train.append(train_score)
col_MSE_train.append(train_MSE)
col_R_cos.append(r2_cross_val)
col_MSE_cos.append(mse_cross_val)
col_R_test.append(test_score)
col_MSE_test.append(test_MSE)





#2. xgb tree based
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
rf = xgb.XGBRegressor( booster = 'gblinear',learning_rate=0.3,n_estimators=180,reg_alpha= 0,  reg_lambda = 0.1)
model = rf
model.fit(X_important_train, y_train)
train_score = model.score(X_important_train, y_train)
train_score
y_train_pre = model.predict(X_important_train)
train_MSE= mean_squared_error(y_train, y_train_pre)
train_MSE
test_score= model.score(X_important_test, y_test)
test_score
y_test_pre = model.predict(X_important_test)
test_MSE=mean_squared_error(y_test, y_test_pre)
test_MSE
y_cross_val_pred = []
y_cross_val_test=[]
#model = xgb.XGBRegressor()
for i in range(X_important_train.shape[0]):
    rest_X_train = np.delete(X_important_train,i,axis=0)
    rest_y_train = np.delete(y_train,i,axis=0)
    # jsut need to transform the np.array shape
    # the shape of directly extracted X_important_train[i] is (3,) not (1,3)
    #   which can not be used for prediction by the rf.model
    out_X = np.empty(([1,15]))
    out_X[0][0] = X_important_train[i][0]
    out_X[0][1] = X_important_train[i][1]
    out_X[0][2] = X_important_train[i][2]
    out_X[0][3] = X_important_train[i][3]
    out_X[0][4] = X_important_train[i][4]
    out_X[0][5] = X_important_train[i][5]
    out_X[0][6] = X_important_train[i][6]
    out_X[0][7] = X_important_train[i][7]
    out_X[0][8] = X_important_train[i][8]
    out_X[0][9] = X_important_train[i][9]
    out_X[0][10] = X_important_train[i][10]
    out_X[0][11] = X_important_train[i][11]
    out_X[0][12] = X_important_train[i][12]
    out_X[0][13] = X_important_train[i][13]
    out_X[0][14] = X_important_train[i][14]
    model.fit(rest_X_train, rest_y_train)
    results = model.predict(out_X)
    y_cross_val_pred.append(results[0])
    y_cross_val_test.append(y_train[i])
from sklearn.metrics import r2_score, mean_squared_error
r2_cross_val=r2_score(y_cross_val_test, y_cross_val_pred)
r2_cross_val
mse_cross_val=mean_squared_error(y_cross_val_test, y_cross_val_pred)
mse_cross_val
##
##assign the values
##
col_R_train.append(train_score)
col_MSE_train.append(train_MSE)
col_R_cos.append(r2_cross_val)
col_MSE_cos.append(mse_cross_val)
col_R_test.append(test_score)
col_MSE_test.append(test_MSE)

col_R_test


### 3. random forest\
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(bootstrap=True, n_estimators=300,  max_depth=3,random_state = 0)
model = rfr
model.fit(X_important_train, y_train)
train_score = model.score(X_important_train, y_train)
train_score
y_train_pre = model.predict(X_important_train)
train_MSE= mean_squared_error(y_train, y_train_pre)
train_MSE
test_score= model.score(X_important_test, y_test)
test_score
y_test_pre = model.predict(X_important_test)
test_MSE=mean_squared_error(y_test, y_test_pre)
test_MSE
y_cross_val_pred = []
y_cross_val_test=[]
#model = xgb.XGBRegressor()
for i in range(X_important_train.shape[0]):
    rest_X_train = np.delete(X_important_train,i,axis=0)
    rest_y_train = np.delete(y_train,i,axis=0)
    # jsut need to transform the np.array shape
    # the shape of directly extracted X_important_train[i] is (3,) not (1,3)
    #   which can not be used for prediction by the rf.model
    out_X = np.empty(([1,15]))
    out_X[0][0] = X_important_train[i][0]
    out_X[0][1] = X_important_train[i][1]
    out_X[0][2] = X_important_train[i][2]
    out_X[0][3] = X_important_train[i][3]
    out_X[0][4] = X_important_train[i][4]
    out_X[0][5] = X_important_train[i][5]
    out_X[0][6] = X_important_train[i][6]
    out_X[0][7] = X_important_train[i][7]
    out_X[0][8] = X_important_train[i][8]
    out_X[0][9] = X_important_train[i][9]
    out_X[0][10] = X_important_train[i][10]
    out_X[0][11] = X_important_train[i][11]
    out_X[0][12] = X_important_train[i][12]
    out_X[0][13] = X_important_train[i][13]
    out_X[0][14] = X_important_train[i][14]
    model.fit(rest_X_train, rest_y_train)
    results = model.predict(out_X)
    y_cross_val_pred.append(results[0])
    y_cross_val_test.append(y_train[i])
from sklearn.metrics import r2_score, mean_squared_error
r2_cross_val=r2_score(y_cross_val_test, y_cross_val_pred)
r2_cross_val
mse_cross_val=mean_squared_error(y_cross_val_test, y_cross_val_pred)
mse_cross_val
##
##assign the values
##
col_R_train.append(train_score)
col_MSE_train.append(train_MSE)
col_R_cos.append(r2_cross_val)
col_MSE_cos.append(mse_cross_val)
col_R_test.append(test_score)
col_MSE_test.append(test_MSE)



# 4. Gradient boosting
from sklearn import ensemble
params = {
    "n_estimators": 200,
    "max_depth": 2,
    "min_samples_split": 2,
    "learning_rate": 0.06
        }
        # 这一组参数不如default的好
reg = ensemble.GradientBoostingRegressor(**params).fit(X_important_train, y_train)
#reg = ensemble.GradientBoostingRegressor().fit(X_important_train, y_train)
model = reg
model.fit(X_important_train, y_train)
train_score = model.score(X_important_train, y_train)
train_score
y_train_pre = model.predict(X_important_train)
train_MSE= mean_squared_error(y_train, y_train_pre)
train_MSE
test_score= model.score(X_important_test, y_test)
test_score
y_test_pre = model.predict(X_important_test)
test_MSE=mean_squared_error(y_test, y_test_pre)
test_MSE
y_cross_val_pred = []
y_cross_val_test=[]
#model = xgb.XGBRegressor()
for i in range(X_important_train.shape[0]):
    rest_X_train = np.delete(X_important_train,i,axis=0)
    rest_y_train = np.delete(y_train,i,axis=0)
    # jsut need to transform the np.array shape
    # the shape of directly extracted X_important_train[i] is (3,) not (1,3)
    #   which can not be used for prediction by the rf.model
    out_X = np.empty(([1,15]))
    out_X[0][0] = X_important_train[i][0]
    out_X[0][1] = X_important_train[i][1]
    out_X[0][2] = X_important_train[i][2]
    out_X[0][3] = X_important_train[i][3]
    out_X[0][4] = X_important_train[i][4]
    out_X[0][5] = X_important_train[i][5]
    out_X[0][6] = X_important_train[i][6]
    out_X[0][7] = X_important_train[i][7]
    out_X[0][8] = X_important_train[i][8]
    out_X[0][9] = X_important_train[i][9]
    out_X[0][10] = X_important_train[i][10]
    out_X[0][11] = X_important_train[i][11]
    out_X[0][12] = X_important_train[i][12]
    out_X[0][13] = X_important_train[i][13]
    out_X[0][14] = X_important_train[i][14]
    model.fit(rest_X_train, rest_y_train)
    results = model.predict(out_X)
    y_cross_val_pred.append(results[0])
    y_cross_val_test.append(y_train[i])
from sklearn.metrics import r2_score, mean_squared_error
r2_cross_val=r2_score(y_cross_val_test, y_cross_val_pred)
r2_cross_val
mse_cross_val=mean_squared_error(y_cross_val_test, y_cross_val_pred)
mse_cross_val
##
##assign the values
##
col_R_train.append(train_score)
col_MSE_train.append(train_MSE)
col_R_cos.append(r2_cross_val)
col_MSE_cos.append(mse_cross_val)
col_R_test.append(test_score)
col_MSE_test.append(test_MSE)


### 5. MLP
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error
est = make_pipeline(QuantileTransformer(),
                    MLPRegressor(hidden_layer_sizes=(15,15),
                                    activation = 'relu',
                                    solver= 'adam' ,
                                 learning_rate_init=0.01,
                                 max_iter = 10000,
                                 early_stopping=False))
model = est
model.fit(X_important_train, y_train)
train_score = model.score(X_important_train, y_train)
train_score
y_train_pre = model.predict(X_important_train)
train_MSE= mean_squared_error(y_train, y_train_pre)
train_MSE
test_score= model.score(X_important_test, y_test)
test_score
y_test_pre = model.predict(X_important_test)
test_MSE=mean_squared_error(y_test, y_test_pre)
test_MSE
y_cross_val_pred = []
y_cross_val_test=[]
#model = xgb.XGBRegressor()
for i in range(X_important_train.shape[0]):
    rest_X_train = np.delete(X_important_train,i,axis=0)
    rest_y_train = np.delete(y_train,i,axis=0)
    # jsut need to transform the np.array shape
    # the shape of directly extracted X_important_train[i] is (3,) not (1,3)
    #   which can not be used for prediction by the rf.model
    out_X = np.empty(([1,15]))
    out_X[0][0] = X_important_train[i][0]
    out_X[0][1] = X_important_train[i][1]
    out_X[0][2] = X_important_train[i][2]
    out_X[0][3] = X_important_train[i][3]
    out_X[0][4] = X_important_train[i][4]
    out_X[0][5] = X_important_train[i][5]
    out_X[0][6] = X_important_train[i][6]
    out_X[0][7] = X_important_train[i][7]
    out_X[0][8] = X_important_train[i][8]
    out_X[0][9] = X_important_train[i][9]
    out_X[0][10] = X_important_train[i][10]
    out_X[0][11] = X_important_train[i][11]
    out_X[0][12] = X_important_train[i][12]
    out_X[0][13] = X_important_train[i][13]
    out_X[0][14] = X_important_train[i][14]
    model.fit(rest_X_train, rest_y_train)
    results = model.predict(out_X)
    y_cross_val_pred.append(results[0])
    y_cross_val_test.append(y_train[i])
from sklearn.metrics import r2_score, mean_squared_error
r2_cross_val=r2_score(y_cross_val_test, y_cross_val_pred)
r2_cross_val
mse_cross_val=mean_squared_error(y_cross_val_test, y_cross_val_pred)
mse_cross_val
##
##assign the values
##
col_R_train.append(train_score)
col_MSE_train.append(train_MSE)
col_R_cos.append(r2_cross_val)
col_MSE_cos.append(mse_cross_val)
col_R_test.append(test_score)
col_MSE_test.append(test_MSE)

###6. KNN
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=2,p=1).fit(X_important_train, y_train)
model = neigh
model.fit(X_important_train, y_train)
train_score = model.score(X_important_train, y_train)
train_score
y_train_pre = model.predict(X_important_train)
train_MSE= mean_squared_error(y_train, y_train_pre)
train_MSE
test_score= model.score(X_important_test, y_test)
test_score
y_test_pre = model.predict(X_important_test)
test_MSE=mean_squared_error(y_test, y_test_pre)
test_MSE
y_cross_val_pred = []
y_cross_val_test=[]
#model = xgb.XGBRegressor()
for i in range(X_important_train.shape[0]):
    rest_X_train = np.delete(X_important_train,i,axis=0)
    rest_y_train = np.delete(y_train,i,axis=0)
    # jsut need to transform the np.array shape
    # the shape of directly extracted X_important_train[i] is (3,) not (1,3)
    #   which can not be used for prediction by the rf.model
    out_X = np.empty(([1,15]))
    out_X[0][0] = X_important_train[i][0]
    out_X[0][1] = X_important_train[i][1]
    out_X[0][2] = X_important_train[i][2]
    out_X[0][3] = X_important_train[i][3]
    out_X[0][4] = X_important_train[i][4]
    out_X[0][5] = X_important_train[i][5]
    out_X[0][6] = X_important_train[i][6]
    out_X[0][7] = X_important_train[i][7]
    out_X[0][8] = X_important_train[i][8]
    out_X[0][9] = X_important_train[i][9]
    out_X[0][10] = X_important_train[i][10]
    out_X[0][11] = X_important_train[i][11]
    out_X[0][12] = X_important_train[i][12]
    out_X[0][13] = X_important_train[i][13]
    out_X[0][14] = X_important_train[i][14]
    model.fit(rest_X_train, rest_y_train)
    results = model.predict(out_X)
    y_cross_val_pred.append(results[0])
    y_cross_val_test.append(y_train[i])
from sklearn.metrics import r2_score, mean_squared_error
r2_cross_val=r2_score(y_cross_val_test, y_cross_val_pred)
r2_cross_val
mse_cross_val=mean_squared_error(y_cross_val_test, y_cross_val_pred)
mse_cross_val
##
##assign the values
##
col_R_train.append(train_score)
col_MSE_train.append(train_MSE)
col_R_cos.append(r2_cross_val)
col_MSE_cos.append(mse_cross_val)
col_R_test.append(test_score)
col_MSE_test.append(test_MSE)
col_R_test
# 7. SVR_rbf
from sklearn.svm import SVR
# import model and set its parameters
#
model_SVR_linear = SVR(kernel='rbf',degree = 3,C= 20).fit(X_important_train, y_train)
model = model_SVR_linear
model.fit(X_important_train, y_train)
train_score = model.score(X_important_train, y_train)
train_score
y_train_pre = model.predict(X_important_train)
train_MSE= mean_squared_error(y_train, y_train_pre)
train_MSE
test_score= model.score(X_important_test, y_test)
test_score
y_test_pre = model.predict(X_important_test)
test_MSE=mean_squared_error(y_test, y_test_pre)
test_MSE
y_cross_val_pred = []
y_cross_val_test=[]
#model = xgb.XGBRegressor()
for i in range(X_important_train.shape[0]):
    rest_X_train = np.delete(X_important_train,i,axis=0)
    rest_y_train = np.delete(y_train,i,axis=0)
    # jsut need to transform the np.array shape
    # the shape of directly extracted X_important_train[i] is (3,) not (1,3)
    #   which can not be used for prediction by the rf.model
    out_X = np.empty(([1,15]))
    out_X[0][0] = X_important_train[i][0]
    out_X[0][1] = X_important_train[i][1]
    out_X[0][2] = X_important_train[i][2]
    out_X[0][3] = X_important_train[i][3]
    out_X[0][4] = X_important_train[i][4]
    out_X[0][5] = X_important_train[i][5]
    out_X[0][6] = X_important_train[i][6]
    out_X[0][7] = X_important_train[i][7]
    out_X[0][8] = X_important_train[i][8]
    out_X[0][9] = X_important_train[i][9]
    out_X[0][10] = X_important_train[i][10]
    out_X[0][11] = X_important_train[i][11]
    out_X[0][12] = X_important_train[i][12]
    out_X[0][13] = X_important_train[i][13]
    out_X[0][14] = X_important_train[i][14]
    model.fit(rest_X_train, rest_y_train)
    results = model.predict(out_X)
    y_cross_val_pred.append(results[0])
    y_cross_val_test.append(y_train[i])
from sklearn.metrics import r2_score, mean_squared_error
r2_cross_val=r2_score(y_cross_val_test, y_cross_val_pred)
r2_cross_val
mse_cross_val=mean_squared_error(y_cross_val_test, y_cross_val_pred)
mse_cross_val
##
##assign the values
##
col_R_train.append(train_score)
col_MSE_train.append(train_MSE)
col_R_cos.append(r2_cross_val)
col_MSE_cos.append(mse_cross_val)
col_R_test.append(test_score)
col_MSE_test.append(test_MSE)

# 8. svr_linear
model_SVR_linear = SVR(kernel='linear',C= 1).fit(X_important_train, y_train)
model = model_SVR_linear
model.fit(X_important_train, y_train)
train_score = model.score(X_important_train, y_train)
train_score
y_train_pre = model.predict(X_important_train)
train_MSE= mean_squared_error(y_train, y_train_pre)
train_MSE
test_score= model.score(X_important_test, y_test)
test_score
y_test_pre = model.predict(X_important_test)
test_MSE=mean_squared_error(y_test, y_test_pre)
test_MSE
y_cross_val_pred = []
y_cross_val_test=[]
#model = xgb.XGBRegressor()
for i in range(X_important_train.shape[0]):
    rest_X_train = np.delete(X_important_train,i,axis=0)
    rest_y_train = np.delete(y_train,i,axis=0)
    # jsut need to transform the np.array shape
    # the shape of directly extracted X_important_train[i] is (3,) not (1,3)
    #   which can not be used for prediction by the rf.model
    out_X = np.empty(([1,15]))
    out_X[0][0] = X_important_train[i][0]
    out_X[0][1] = X_important_train[i][1]
    out_X[0][2] = X_important_train[i][2]
    out_X[0][3] = X_important_train[i][3]
    out_X[0][4] = X_important_train[i][4]
    out_X[0][5] = X_important_train[i][5]
    out_X[0][6] = X_important_train[i][6]
    out_X[0][7] = X_important_train[i][7]
    out_X[0][8] = X_important_train[i][8]
    out_X[0][9] = X_important_train[i][9]
    out_X[0][10] = X_important_train[i][10]
    out_X[0][11] = X_important_train[i][11]
    out_X[0][12] = X_important_train[i][12]
    out_X[0][13] = X_important_train[i][13]
    out_X[0][14] = X_important_train[i][14]
    model.fit(rest_X_train, rest_y_train)
    results = model.predict(out_X)
    y_cross_val_pred.append(results[0])
    y_cross_val_test.append(y_train[i])
from sklearn.metrics import r2_score, mean_squared_error
r2_cross_val=r2_score(y_cross_val_test, y_cross_val_pred)
r2_cross_val
mse_cross_val=mean_squared_error(y_cross_val_test, y_cross_val_pred)
mse_cross_val
##
##assign the values
##
col_R_train.append(train_score)
col_MSE_train.append(train_MSE)
col_R_cos.append(r2_cross_val)
col_MSE_cos.append(mse_cross_val)
col_R_test.append(test_score)
col_MSE_test.append(test_MSE)

## 9. lassco
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
# lassi
model_Lassco = Lasso(alpha=0.01,tol = 1e-3)
model = model_Lassco
model.fit(X_important_train, y_train)
train_score = model.score(X_important_train, y_train)
train_score
y_train_pre = model.predict(X_important_train)
train_MSE= mean_squared_error(y_train, y_train_pre)
train_MSE
test_score= model.score(X_important_test, y_test)
test_score
y_test_pre = model.predict(X_important_test)
test_MSE=mean_squared_error(y_test, y_test_pre)
test_MSE
y_cross_val_pred = []
y_cross_val_test=[]
#model = xgb.XGBRegressor()
for i in range(X_important_train.shape[0]):
    rest_X_train = np.delete(X_important_train,i,axis=0)
    rest_y_train = np.delete(y_train,i,axis=0)
    # jsut need to transform the np.array shape
    # the shape of directly extracted X_important_train[i] is (3,) not (1,3)
    #   which can not be used for prediction by the rf.model
    out_X = np.empty(([1,15]))
    out_X[0][0] = X_important_train[i][0]
    out_X[0][1] = X_important_train[i][1]
    out_X[0][2] = X_important_train[i][2]
    out_X[0][3] = X_important_train[i][3]
    out_X[0][4] = X_important_train[i][4]
    out_X[0][5] = X_important_train[i][5]
    out_X[0][6] = X_important_train[i][6]
    out_X[0][7] = X_important_train[i][7]
    out_X[0][8] = X_important_train[i][8]
    out_X[0][9] = X_important_train[i][9]
    out_X[0][10] = X_important_train[i][10]
    out_X[0][11] = X_important_train[i][11]
    out_X[0][12] = X_important_train[i][12]
    out_X[0][13] = X_important_train[i][13]
    out_X[0][14] = X_important_train[i][14]
    model.fit(rest_X_train, rest_y_train)
    results = model.predict(out_X)
    y_cross_val_pred.append(results[0])
    y_cross_val_test.append(y_train[i])
from sklearn.metrics import r2_score, mean_squared_error
r2_cross_val=r2_score(y_cross_val_test, y_cross_val_pred)
r2_cross_val
mse_cross_val=mean_squared_error(y_cross_val_test, y_cross_val_pred)
mse_cross_val
##
##assign the values
##
col_R_train.append(train_score)
col_MSE_train.append(train_MSE)
col_R_cos.append(r2_cross_val)
col_MSE_cos.append(mse_cross_val)
col_R_test.append(test_score)
col_MSE_test.append(test_MSE)



# 10. Ridge
model_ridge = Ridge(alpha=1,tol = 1e-1)
model = model_ridge
model.fit(X_important_train, y_train)
train_score = model.score(X_important_train, y_train)
train_score
y_train_pre = model.predict(X_important_train)
train_MSE= mean_squared_error(y_train, y_train_pre)
train_MSE
test_score= model.score(X_important_test, y_test)
test_score
y_test_pre = model.predict(X_important_test)
test_MSE=mean_squared_error(y_test, y_test_pre)
test_MSE
y_cross_val_pred = []
y_cross_val_test=[]
#model = xgb.XGBRegressor()
for i in range(X_important_train.shape[0]):
    rest_X_train = np.delete(X_important_train,i,axis=0)
    rest_y_train = np.delete(y_train,i,axis=0)
    # jsut need to transform the np.array shape
    # the shape of directly extracted X_important_train[i] is (3,) not (1,3)
    #   which can not be used for prediction by the rf.model
    out_X = np.empty(([1,15]))
    out_X[0][0] = X_important_train[i][0]
    out_X[0][1] = X_important_train[i][1]
    out_X[0][2] = X_important_train[i][2]
    out_X[0][3] = X_important_train[i][3]
    out_X[0][4] = X_important_train[i][4]
    out_X[0][5] = X_important_train[i][5]
    out_X[0][6] = X_important_train[i][6]
    out_X[0][7] = X_important_train[i][7]
    out_X[0][8] = X_important_train[i][8]
    out_X[0][9] = X_important_train[i][9]
    out_X[0][10] = X_important_train[i][10]
    out_X[0][11] = X_important_train[i][11]
    out_X[0][12] = X_important_train[i][12]
    out_X[0][13] = X_important_train[i][13]
    out_X[0][14] = X_important_train[i][14]
    model.fit(rest_X_train, rest_y_train)
    results = model.predict(out_X)
    y_cross_val_pred.append(results[0])
    y_cross_val_test.append(y_train[i])
from sklearn.metrics import r2_score, mean_squared_error
r2_cross_val=r2_score(y_cross_val_test, y_cross_val_pred)
r2_cross_val
mse_cross_val=mean_squared_error(y_cross_val_test, y_cross_val_pred)
mse_cross_val
##
##assign the values
##
col_R_train.append(train_score)
col_MSE_train.append(train_MSE)
col_R_cos.append(r2_cross_val)
col_MSE_cos.append(mse_cross_val)
col_R_test.append(test_score)
col_MSE_test.append(test_MSE)

### 11. SGD
#
from sklearn.linear_model import SGDRegressor
reg=SGDRegressor(alpha=0.001, max_iter=10000, tol=1e-5)
model = reg
model.fit(X_important_train, y_train)
train_score = model.score(X_important_train, y_train)
train_score
y_train_pre = model.predict(X_important_train)
train_MSE= mean_squared_error(y_train, y_train_pre)
train_MSE
test_score= model.score(X_important_test, y_test)
test_score
y_test_pre = model.predict(X_important_test)
test_MSE=mean_squared_error(y_test, y_test_pre)
test_MSE
y_cross_val_pred = []
y_cross_val_test=[]
#model = xgb.XGBRegressor()
for i in range(X_important_train.shape[0]):
    rest_X_train = np.delete(X_important_train,i,axis=0)
    rest_y_train = np.delete(y_train,i,axis=0)
    # jsut need to transform the np.array shape
    # the shape of directly extracted X_important_train[i] is (3,) not (1,3)
    #   which can not be used for prediction by the rf.model
    out_X = np.empty(([1,15]))
    out_X[0][0] = X_important_train[i][0]
    out_X[0][1] = X_important_train[i][1]
    out_X[0][2] = X_important_train[i][2]
    out_X[0][3] = X_important_train[i][3]
    out_X[0][4] = X_important_train[i][4]
    out_X[0][5] = X_important_train[i][5]
    out_X[0][6] = X_important_train[i][6]
    out_X[0][7] = X_important_train[i][7]
    out_X[0][8] = X_important_train[i][8]
    out_X[0][9] = X_important_train[i][9]
    out_X[0][10] = X_important_train[i][10]
    out_X[0][11] = X_important_train[i][11]
    out_X[0][12] = X_important_train[i][12]
    out_X[0][13] = X_important_train[i][13]
    out_X[0][14] = X_important_train[i][14]
    model.fit(rest_X_train, rest_y_train)
    results = model.predict(out_X)
    y_cross_val_pred.append(results[0])
    y_cross_val_test.append(y_train[i])
from sklearn.metrics import r2_score, mean_squared_error
r2_cross_val=r2_score(y_cross_val_test, y_cross_val_pred)
r2_cross_val
mse_cross_val=mean_squared_error(y_cross_val_test, y_cross_val_pred)
mse_cross_val
##
##assign the values
##
col_R_train.append(train_score)
col_MSE_train.append(train_MSE)
col_R_cos.append(r2_cross_val)
col_MSE_cos.append(mse_cross_val)
col_R_test.append(test_score)
col_MSE_test.append(test_MSE)



# 12. kernel ridge
#
from sklearn.kernel_ridge import KernelRidge

kr=KernelRidge(alpha = 1,degree=2).fit(X_important_train, y_train)
model = kr
model.fit(X_important_train, y_train)
train_score = model.score(X_important_train, y_train)
train_score
y_train_pre = model.predict(X_important_train)
train_MSE= mean_squared_error(y_train, y_train_pre)
train_MSE
test_score= model.score(X_important_test, y_test)
test_score
y_test_pre = model.predict(X_important_test)
test_MSE=mean_squared_error(y_test, y_test_pre)
test_MSE
y_cross_val_pred = []
y_cross_val_test=[]
#model = xgb.XGBRegressor()
for i in range(X_important_train.shape[0]):
    rest_X_train = np.delete(X_important_train,i,axis=0)
    rest_y_train = np.delete(y_train,i,axis=0)
    # jsut need to transform the np.array shape
    # the shape of directly extracted X_important_train[i] is (3,) not (1,3)
    #   which can not be used for prediction by the rf.model
    out_X = np.empty(([1,15]))
    out_X[0][0] = X_important_train[i][0]
    out_X[0][1] = X_important_train[i][1]
    out_X[0][2] = X_important_train[i][2]
    out_X[0][3] = X_important_train[i][3]
    out_X[0][4] = X_important_train[i][4]
    out_X[0][5] = X_important_train[i][5]
    out_X[0][6] = X_important_train[i][6]
    out_X[0][7] = X_important_train[i][7]
    out_X[0][8] = X_important_train[i][8]
    out_X[0][9] = X_important_train[i][9]
    out_X[0][10] = X_important_train[i][10]
    out_X[0][11] = X_important_train[i][11]
    out_X[0][12] = X_important_train[i][12]
    out_X[0][13] = X_important_train[i][13]
    out_X[0][14] = X_important_train[i][14]
    model.fit(rest_X_train, rest_y_train)
    results = model.predict(out_X)
    y_cross_val_pred.append(results[0])
    y_cross_val_test.append(y_train[i])
from sklearn.metrics import r2_score, mean_squared_error
r2_cross_val=r2_score(y_cross_val_test, y_cross_val_pred)
r2_cross_val
mse_cross_val=mean_squared_error(y_cross_val_test, y_cross_val_pred)
mse_cross_val
##
##assign the values
##
col_R_train.append(train_score)
col_MSE_train.append(train_MSE)
col_R_cos.append(r2_cross_val)
col_MSE_cos.append(mse_cross_val)
col_R_test.append(test_score)
col_MSE_test.append(test_MSE)




#13. HUber
from sklearn.linear_model import HuberRegressor
huber = HuberRegressor(epsilon = 10,max_iter = 1000,alpha = 10).fit(X_important_train, y_train)
model = huber
model.fit(X_important_train, y_train)
train_score = model.score(X_important_train, y_train)
train_score
y_train_pre = model.predict(X_important_train)
train_MSE= mean_squared_error(y_train, y_train_pre)
train_MSE
test_score= model.score(X_important_test, y_test)
test_score
y_test_pre = model.predict(X_important_test)
test_MSE=mean_squared_error(y_test, y_test_pre)
test_MSE
y_cross_val_pred = []
y_cross_val_test=[]
#model = xgb.XGBRegressor()
for i in range(X_important_train.shape[0]):
    rest_X_train = np.delete(X_important_train,i,axis=0)
    rest_y_train = np.delete(y_train,i,axis=0)
    # jsut need to transform the np.array shape
    # the shape of directly extracted X_important_train[i] is (3,) not (1,3)
    #   which can not be used for prediction by the rf.model
    out_X = np.empty(([1,15]))
    out_X[0][0] = X_important_train[i][0]
    out_X[0][1] = X_important_train[i][1]
    out_X[0][2] = X_important_train[i][2]
    out_X[0][3] = X_important_train[i][3]
    out_X[0][4] = X_important_train[i][4]
    out_X[0][5] = X_important_train[i][5]
    out_X[0][6] = X_important_train[i][6]
    out_X[0][7] = X_important_train[i][7]
    out_X[0][8] = X_important_train[i][8]
    out_X[0][9] = X_important_train[i][9]
    out_X[0][10] = X_important_train[i][10]
    out_X[0][11] = X_important_train[i][11]
    out_X[0][12] = X_important_train[i][12]
    out_X[0][13] = X_important_train[i][13]
    out_X[0][14] = X_important_train[i][14]
    model.fit(rest_X_train, rest_y_train)
    results = model.predict(out_X)
    y_cross_val_pred.append(results[0])
    y_cross_val_test.append(y_train[i])
from sklearn.metrics import r2_score, mean_squared_error
r2_cross_val=r2_score(y_cross_val_test, y_cross_val_pred)
r2_cross_val
mse_cross_val=mean_squared_error(y_cross_val_test, y_cross_val_pred)
mse_cross_val
##
##assign the values
##
col_R_train.append(train_score)
col_MSE_train.append(train_MSE)
col_R_cos.append(r2_cross_val)
col_MSE_cos.append(mse_cross_val)
col_R_test.append(test_score)
col_MSE_test.append(test_MSE)




# 14 bagging methods (decision tree)
# you can change the estiminator method as you wish
from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import make_regression
#X, y = make_regression(n_samples=100, n_features=4,
#                       n_informative=2, n_targets=1,
#                       random_state=0, shuffle=False)
#regr = BaggingRegressor(base_estimator=SVR(C=1),
#                        n_estimators=100, random_state=0).fit(X_important_train, y_train)
regr = BaggingRegressor(n_estimators=200,max_features=10, random_state=0).fit(X_important_train, y_train)
model = regr
model.fit(X_important_train, y_train)
train_score = model.score(X_important_train, y_train)
train_score
y_train_pre = model.predict(X_important_train)
train_MSE= mean_squared_error(y_train, y_train_pre)
train_MSE
test_score= model.score(X_important_test, y_test)
test_score
y_test_pre = model.predict(X_important_test)
test_MSE=mean_squared_error(y_test, y_test_pre)
test_MSE
y_cross_val_pred = []
y_cross_val_test=[]
#model = xgb.XGBRegressor()
for i in range(X_important_train.shape[0]):
    rest_X_train = np.delete(X_important_train,i,axis=0)
    rest_y_train = np.delete(y_train,i,axis=0)
    # jsut need to transform the np.array shape
    # the shape of directly extracted X_important_train[i] is (3,) not (1,3)
    #   which can not be used for prediction by the rf.model
    out_X = np.empty(([1,15]))
    out_X[0][0] = X_important_train[i][0]
    out_X[0][1] = X_important_train[i][1]
    out_X[0][2] = X_important_train[i][2]
    out_X[0][3] = X_important_train[i][3]
    out_X[0][4] = X_important_train[i][4]
    out_X[0][5] = X_important_train[i][5]
    out_X[0][6] = X_important_train[i][6]
    out_X[0][7] = X_important_train[i][7]
    out_X[0][8] = X_important_train[i][8]
    out_X[0][9] = X_important_train[i][9]
    out_X[0][10] = X_important_train[i][10]
    out_X[0][11] = X_important_train[i][11]
    out_X[0][12] = X_important_train[i][12]
    out_X[0][13] = X_important_train[i][13]
    out_X[0][14] = X_important_train[i][14]
    model.fit(rest_X_train, rest_y_train)
    results = model.predict(out_X)
    y_cross_val_pred.append(results[0])
    y_cross_val_test.append(y_train[i])
from sklearn.metrics import r2_score, mean_squared_error
r2_cross_val=r2_score(y_cross_val_test, y_cross_val_pred)
r2_cross_val
mse_cross_val=mean_squared_error(y_cross_val_test, y_cross_val_pred)
mse_cross_val
##
##assign the values
##
col_R_train.append(train_score)
col_MSE_train.append(train_MSE)
col_R_cos.append(r2_cross_val)
col_MSE_cos.append(mse_cross_val)
col_R_test.append(test_score)
col_MSE_test.append(test_MSE)

len(col_MSE_test)
one = pd.DataFrame(col_R_train)
two = pd.DataFrame(col_MSE_train)
three = pd.DataFrame(col_R_cos)
four = pd.DataFrame(col_MSE_cos)
five = pd.DataFrame(col_R_test)
six = pd.DataFrame(col_MSE_test)

frame = [one, two, three, four,five,six]

summary = pd.concat(frame,axis=1)
summary.to_csv('new_REF_RandomForest_.csv')
