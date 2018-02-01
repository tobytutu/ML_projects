#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
The code is for the project to forcast how many future visitors a restaurant will receive.
https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting

(1) Neural network with Tensorflow
(2) SVM and Random Forest
(3) XGBoost

"""

#%%

import numpy as np
import pandas as pd
from sklearn import preprocessing,metrics
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,train_test_split
from scipy.stats import randint as sp_randint

import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.svm import SVR

from xgboost.sklearn import XGBRegressor

#See example in http://nbviewer.jupyter.org/github/trevorstephens/gplearn/blob/master/doc/gp_examples.ipynb
from gplearn.genetic import SymbolicRegressor
import tensorflow as tf
import os
path='~/Desktop/kaggle/Restaurant vistor forecasting'
os.chdir=path


np.random.seed(565)

data = {
    'tra':   pd.read_csv(path+'/air_visit_data.csv'),
    'as':    pd.read_csv(path+'/air_store_info.csv'),
    'hs':    pd.read_csv(path+'/hpg_store_info.csv'),
    'ar':    pd.read_csv(path+'/air_reserve.csv'),
    'hr':    pd.read_csv(path+'/hpg_reserve.csv'),
    'id':    pd.read_csv(path+'/store_id_relation.csv'),
    'tes':   pd.read_csv(path+'/sample_submission.csv'),
    'hol':   pd.read_csv(path+'/date_info.csv').rename(columns={'calendar_date': 'visit_date' })
}

#%% 
# the following code cleans the dataset for training and testing
### data cleaning part is modified based on the code provided by https://www.kaggle.com/lscoelho/genetic-programming-approach-gplearn
def datapreparation(data):

    data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])
    
    for df in ['ar', 'hr']:
        data[df]['visit_datetime']   = pd.to_datetime(data[df]['visit_datetime'])
        data[df]['visit_datetime']   = data[df]['visit_datetime'].dt.date
        data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
        data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
        data[df]['reserve_datetime_diff'] = data[df].apply(
            lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
            
        #--- begin  new features    
        data[df]['reserve_datetime_diff_2'] = data[df].apply(
            lambda r: ( (r['visit_datetime'] - r['reserve_datetime']).days)**2.1, axis=1)
        data[df]['reserve_datetime_diff_3'] = data[df].apply(
            lambda r: ( (r['visit_datetime'] - r['reserve_datetime']).days)**3.2, axis=1)
        #--- end new features        
            
        data[df] = data[df].groupby(
            ['air_store_id', 'visit_datetime'], as_index=False)[[
                'reserve_datetime_diff', 'reserve_visitors'
            ]].sum().rename(columns={
                'visit_datetime': 'visit_date'
            })
            
        show_data = 0    
        if (show_data==1):
            print(data[df].head())
    
    data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
    data['tra']['dow']        = data['tra']['visit_date'].dt.dayofweek
    data['tra']['year']       = data['tra']['visit_date'].dt.year
    data['tra']['month']      = data['tra']['visit_date'].dt.month
    data['tra']['visit_date'] = data['tra']['visit_date'].dt.date
    
    data['tes']['visit_date'] = data['tes']['id'].map(
        lambda x: str(x).split('_')[2])
    data['tes']['air_store_id'] = data['tes']['id'].map(
        lambda x: '_'.join(x.split('_')[:2]))
    
    
    data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
    data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
    data['tes']['year'] = data['tes']['visit_date'].dt.year
    data['tes']['month'] = data['tes']['visit_date'].dt.month
    data['tes']['visit_date'] = data['tes']['visit_date'].dt.date
    
    unique_stores = data['tes']['air_store_id'].unique()
    stores = pd.concat(
        [
            pd.DataFrame({
                'air_store_id': unique_stores,
                'dow': [i] * len(unique_stores)
            }) for i in range(7)
        ],
        axis=0,
        ignore_index=True).reset_index(drop=True)
    
    #sure it can be compressed...
    tmp = data['tra'].groupby(
        ['air_store_id', 'dow'],
        as_index=False)['visitors'].min().rename(columns={
            'visitors': 'min_visitors'
        })
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
    tmp = data['tra'].groupby(
        ['air_store_id', 'dow'],
        as_index=False)['visitors'].mean().rename(columns={
            'visitors': 'mean_visitors'
        })
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
    tmp = data['tra'].groupby(
        ['air_store_id', 'dow'],
        as_index=False)['visitors'].median().rename(columns={
            'visitors': 'median_visitors'
        })
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
    tmp = data['tra'].groupby(
        ['air_store_id', 'dow'],
        as_index=False)['visitors'].max().rename(columns={
            'visitors': 'max_visitors'
        })
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
    tmp = data['tra'].groupby(
        ['air_store_id', 'dow'],
        as_index=False)['visitors'].count().rename(columns={
            'visitors': 'count_observations'
        })
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
    
    
    stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])
    lbl = preprocessing.LabelEncoder()
    stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
    stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])
    
    data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
    data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
    data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
    
    train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])
    test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date'])
    
    train = pd.merge(data['tra'], stores, how='left', on=['air_store_id', 'dow'])
    test = pd.merge(data['tes'], stores, how='left', on=['air_store_id', 'dow'])
    
    for df in ['ar', 'hr']:
        train = pd.merge(
            train, data[df], how='left', on=['air_store_id', 'visit_date'])
        test = pd.merge(
            test, data[df], how='left', on=['air_store_id', 'visit_date'])
    
    col = [
        c for c in train
        if c not in ['id', 'air_store_id', 'visit_date', 'visitors']
    ]
    train = train.fillna(-1)
    test = test.fillna(-1)
    
    # XGB starter template borrowed from @anokas: https://www.kaggle.com/anokas/simple-xgboost-starter-0-0655
    
    for c, dtype in zip(train.columns, train.dtypes):
        if dtype == np.float64:
            train[c] = train[c].astype(np.float32)
    
    for c, dtype in zip(test.columns, test.dtypes):
        if dtype == np.float64:
            test[c] = test[c].astype(np.float32)
    
    train_x = train.drop(['air_store_id', 'visit_date', 'visitors'], axis=1)
    train_y = np.log1p(train['visitors'].values)
    
    if (show_data==1):
        print(train_x.shape, train_y.shape)
        
    test_x  = test.drop(['id', 'air_store_id', 'visit_date', 'visitors'], axis=1)
    
    return train_x,train_y,test_x


#%% Prepare the training and test datasets for the NN model

train_x,train_y,test_x = datapreparation(data)

train_y_real=np.expm1(train_y)
###### The following code proceses the data for training and testing
train_x1,valid_x1,train_y1,valid_y1=train_test_split(train_x,train_y_real,test_size=0.2,random_state=23)

def processtraindataX(x):
    train1_x=x
    scaler=preprocessing.StandardScaler()
    scaler.fit(train1_x)
    train1_x=scaler.transform(train1_x)
    train_x=train1_x.transpose()
    return train_x

    
tranx=processtraindataX(train_x1)

valid_x=processtraindataX(valid_x1)

trany=np.reshape(train_y1,newshape=[1,-1])

valid_y=np.reshape(valid_y1,newshape=[1,-1])


testx=processtraindataX(test_x)

######### neural network tensorflow initialization

##### !!!! be careful: what's the shape of the train_x !!!!!!!

nx,mx=tranx.shape 

ny,my=trany.shape


learning_rate=0.01
num_epochs=100
display_step=5
minibatch_size=128
seed=23

n_hidden1=256 # 1st layer # of neurons
n_hidden2=10 # 2nd layer # of neurons

#xx=tf.placeholder('float',shape=[None,nx])

xx=tf.placeholder('float',shape=[nx,None])
yy=tf.placeholder('float',shape=[ny,None])

weights={
        'W1':tf.Variable(tf.random_uniform([n_hidden1,nx],-1,1)),
        'W2':tf.Variable(tf.random_uniform([n_hidden2,n_hidden1],-1,1)),
        'out':tf.Variable(tf.random_uniform([ny,n_hidden2],-1,1))
        }

biases={
        'b1':tf.Variable(tf.zeros([n_hidden1,1])),
        'b2':tf.Variable(tf.zeros([n_hidden2,1])),
        'out':tf.Variable(tf.zeros([ny,1]))
        }


# Create NN model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(weights['W1'],x), biases['b1'])
    
    layer_1 =tf.nn.relu(layer_1)
    
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(weights['W2'],layer_1), biases['b2'])
    
    layer_2 = tf.nn.relu(layer_2)
    
    # Output fully connected layer with a neuron for each class
    out_layer = tf.add(tf.matmul(weights['out'],layer_2), biases['out'])
    
    out_layer = tf.nn.relu(out_layer)
    
    return out_layer

### generate mini_batches
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = np.int(np.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, (k*mini_batch_size) :((k+1)* mini_batch_size)]
        mini_batch_Y = shuffled_Y[:, (k*mini_batch_size) :((k+1)* mini_batch_size)]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:,mini_batch_size*num_complete_minibatches:m]
        mini_batch_Y = shuffled_Y[:,mini_batch_size*num_complete_minibatches:m]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


# Construct NN model
pred = neural_net(xx)

# cost computed by RMSE
cost=tf.sqrt(tf.reduce_mean(tf.pow(pred-yy,2)))

regularizer=tf.nn.l2_loss(weights['W1'])+tf.nn.l2_loss(weights['W2'])+tf.nn.l2_loss(weights['out'])

beta=0.005


optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost+beta*regularizer)
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

costs=[]
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init)
    
    for epoch in range(num_epochs):
        
        minibatch_cost=0.
        num_minibatches=int(mx/minibatch_size)
        seed=seed+1
        minibatches=random_mini_batches(tranx,trany,minibatch_size,seed)
        for minibatch in minibatches:
            
            (minibatch_X,minibatch_Y)=minibatch
            _, temp_cost=sess.run([optimizer,cost],feed_dict={xx:minibatch_X,yy:minibatch_Y})
             
            minibatch_cost+=temp_cost/num_minibatches
        # print the cost every epoch
        if epoch % display_step==0:
            print('cost after epoch %i:%f'%(epoch,minibatch_cost))
            costs.append(minibatch_cost)
        
    print('Optimization finished!')
    _,training_cost=sess.run([optimizer,cost],feed_dict={xx:tranx,yy:trany})
    print("Training cost=", training_cost, '\n')

    
    # plot
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iteration (per 5 epochs)')
    plt.title('Learning rate='+str(learning_rate))
    plt.show()


#### how to use the model to do the prediction ????????
    val_y=neural_net(xx)
    pred_val_y=sess.run(val_y,feed_dict={xx:valid_x})
    
    test['vistors']=sess.run(val_y,feed_dict={xx:testx})
    
    val_err=np.sqrt(np.mean(np.power(pred_val_y-valid_y,2)))
    print('The validation error is:',val_err)
###############################################################    
#%%

### the following is a sample code for the application of Support Vector Machine and
### Random Forest.
train_xx1=train_x1.iloc[:2000,:]
train_yy1=train_y1[:2000,]

valid_xx1=valid_x1.iloc[:2000,:]
valid_yy1=valid_y1[:2000,]


scaler=preprocessing.StandardScaler()
scaler.fit(train_x1)
train_xx=scaler.transform(train_xx1)
train_yy=train_yy1

regressor = SVR(kernel='rbf',C=1000,gamma='auto')

#regressor = RandomForestRegressor(n_estimators=200, min_samples_split=1)

regressor.fit(train_xx,train_yy)
valid_score=metrics.mean_squared_error(regressor.predict(scaler.transform(valid_xx1)), valid_yy1)
X_ty = regressor.predict(train_xx)
train_score = metrics.mean_squared_error(X_ty, train_yy)

print('Training MSE: {0:.10f}'.format(train_score))
print('Validation MSE: {0:f}'.format(valid_score))

# plot the results
fig,ax = plt.subplots()
ax.scatter(train_yy, X_ty)
ax.plot([train_yy.min(), train_yy.max()],[train_yy.min(), train_yy.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()  


#%% 
####################### XGB for prediction
regressor = 1

print('\n\nAdopted regressor = ', regressor,'\n')

if (regressor == 1):
    print('Starting XGBoost')
#    boost_params = {'eval_metric': 'rmse'}
    xgb0 = xgb.XGBRegressor(
        max_depth        = 8,
        learning_rate    = 0.05,
        n_estimators     = 500,
        objective        = 'reg:linear',
        gamma            = 0,
        min_child_weight = 1,
        subsample        = 1,
        colsample_bytree = 1,
        scale_pos_weight = 1,
        silent=0,
        seed             = 23)
 #       **boost_params)
    
 ########## xgb cross-validation
    xgb_pram=xgb0.get_params()
    xgbtrain=xgb.DMatrix(train_x,train_y)
    xgbcv=xgb.cv(xgb_pram,xgbtrain,num_boost_round=xgb_pram['n_estimators'],nfold=5,
                 metrics='rmse',early_stopping_rounds=10)
    xgb0.set_params(n_estimators=xgbcv.shape[0])

 ########## 
#the following 3 lines of code are included in the original code  
    xgb0.fit(train_x, train_y,eval_metric='rmse')
    predict_y = xgb0.predict(test_x)    
    print('Finished XGBoost')
 ##########    ##########    ##########       
  predict_y1=xgb0.predict(train_x)
  print(xgb0.score(train_x,train_y)) # accuracy 
  
       
  feat_imp=pd.Series(xgb0.booster().get_fscore()).sort_values(ascending=False)
  feat_imp.plot(kind='bar',title='Feature importance')
  plt.ylabel('Feature Importance Score')

### GridSearchCV  
 param_test1={
           'max_depth':range(3,10,2),
          'min_child_weight':range(1,6,2)          
         }
 gsearch1=GridSearchCV(estimator=XGBRegressor(learning_rate =0.05, n_estimators=xgbcv.shape[0], max_depth=8,
 min_child_weight=1, gamma=0, subsample=1, colsample_bytree=1,
 objective= 'reg:linear', scale_pos_weight=1, seed=23),param_grid = param_test1,
    scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)
 
 gsearch1.fit(train_x,train_y)
 gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
 
 print(gsearch1.score(train_x,train_y)) # accuracy 

### RandomizedSearchCV
param_test2={
        'max_depth':sp_randint(3,10)}
#        'min_child_weight':sp_randint(1,6)
#        }
#param_test2={
#        'max_depth':sp_randint(3,10),
#        'min_child_weight':sp_randint(1,6)
#        }
n_iter_search=5
clf=XGBRegressor(learning_rate =0.05, n_estimators=xgbcv.shape[0], max_depth=8,
 min_child_weight=1, gamma=0, subsample=1, colsample_bytree=1,
 objective= 'reg:linear', scale_pos_weight=1, seed=23)
random_search1= RandomizedSearchCV(clf, param_distributions=param_test2,
                                   n_iter=n_iter_search,scoring='neg_mean_squared_error',
                                   n_jobs=4,iid=False,cv=5)
random_search1.fit(train_x,train_y)
report(random_search.cv_results_)


test['visitors'] = np.expm1(predict_y)

#%% output results
fname = 'submissionr v01 regressor ' + str(regressor) + '.csv'

test[['id', 'visitors']].to_csv(fname, index=False, float_format='%.3f')  


print ("Total processing time %s min" % nm)