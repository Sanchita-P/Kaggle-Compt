import pandas as pd
import numpy as np
from itertools import product
import seaborn as sns
import os
import matplotlib.pyplot as plt
import scipy.sparse 

from sklearn.metrics import mean_squared_error,make_scorer
from math import sqrt
#from sklearn.ensemble import RandomForestRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import lightgbm as lgb
#import xgboost as xgb

## Code for getting the predicted outputs in the desired format
def get_submission(item_cnt_month,sub_name,clip=20,data_path ='C:/Users/as14478/Sanchita Kaggle' ):
    item_cnt_month = np.clip(item_cnt_month,0,clip)
    test= pd.read_csv(os.path.join(data_path, 'test.csv.gz'))
    sub = test.copy()
    sub['item_cnt_month'] = item_cnt_month
    sub.drop(['item_id','shop_id'],axis=1,inplace=True)
    sub.to_csv(data_path+'submission/' + sub_name+'.csv',index=False)
    return sub

##Converting 64 bits to 32 bits
def downcast_dtypes(df):
    '''
    Changes column types in the dataframe: 

    `float64` type to `float32`
    `int64`   type to `int32`
    '''

    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]

    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int32)
    return df

##Creating train and validation for K-folds cross validation
def get_cv_idxs(df,start,end):
    result=[]
    for i in range(start,end+1):
        dates = df.date_block_num
        train_idx = np.array(df.loc[dates <i].index)
        val_idx = np.array(df.loc[dates == i].index)
        result.append((train_idx,val_idx))
    return np.array(result)
   
##Creating Y and X Variables from input dataset and  clipping the Y Variable   
def get_X_y(df,end,clip=20):
# don't drop date_block_num
    df = df.loc[df.date_block_num <= end]
    cols_to_drop=['target','item_name'] + df.columns.values[6:12].tolist()
    y = np.clip(df.target.values,0,clip)
    X = df.drop(cols_to_drop,axis=1)
    return X,y

##Retaining additional Features
def get_X_y_add_features(df,end,clip=20):
    # don't drop date_block_num
    df = df.loc[df.date_block_num <= end]
    cols_to_drop=['target','item_name'] 
    y = np.clip(df.target.values,0,clip)
    X = df.drop(cols_to_drop,axis=1)
    return X,y

##RMSE
def root_mean_squared_error(truth,pred):
    return sqrt(mean_squared_error(truth,pred))

##Importing the data
def get_all_data(data_path,filename):
    all_data = pd.read_pickle(data_path + filename)
    all_data = downcast_dtypes(all_data)
    all_data = all_data.reset_index().drop('index',axis=1)
    return all_data
