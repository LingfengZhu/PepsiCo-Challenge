##### Lingfeng Zhu: PepsiCo Challenge Code #####
# import packages
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from hyperopt import fmin, tpe, hp, partial

# read the data
snack = pd.read_excel('shelf-life-study-data-for-analytics-challenge-v2.xlsx')
original_snack = snack # backup for the raw data

# pre-processing
# one-hot encoding for caterogical columns
snack = snack.iloc[:,2:]
snack = pd.get_dummies(snack, prefix_sep="_")
# split the data: X and y
X = snack.drop(columns=['Sample Age (Weeks)'])
y = snack['Sample Age (Weeks)']

### this part is to perform hyper-parameter tuning:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=30)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=30)

# convert the data to XGboost's format to save time during training
dtrain = xgb.DMatrix(data=X_train, label=y_train.values.ravel())
dvalid = xgb.DMatrix(data=X_valid, label=y_valid.values.ravel())

evallist = [(dvalid, 'eval'), (dtrain, 'train')]

# define parameter space for hyperopt
space = {"max_depth": hp.randint("max_depth", 20),
         "n_estimators": hp.randint("n_estimators",400),
         "learning_rate": hp.uniform('learning_rate', 1e-3, 5e-1),
         "subsample": hp.randint("subsample", 5),
         "min_child_weight": hp.randint("min_child_weight", 6),
         "gamma": hp.uniform("gamma", 0, 2),
         "reg_alpha": hp.uniform("reg_alpha", 0, 1),
         "reg_lambda": hp.uniform("reg_lambda", 0, 1)
         # "booster": hp.choice('booster', ["gbtree", "gblinear", "dart"])
         }

# transform the space since the hp.randint() function generates values from 0
def argsDict_tranform(argsDict, isPrint=False):
    argsDict["max_depth"] = argsDict["max_depth"] + 5
    argsDict['n_estimators'] = argsDict['n_estimators'] + 150
    argsDict["learning_rate"] = argsDict["learning_rate"] * 0.02 + 0.05
    argsDict["subsample"] = argsDict["subsample"] * 0.1 + 0.5
    argsDict["min_child_weight"] = argsDict["min_child_weight"] + 1
    if isPrint:
        print(argsDict)
    else:
        pass

    return argsDict

# generate the model
def xgboost_factory(argsDict):
    argsDict = argsDict_tranform(argsDict)
    
    params = {'nthread': -1, 
              'booster': "gbtree",
              'max_depth': argsDict['max_depth'], 
              'n_estimators': argsDict['n_estimators'],  
              'eta': argsDict['learning_rate'], 
              'subsample': argsDict['subsample'], 
              'min_child_weight': argsDict['min_child_weight'], 
              'objective': 'reg:squarederror',
              'silent': 0,  
              # 'gamma': argsDict['gamma'], 
              'colsample_bytree': 0.7,  
              # 'reg_alpha': argsDict['reg_alpha'],  
              # 'reg_lambda': argsDict['reg_lambda'], 
              'scale_pos_weight': 0,  
              'seed': 30,  
              'missing': -999,  
              }
    params['eval_metric'] = ['rmse']

    xrf = xgb.train(params, dtrain, params['n_estimators'], evallist, early_stopping_rounds=100)

    return get_tranformer_score(xrf)

def get_tranformer_score(tranformer):
    
    xrf = tranformer
    dpredict = xgb.DMatrix(X_test)
    prediction = xrf.predict(dpredict, ntree_limit=xrf.best_ntree_limit)
  
    return metrics.mean_squared_error(y_test, prediction)

## use this cell to perform hyper-parameter tuning
# !pip uninstall bson && pip install pymongo
# algo = partial(tpe.suggest, n_startup_jobs=1)
# best = fmin(xgboost_factory, space, algo=algo, max_evals=20, pass_expr_memo_ctrl=None)

## the hyper-parameter tuning result
# RMSE = xgboost_factory(best)
# print('best :', best)
# print('best param after transform :')
# argsDict_tranform(best, isPrint=True)
# print('rmse of the best xgboost:', np.sqrt(RMSE))

# train the xgboost model: hyper-parameters where tuned using hyperopt
bst = xgb.XGBRegressor(seed=30,
                       learning_rate=0.05113394269783118,
                       objective = 'reg:squarederror',
                       max_depth=12,
                       min_child_weight=7,
                       n_estimators=439,
                       subsample=0.9)
bst.fit(X, y.values.ravel(), verbose=True)

# prediction
def predict_shelf_life(original_df, model):
    '''
    input:
    original_df: the original data frame of snack products which has the same format of the given data
    model: my model based on the given data

    output:
    res: a vector of predicted shelf life
    '''
    # one hot encoding
    df = original_df.iloc[:,2:] # drop the study number and sample ID
    df = pd.get_dummies(df, prefix_sep="_")
    # set "Difference From Fresh" to be 20 to calculate the shelf life
    df["Difference From Fresh"] = 20
#     # set the measures of Chemical material to be missing value
#     df["Moisture (%)"] = np.nan
#     df['Residual Oxygen (%)'] = np.nan
#     df['Hexanal (ppm)'] = np.nan
    # define X
    X = df.drop(columns=['Sample Age (Weeks)'])
    # prediction
    res = model.predict(X)

    return res

# add the predicted shelf life to the original data
life = predict_shelf_life(original_snack, bst).round(decimals=0, out=None)
original_snack['Prediction (Weeks)'] = life

# save the result
original_snack.to_excel('LingfengZhu.xlsx')
