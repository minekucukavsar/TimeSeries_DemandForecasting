#!pip install lightgbm

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')


###Loading the data###
train = pd.read_csv(r'C:\Users\hp\PycharmProjects\pythonProject2\demand_forecasting\train.csv', parse_dates=['date'])
test = pd.read_csv(r'C:\Users\hp\PycharmProjects\pythonProject2\demand_forecasting\test.csv', parse_dates=['date'])

#Expected Output File
sample_sub = pd.read_csv(r'C:\Users\hp\PycharmProjects\pythonProject2\demand_forecasting\sample_submission.csv')

#We have to concatenate data. Because we analyze and preprocess the train set but the test set has to be in the same format. We can apply the same preprocess to the test file but this concat method is more manageable.
df=pd.concat([train,test],sort=False)

###Exploratary Data Analysis

df["date"].min(), df["date"].max()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

df[["store"]].nunique()
df[["item"]].nunique()

df.groupby(["store"])["item"].nunique()
df.groupby(["store", "item"]).agg({"sales": ["sum"]})
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})

df.head()

###FEATURE ENGINEERING
#To understand the trend and seasonality, we need to extract new features from the dataset.
#with "create_date_features" function we can effectively extract the date features from date columns.
def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)

df.groupby(["store","item","month"]).agg({"sales":["sum","mean","median","std"]})


###Random Noise###
#We add "random noise" to avoid over-fitting.Because I added new observations in dataframe size and put them into newly created variables related to sales.
#np.random.normal/draw random samples from a normal (Gaussian) distribution.

#Normal distribution (Gaussian distribution) is a probability distribution that is symmetric about the mean, demonstrating that data around the mean are more frequent in occurrence than data far from the mean.
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

###Lag/Shifted Features###
#The data must be sorted by store, product and date.This is important for calculating "lag features".
df.sort_values(by=["store","item","date"],axis=0,inplace=True)
df.head()
#We will add features based on the sales variable.
#Our aim is to do the time series analysis with a machine learning algorithm. It is important to reflect the effects such as trend and seasonality to the model.
pd.DataFrame({"sales": df["sales"].values[0:10],
              "lag1": df["sales"].shift(1).values[0:10],
              "lag2": df["sales"].shift(2).values[0:10],
              "lag3": df["sales"].shift(3).values[0:10],
              "lag4": df["sales"].shift(4).values[0:10]})
#shift() function shift index by preferred number of periods with an optional time frequency. This function takes a scalar parameter called the period, which indicates the number of shifts to be made over the desired axis.
#Axis can be choosen for the purpose and it is used axis=0 for this project).

df.groupby(["store", "item"])['sales'].head()

df.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(1))


def lag_features(dataframe, lags):
    for lag in lags:
        dataframe["sales_lag"+str(lag)]=dataframe.groupby(["store","item"])["sales"].transform(lambda x: x.shift(lag))+random_noise(dataframe)
    return dataframe


df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])
#We create the values of the sales values that in the past periods.

df.head()
#When we look at the date 2013-01-01, the lag features are "NaN". The reason is that there is no value in the data set before that date.
df.tail()

check_df(df)

###Rolling Mean Features
#Rolling functions serve many different kinds of calculations on subsets of your data such as sum, std, mean, etc.
#Moving Avg = ([t] + [t-1]) / 2 for calculation meaning of t and t-1 term.
pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].rolling(window=5).mean().values[0:10]})

pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].shift(1).rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].shift(1).rolling(window=5).mean().values[0:10]})

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df, [365, 546])


# Exponentially Weighted Mean Features
#"ewm" provide exponentially weighted (EW) calculations
#
pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "ewm099": df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10],
              "ewm095": df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm02": df["sales"].shift(1).ewm(alpha=0.1).mean().values[0:10]})

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)
check_df(df)

###One-Hot Encoding
#Encoding for categorical variables.
df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])

check_df(df)

###Converting sales to log(1+sales)
#This part is optional.
#The log-transformation is used to deal with skewed data.
#Skewness is a measurement of the distortion of symmetrical distribution or asymmetry in a data set.
df['sales'] = np.log1p(df["sales"].values)

check_df(df)

###Model###

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)
# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# mse: l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


# Time-Based Validation Sets

train=df.loc[(df[["date"]<"2017-01-01"]),:]

val=df.loc[(df["date"]>="2017-01-01")&(df["date"] < "2017-04-01"),:]

cols=[col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

###LightGBM

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}


lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))




