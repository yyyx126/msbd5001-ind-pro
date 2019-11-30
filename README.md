# msbd5001-ind-pro
2019 msbd 5001 individual project

# Team name on kaggle: 
YYYX

# Language
python3

# Required packages:
import warnings

from keras.losses import mean_squared_error

warnings.filterwarnings("ignore")

import pandas as pd

import numpy as np

from datetime import datetime

from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn.model_selection import GridSearchCV, cross_validate,train_test_split

import math

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from xgboost.sklearn import XGBRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.linear_model import PassiveAggressiveRegressor

from sklearn.linear_model import SGDRegressor

from sklearn.neighbors import KNeighborsRegressor


# Code
## ind.py
0.Contact train set and test set = df

1.feature —— isfree boolen to int（0/1）

2.feature —— genres/cates/tags to one_hotting [only use genres to avoid overfitting]

3.feature —— datetime[date]. split date to year/month/day  compute gap between pur_date and rel_date/now_date and pur_date

4.feature —— reviews MinMaxScaler

5.feature —— price MinMaxScaler

6.Split data to train and test sets

7.Evaltion —— rmse

8.Select different model

9.Choose RF and PA as my model to get submission file 


## ind_2.py

2.feature —— genres/cates/tags to target_encoding [only use genres to avoid overfitting]

other parts are same as ind.py

# Finally submissions
Considering the advise from TA, I chose the RF generated of ind.py and the PA generated of ind1.py.

# Run
Directly run the ind.py and ind_2.py
However, the result depends on luck.

