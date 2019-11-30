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


train = pd.read_csv("/Users/yangyang/Desktop/5001/ind pro/msbd5001-fall2019/train.csv")
test = pd.read_csv("/Users/yangyang/Desktop/5001/ind pro/msbd5001-fall2019/test.csv")
submission = pd.read_csv('/Users/yangyang/Desktop/5001/ind pro/msbd5001-fall2019/samplesubmission.csv')

#0.combine train & test to df
#df = pd.DataFrame()
df = train.append(test,ignore_index = True)

###########################################
#################  Feature ################
###########################################


#1.isfree
def free_fea(df):
    df['is_free']=df['is_free'].astype('int')
    return df

df=free_fea(df)
print(df.shape)

#2.gen/cat/tag
def text_fea(df):
    genre = list(df['genres'])
    genre_lis = [i.split(",") for i in genre]
    all_genres = [genre.strip().upper() for listin in genre_lis for genre in listin]
    all_genres = list(set(all_genres))

    cat= list(df['categories'])
    cat_lis = [i.split(",") for i in cat]
    all_cats = [cat.strip().upper() for listin in cat_lis for cat in listin]
    all_cats = list(set(all_cats))

    tag = list(train['tags'])
    tag_lis = [i.split(",") for i in tag]
    all_tags = [tag.strip().upper() for listin in tag_lis for tag in listin]
    all_tags = list(set(all_tags))

    all = all_genres + all_cats + all_tags
    all = list(set(all))

    df_a = pd.DataFrame(columns=all)

    for i in range(len(df)):
        temp_gen = df['genres'][i].split(",")
        temp_cat = df['categories'][i].split(",")
        temp_tag = df['tags'][i].split(",")

        temp_gen = [listin.strip().upper() for listin in temp_gen]
        temp_cat = [listin.strip().upper() for listin in temp_cat]
        temp_tag = [listin.strip().upper() for listin in temp_tag]

        #temp_list = temp_gen
        temp_list = temp_gen
                    # + temp_cat + temp_tag

        for j in range(len(temp_list)):
            if temp_list[j] in all:
                df_a.loc[i,temp_list[j]] = 1

    df_a = df_a.replace(np.nan,0)
    df = df.join(df_a)
    df.drop(['tags','genres','categories'],axis=1,inplace=True)
    return df


df=text_fea(df)
print(df.shape)

#3.date
def date_fea(df):
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['release_date'] = pd.to_datetime(df['release_date'])

    for i in range(len(df)):
        if (df.loc[i, 'purchase_date'] - df.loc[i, 'release_date']).days < 0:
            temp = df.loc[i, 'purchase_date']
            df.loc[i, 'purchase_date'] = df.loc[i, 'release_date']
            df.loc[i, 'release_date'] = temp

    df['diffdate'] = (df['purchase_date'] - df['release_date']).astype('timedelta64[D]')
    mean_date = df['diffdate'].mean()
    df['diffdate'] = df['diffdate'].fillna(mean_date)

    df['purchase_date'] = df['purchase_date'].fillna(df['release_date'] + timedelta(days=mean_date))

    df['re_year'] = df["release_date"].apply(lambda dt: dt.year)
    df['re_month'] = df["release_date"].apply(lambda dt: dt.month)
    df['re_day'] = df["release_date"].apply(lambda dt: dt.day)

    df['pur_year'] = df["purchase_date"].apply(lambda dt: dt.year)
    df['pur_month'] = df["purchase_date"].apply(lambda dt: dt.month)
    df['pur_day'] = df["purchase_date"].apply(lambda dt: dt.day)

    now_date = datetime(2019, 9,7)
    df['pur_to_now'] = (now_date - df['purchase_date']).astype('timedelta64[D]')
    df['rel_to_now'] = (now_date - df['release_date']).astype('timedelta64[D]')

    cols = ['diffdate','pur_to_now','rel_to_now']
    scaler = MinMaxScaler(feature_range=(0,1))
    df[cols] = scaler.fit_transform(df[cols])
    df[cols] = scaler.transform(df[cols])

    df.drop(['purchase_date','release_date'],axis=1,inplace=True)
    return df

df = date_fea(df)
print(df.columns)
#4.review
def review_fea(df):
    mean_pos = df['total_positive_reviews'].mean()
    mean_neg = df['total_negative_reviews'].mean()

    df['total_positive_reviews'].fillna((mean_pos), inplace=True)
    df['total_negative_reviews'].fillna((mean_neg), inplace=True)
    # df['total_reviews'] = df['total_positive_reviews'] + df['total_negative_reviews']

    scaler = StandardScaler()
    cols = ['total_positive_reviews', 'total_negative_reviews']
    df[cols] = scaler.fit_transform(df[cols])
    df[cols] = scaler.transform(df[cols])
    return df

df = review_fea(df)
print(df.shape)

#5.price
def price_fea(df):
    # pri_bound = 20000
    # px_mean = df.price[df.price <= pri_bound].mean()
    # df.price = np.where(df.price > pri_bound, px_mean, df.price)
    col = ['price']
    scaler = MinMaxScaler()
    df[col] = scaler.fit_transform(df[col])
    df[col] = scaler.transform(df[col])
    return df

df = price_fea(df)
print(df.shape)

#6.split df to train & test

traindf= df[:357]
traindf.to_csv("newtrain.csv")
testdf = df[357:]
testdf.to_csv("newtest.csv")

#7.feature select

correlated_features = set()
correlation = traindf.corr()['playtime_forever']
correlation = correlation.fillna(0)
corr_col = correlation.index

for i in range(len(correlation)):
    if abs(correlation[i]) == 0 :
        featurename = corr_col[i]
        correlated_features.add(featurename)

correlated_features.add('id')
correlated_features.add('playtime_forever')

test_= testdf.drop(['id','playtime_forever'],axis=1,inplace=False)
#test_= testdf.drop(correlated_features,axis=1,inplace=False)
print(test_.shape)
print(test_.isnull().sum())

X = traindf.drop(['id','playtime_forever'],axis=1,inplace=False)
#X = traindf.drop(correlated_features,axis=1,inplace=False)
y = traindf['playtime_forever']
print(X.shape)

traindf2 = traindf.loc[(traindf['playtime_forever'] > 1)]
X2 = traindf2.drop(['id','playtime_forever'],axis=1,inplace=False)
#X2 = traindf2.drop(correlated_features,axis=1,inplace=False)
y2 = traindf2['playtime_forever']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#8.rmse

def scoreReg(testY,predicY):
    #testY是一维数组，predicY是二维数组，故需要将testY转换一下

    MSE2 = math.sqrt(mean_squared_error(testY, predicY))
    print('RMSE:',MSE2,"/n")


#9.build model
#lr
lr_model =  LinearRegression()
lr_model.fit(X_train, y_train)

# lr_model2 = LinearRegression(n_estimators=5)
# lr_model2.fit(X2,y2)

preds = lr_model.predict(X_test)
preds[preds<0] = 0

# preds2 = lr_model2.predict(X_test)
# for i in range(len(preds)):
#     if preds[i] >= 5:
#         preds[i] = preds2[i]

scoreReg(y_test,preds)
#rf
rf_model = RandomForestRegressor(n_estimators=5)
rf_model.fit(X_train, y_train)

# rf_model2 = RandomForestRegressor(n_estimators=5)
# rf_model2.fit(X2,y2)

preds = rf_model.predict(X_test)
preds[preds<0] = 0

# preds2 = rf_model2.predict(X_test)
# for i in range(len(preds)):
#     if preds[i] >= 5:
#         preds[i] = preds2[i]

scoreReg(y_test,preds)

#xgb
xgb_model =  XGBRegressor(n_estimators=5)
xgb_model.fit(X_train, y_train)

# xgb_model2 = XGBRegressor(n_estimators=5)
# xgb_model2.fit(X2,y2)

preds = xgb_model.predict(X_test)
preds[preds<0] = 0

# preds2 = xgb_model2.predict(X_test)
# for i in range(len(preds)):
#     if preds[i] >= 5:
#         preds[i] = preds2[i]

scoreReg(y_test,preds)

#ada
ada_model =  AdaBoostRegressor(n_estimators=5)
ada_model.fit(X_train, y_train)

# ada_model2 = AdaBoostRegressor(n_estimators=5)
# ada_model2.fit(X2,y2)

preds = ada_model.predict(X_test)
preds[preds<0] = 0

# preds2 = ada_model2.predict(X_test)
# for i in range(len(preds)):
#     if preds[i] >= 5:
#         preds[i] = preds2[i]

scoreReg(y_test,preds)

#Bagging
bag_model = BaggingRegressor(n_estimators=5)

bag_model.fit(X_train, y_train)

# bag_model2 = BaggingRegressor(n_estimators=5)
# bag_model2.fit(X2,y2)

preds = bag_model.predict(X_test)
preds[preds<0] = 0

# preds2 = bag_model2.predict(X_test)
# for i in range(len(preds)):
#     if preds[i] >= 5:
#         preds[i] = preds2[i]

scoreReg(y_test,preds)

#knn
#knn
knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)

# knn_model2 = KNeighborsRegressor(n_estimators=5)
# knn_model2.fit(X2,y2)

preds = knn_model.predict(X_test)
preds[preds<0] = 0

# preds2 = knn_model2.predict(X_test)
# for i in range(len(preds)):
#     if preds[i] >= 5:
#         preds[i] = preds2[i]

scoreReg(y_test,preds)

#pa
pa_model =  PassiveAggressiveRegressor()
pa_model.fit(X_train, y_train)

# pa_model2 = PassiveAggressiveRegressor(n_estimators=5)
# pa_model2.fit(X2,y2)

preds = pa_model.predict(X_test)
preds[preds<0] = 0

# preds2 = pa_model2.predict(X_test)
# for i in range(len(preds)):
#     if preds[i] >= 5:
#         preds[i] = preds2[i]

scoreReg(y_test,preds)

#9.get_result of model

#rf
rf_model = RandomForestRegressor(n_estimators=5)
rf_model.fit(X, y)

rf_model2 = RandomForestRegressor(n_estimators=5)
rf_model2.fit(X2,y2)

preds = rf_model.predict(test_)
preds[preds<0] = 0
preds[preds<0.05] = 0

preds2 = rf_model2.predict(test_)
for i in range(len(preds)):
    if preds[i] >= 5:
        if abs(preds2[i] - preds[i]) < 5:
            preds[i] = preds2[i]

sub_rf = pd.DataFrame(index=submission['id'])
#print(len(preds))
sub_rf['playtime_forever'] = preds
sub_rf.to_csv('/Users/yangyang/Desktop/5001/ind pro/submission1.csv')


#pa
pa_model =  PassiveAggressiveRegressor()
pa_model.fit(X, y)

# pa_model2 = PassiveAggressiveRegressor()
# pa_model2.fit(X2,y2)

preds = pa_model.predict(test_)
preds[preds<0] = 0

# preds2 = pa_model2.predict(test_)
# for i in range(len(preds)):
#     if preds[i] >= 5:
#         preds[i] = preds2[i]

sub_pa = pd.DataFrame(index=submission['id'])
#print(len(preds))
sub_pa['playtime_forever'] = preds
sub_pa.to_csv('/Users/yangyang/Desktop/5001/ind pro/nouse.csv')



