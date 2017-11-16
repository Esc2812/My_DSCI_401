#Imported libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import make_scorer
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn import preprocessing

#Imported and viewed data
housing = pd.read_csv('./data/AmesHousingSetA.csv')
housing_test = pd.read_csv('./data/AmesHousingSetB.csv')
housing.head()

#Filtered NAN and TAs
cat_feat = train.select_dtypes(include = ["object"]).columns
num_feat = train.select_dtypes(exclude = ["object"]).columns
housing_cat = train[cat_feat]
housing_num = train[num_feat]

#transforming data into one hot encoding 
housing_cat = pd.get_dummies(housing_cat)
housing_num = housing_num.fillna(housing_num.mean())
housing = pd.concat([housing_num,housing_cat],axis=1)

features = list(housing)
features.remove('SalePrice')

data_x = housing[features]
data_y = housing['SalePrice']

#Base Model
base_mod = linear_model.LinearRegression()
x_train,x_test,y_train,y_test = train_test_split(data_x,data_y, test_size =0.2, random_state = 4)
base_mod.fit(x_train,y_train)
preds = base_mod.predict(x_test)

print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)]))