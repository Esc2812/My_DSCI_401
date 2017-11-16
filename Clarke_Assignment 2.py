#Imported libraries
import numpy as np
import pandas as pd
import seaborn as sns
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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#Importing data
housing = pd.read_csv('./data/AmesHousingSetA.csv')
housing_test = pd.read_csv('./data/AmesHousingSetB.csv')
housing.head()

#Categorical data features and making list of the indices
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)
	
#filling empty data and setting data_x/y
housing = pd.get_dummies(housing, columns = cat_features(housing))
del housing["PID"]
data_x = housing[list(housing)[0:78]]
data_y = housing[list(housing)[79]]

#imputting new data
imp = preprocessing.Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0)
data_x = imp.fit_transform(data_x)
normed_data_x = preprocessing.normalize(data_x, axis = 0)

#Base Model
x_train,x_test,y_train,y_test = train_test_split(data_x,data_y, test_size=0.2, random_state = 4)
base_model = linear_model.LinearRegression()
base_model.fit(x_train,y_train)
preds = base_model.predict(x_test)

pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))
print('MSE, MAE, R^2, EVS (Base Model): ' + str([mean_squared_error(y_test, preds), \
													median_absolute_error(y_test, preds), \
													r2_score(y_test, preds), \
													explained_variance_score(y_test, preds)]))
# Lasso	Regression w/ different alphas									
alphas = [0.01, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
for a in alphas:
	 lasso_model = linear_model.Lasso(alpha=a, normalize=True, fit_intercept=True)
	 lasso_model.fit(x_train, y_train)
	 preds = lasso_model.predict(x_test)
	 print('R^2 (Lasso Model with alpha=' + str(a) + '): ' + str(r2_score(y_test, preds)))
	 
lasso_model = linear_model.Lasso(alpha=5.0, normalize=True, fit_intercept=True)
lasso_model.fit(x_train, y_train)
preds = model.predict(x_test)
pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))
print('For data_x: ' + 'MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)])) 
							   					
# Top 25% feature selector
selector_f = SelectPercentile(f_regression, percentile=25)
selector_f.fit(x_train, y_train) 
	
xt_train, xt_test = selector_f.transform(x_train), selector_f.transform(x_test)
percent_model = linear_model.LinearRegression()
percent_model.fit(xt_train, y_train) 
preds = percent_model.predict(xt_test)  
pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))

print('MSE, MAE, R^2, EVS (Top 25% Model): ' + str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)])) 
							   

#DATA B

def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)
	
housing_test = pd.get_dummies(housing_test, columns = cat_features(housing_test))
del housing_test["PID"]
datab_x = housing_test[list(housing_test)[0:78]]
datab_y = housing_test[list(housing_test)[79]]

#imputting new data
imp = preprocessing.Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0)
datab_x = imp.fit_transform(data_x)
normed_datab_x = preprocessing.normalize(datab_x, axis = 0)

#Linear Regression
datab_model = linear_model.LinearRegression()
datab_model.fit(datab_x, datab_y)

preds = datab_model.predict(datab_x)
pprint.pprint(pd.DataFrame({'Actual':datab_y, 'Predicted':preds}))

print('DATA B  - MSE, MAE, R^2, EVS (Base Model): ' + str([mean_squared_error(datab_y, preds), \
							   median_absolute_error(datab_y, preds), \
							   r2_score(datab_y, preds), \
							   explained_variance_score(datab_y, preds)])) 
							   
# Lasso Regression
alphas = [0.01, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
for a in alphas:
	 lasso_model = linear_model.Lasso(alpha=a, normalize=True, fit_intercept=True)
	 lasso_model.fit(datab_x, datab_y)
	 preds = lasso_model.predict(datab_x)
	 print('R^2 (Lasso Model with alpha=' + str(a) + '): ' + str(r2_score(datab_y, preds)))
	 
lasso_model = linear_model.Lasso(alpha=5.0, normalize=True, fit_intercept=True)
lasso_model.fit(datab_x, datab_y)
preds = lasso_model.predict(datab_x)
pprint.pprint(pd.DataFrame({'Actual':datab_y, 'Predicted':preds}))
print('For data_x: ' + 'MSE, MAE, R^2, EVS: ' + str([mean_squared_error(datab_y, preds), \
							   median_absolute_error(datab_y, preds), \
							   r2_score(datab_y, preds), \
							   explained_variance_score(datab_y, preds)])) 