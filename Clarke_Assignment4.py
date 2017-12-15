#import libraries 
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from data_util import *


life= pd.read_csv('./data/life.csv')
life_test = pd.read_csv('./data/life_test.csv')

#transformations
life = life[life.Year != 2015]
life = life[life.Year != 2016]
features = list(life)
features.remove('Shorter Life')
data_x = life[features]
data_y = life['Shorter Life']

le = preprocessing.LabelEncoder()
data_y = le.fit_transform(data_y)
data_x = pd.get_dummies(data_x)

# Model Fitting 
	#KNN 
	knn_mod = KNeighborsClassifier()
	knn_mod.fit(x_train, y_train)
	kpred = knn_mod.predict(x_test)
		#Reapplying labels
		y_test_label = le.inverse_transform(y_test)
		pred_label = le.inverse_transform(kpred)
		print('(Actual, Predicted): \n' + str(zip(y_test_label, pred_label)))
	print_binary_classif_error_report(y_test, kpred)
	# Accuracy - .714, Precision - 0.773, Recall - 0.833,F1 - 0.802, ROC AUC - 0.638, Confusion Matrix [[42 53][3 180]]
	
	#GNB 
	gnb_mod = naive_bayes.GaussianNB()
	gnb_mod.fit(x_train, y_train)
	gpred = gnb_mod.predict(x_test)
		#Reapplying labels
		y_test_label = le.inverse_transform(y_test)
		pred_label = le.inverse_transform(gpred)
		print('(Actual, Predicted): \n' + str(zip(y_test_label, pred_label)))
	print_binary_classif_error_report(y_test, gpred)
	# Accuracy - .695, Precision - 0.701, Recall - 0.977,F1 - 0.816, ROC AUC - 0.515, Confusion Matrix [[5 90][5 211]]

	# Linear 
	lin_mod = linear_model.LinearRegression()
	lin_mod.fit(x_train,y_train)
	lpred = lin_mod.predict(x_test)
	#Reapplying labels
		y_test_label = le.inverse_transform(y_test)
		pred_label = le.inverse_transform(lpred.any())
		print('(Actual, Predicted): \n' + str(zip(y_test_label, pred_label)))
	# MSE - .205, MAE - .345, R^2 - 0.033, EVS - 0.033 
	
	--------------------------------------------------------------------
	Validation data 
	life_test = life_test[life_test.Year != 2015]
	life_test = life_test[life_test.Year != 2016]
	features = list(life_test)
	features.remove('Shorter Life')
	test_x = life_test[features]
	test_y = life_test['Shorter Life']

	le = preprocessing.LabelEncoder()
	test_y = le.fit_transform(test_y)
	test_x = pd.get_dummies(test_x)
	
	knn_mod = KNeighborsClassifier()
	knn_mod.fit(x_train, y_train)
	vpred = knn_mod.predict(x_val)
	print_binary_classif_error_report(y_test, vpred)
	# Accuracy - .892, Precision - 0.792, Recall - 0.877,F1 - 0.816, ROC AUC - 0.515, Confusion Matrix [[3 45][2 187]]
	