#import libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from data_util import *

#import data 
churn = pd.read_csv('./data/churn_data.csv')
churn_test = pd.read_csv('./data/churn_validation.csv')
churn.head()

features = list(churn)
features.remove('Churn')
features.remove('CustID')
data_x = churn[features]
data_y = churn['Churn']

#transformations
le = preprocessing.LabelEncoder()
data_y = le.fit_transform(data_y)
data_x = pd.get_dummies(data_x)

#split test and training data
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# Model Fitting 
	#KNN 
	knn_mod = KNeighborsClassifier()
	knn_mod.fit(x_train, y_train)
	preds = knn_mod.predict(x_test)
	print_multiclass_classif_error_report(y_test, preds)
		#Reapplying labels
		y_test_label = le.inverse_transform(y_test)
		pred_label = le.inverse_transform(preds)
		print('(Actual, Predicted): \n' + str(zip(y_test_label, pred_label)))

	pred_knn = knn_mod.predict(x_test)
	print_binary_classif_error_report(y_test, pred_knn)
	# Accuracy - .564, Precision - 0.6, Recall - 0.571, ROC AUC - 0.563, Confusion Matrix [[10 8][9 12]]

	#GNB 
	gnb_mod = naive_bayes.GaussianNB()
	gnb_mod.fit(x_train, y_train)
	preds = gnb_mod.predict(x_test)
	print_multiclass_classif_error_report(y_test, preds)
		#Reapplying labels
		y_test_label = le.inverse_transform(y_test)
		pred_label = le.inverse_transform(preds)
		print('(Actual, Predicted): \n' + str(zip(y_test_label, pred_label)))
		
	pred_gnb = gnb_mod.predict(x_test)
	print_binary_classif_error_report(y_test, pred_gnb)
	# Accuracy - 0.692, Precision - 0.6, Recall - 0.571, ROC AUC - 0.563, Confusion Matrix [[10 8][9 12]]

	#Random Forest 
	n_est = [10,50,70,100]
depth = [3, 6, None]
for n in n_est:
	for dp in depth:
		rf_mod = ensemble.RandomForestClassifier(n_estimators=n, max_depth=dp)
		rf_mod.fit(x_train, y_train)
		preds = rf_mod.predict(x_test)
		print('EVALUATING MODEL: n_estimators = ' + str(n_est) + ', depth =' + str(dp) + ' -------------------')
		print_multiclass_classif_error_report(y_test, preds)
		#Reapplying labels
		y_test_label = le.inverse_transform(y_test)
		pred_label = le.inverse_transform(preds)
		print('(Actual, Predicted): \n' + str(zip(y_test_label, pred_label)))
		
	pred_rf = rf_mod.predict(x_test)
	print_binary_classif_error_report(y_test, pred_rf)
	#10,3:  Accuracy - 0.795, Precision - 0.933, Recall - 0.667, ROC AUC - 0.806, Confusion Matrix [[17 1][7 14]]
	#10,6:  Accuracy - 0.846, Precision - 1.0, Recall - 0.714, ROC AUC - 0.857, Confusion Matrix [[18 0][6 15]]
	#10,None: Accuracy - 0.846, Precision - 1.0, Recall - 0.714, ROC AUC - 0.857, Confusion Matrix [[18 0][6 15]]
	#50,3: Accuracy - 0.846, Precision - 0.895, Recall - 0.809, ROC AUC - 0.849, Confusion Matrix [[16 2][4 17]]
	#50,6:Accuracy - 0.821, Precision - 0.889, Recall - 0.762, ROC AUC - 0.825, Confusion Matrix [[16 2][5 1]]
	#50,None: Accuracy - 0.872, Precision - 1.0, Recall - 0.762, ROC AUC - 0.881, Confusion Matrix [[18 0][5 16]]
	#70,3: 5#50,6:Accuracy - 0.821, Precision - 0.889, Recall - 0.762, ROC AUC - 0.825, Confusion Matrix [[16 2][5 1]]
	#70,6:Accuracy - 0.846, Precision - 0.941, Recall - 0.762, ROC AUC - 0.853, Confusion Matrix [[17 1][5 16]]
	#70,None: Accuracy - 0.846, Precision - 0.941, Recall - 0.762, ROC AUC - 0.853, Confusion Matrix [[17 1][5 16]]
	#100,3:Accuracy - 0.744, Precision - .867, Recall - 0.619, ROC AUC - 0.754, Confusion Matrix [[16 2][28 13]]
	#100,6:Accuracy - 0.846, Precision - 0.941, Recall - 0.762, ROC AUC - 0.853, Confusion Matrix [[17 1][5 16]]
	#100,None:Accuracy - 0.897, Precision - 1.0, Recall - 0.809, ROC AUC - 0.905, Confusion Matrix [[18 0][4 17]]
-----------------------------------------------------------------------------------------------------------------------------
#churn validation data 
features = list(churn_test)
features.remove('Churn')
features.remove('CustID')
data_x = churn_test[features]
data_y = churn_test['Churn']

#transformations
le = preprocessing.LabelEncoder()
data_y = le.fit_transform(data_y)
data_x = pd.get_dummies(data_x)
	
rf_mod = ensemble.RandomForestClassifier(n_estimators=100, max_depth=None)
preds = rf_mod.predict(data_x)

print_multiclass_classif_error_report(data_y, preds)
#Reapplying labels
data_y_label = le.inverse_transform(data_y)
pred_label = le.inverse_transform(preds)
print('(Actual, Predicted): \n' + str(zip(data_y_label, pred_label)))
		
pred_rf = rf_mod.predict(data_x)
print_binary_classif_error_report(y_test, pred_rf)

# Accuracy - 1.0, Precision - 1.0, Recall - 1.0, ROC AUC - 1.0, Confusion Matrix [[19 0][0 13]]