import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix

# Plotting function
from Plot import plot_feat_Imp, plot_scatter_mat

# Scikit-learn utilities
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold, ShuffleSplit
#from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.neural_network import MLPClassifier
#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pickle

def ML_Model(data, sorted_Imp_name, feat_Imp_figname, Scatter_Mat_figname):
	X = data.iloc[:, :-1]
	y = data['label']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	scaler = StandardScaler()
	scaled_X_train = scaler.fit_transform(X_train)
	scaled_X_test = scaler.transform(X_test)

	# Random Forest Classifier Model
	RF = RandomForestClassifier(max_depth=50, max_features=80, min_samples_leaf=2,
								n_estimators=400)

	#RF.fit(X_train, y_train)
	RF.fit(scaled_X_train, y_train)
	
	#RF.score(X_test, y_test)
	Score = RF.score(scaled_X_test, y_test)
	print('Accuracy score of the RF model: {}'.format(Score))

	feature_importances = RF.feature_importances_
	feature_names = data.columns

	sorted_feature_imp = sorted(list(zip(feature_names, feature_importances.flatten())), 
								key=lambda x:x[1], reverse=True)

	# Save sorted importance data
	Name = sorted_Imp_name 
	with open(Name, "wb") as fp:
 		pickle.dump(sorted_feature_imp, fp)
		
	# Plot feature Importance
	n_feat = 10
	List = []

	for i in range(0, n_feat):
    	List.append(sorted_feature_imp[i][0])

	top_10_feat = data[List]

	plt.rcParams['font.size'] = 15
	plt.rcParams['axes.linewidth'] = 1.5

	# Plot top {n_feat} feature importance
	plot_feat_Imp(sorted_feature_imp, n_feat, feat_Imp_figname)

	# Plot scatter matrix for top {n_feat} feature
	plot_scatter_mat(data, top_10_feat, Scatter_Mat_figname)
