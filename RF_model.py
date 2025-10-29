import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix

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

R121S = '../CA_CA_Distance_MDAnalysis_data/R121S_CA_distances_MDAanalysis.pkl'
Y126F = '../CA_CA_Distance_MDAnalysis_data/Y126F_CA_distances_MDAanalysis.pkl'

def make_df(pkl, target):
	with open(pkl1, 'rb') as f1:
		df1 = pickle.load(f1)
	f1.close()

	df1['label'] = target
	return (df1)

df = pd.DataFrame()
if __name_ == "__main__":
	sys = ['d_ox', 'd_red', 'b_ox', 'b_red', 'R121S', 'Y126F', 'P141G', 'With', 'WithOut']
	for Sys in sys:
		stride = 1 # To open the dataframe
		data = make_df(sys)

	df = pd.concat([df1, df2], ignore_index=True)
	print('Shape of the total dataframe: {}'.format(df.shape))

	# Cut dataframe by 5 times
	cut_df = df.loc[0:60000:cut, :]
	print('Shape of the cut_df: {}'.format(cut_df.shape))

	# Shuffle the data
	shuffle_df = shuffle(cut_df, random_state=20)

	return (shuffle_df)
		

	
	
