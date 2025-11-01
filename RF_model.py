import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix

# Load features
from Create_Dictionary import load_sys

# ML training and prediction
from ML_Pred import ML_Model

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

def make_df(pkl, target):
	with open(pkl, 'rb') as f1:
		df1 = pickle.load(f1)
	f1.close()

	# Add target : active (1) or inactive (0)
	df1['label'] = target
	
	return (df1)

stride = 1 # To open the dataframe
# Create a blank dataframe
df = pd.DataFrame()

if __name_ == "__main__":
	sys = ['d_ox', 'd_red', 'b_ox', 'b_red', 'R121S', 'Y126F', 'P141G', 'With', 'WithOut']
	for Sys in sys:
		system = load_sys()[Sys]
		d = make_df(system['pkl'], system['target'])

		df = pd.concat([df, d], ignore_index=True)
		print('Shape of the total dataframe after : {} after system {}'.format(df.shape, Sys))

	# Cut dataframe by 5 times
	cut_df = df.loc[::stride, :]
	print('Shape of the cut_df: {}'.format(cut_df.shape))

	# Shuffle the data
	shuffle_df = shuffle(cut_df, random_state=20)

	######### Cross-Check this part ##################
	# Drop features if correlation coefficient > cutoff_th
	features = shuffle_df.iloc[:, :-1]
	corr = features.corr().abs()
	cutoff_th = 0.9

	upper_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
	drop_col = [column for column in upper_corr.columns if any(upper_corr[column] > cutoff_th)]
	df_dropped = shuffle_df.drop(columns = drop_col)
	##################################################

	print('Shape of the dataframe after dropping high correlated features : {}'.format(df_dropped.shape))

	# Training and plot
	ML_Model(df_dropped, 'Feature_Importance_RF_model.pkl', 
			 'RF_Top20_Feature_Imp.png', 'Scatter_Matrix_Top10_Feat.png')
