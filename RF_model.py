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
	
	

	df = pd.concat([df1, df2], ignore_index=True)
	print('Shape of the total dataframe: {}'.format(df.shape))

	# Cut dataframe by 5 times
	cut_df = df.loc[0:60000:cut, :]
	print('Shape of the cut_df: {}'.format(cut_df.shape))

	# Shuffle the data
	shuffle_df = shuffle(cut_df, random_state=20)

	return (shuffle_df)

def plot_feat_Imp(Important_feat_list, n_feat, figname):
	""" Function to plot feature importance for few most important features.

	Parameter:
	----------
	Important_feat_list: sorted list of (feature name, importance)
	n_feat: number of features want to plot
	figname: Name of figure 
	
	Returns:
	--------
	None
	"""
    xtick_labels = []

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(0,n_feat):
            ax.bar(i, Important_feat_list[i][1])
            xtick_labels.append(Important_feat_list[i][0])

    ax.set_xticks(np.arange(0,n_feat,1))
    ax.set_xticklabels(xtick_labels, rotation=90, ha='right', rotation_mode='anchor')
    ax.set_title("Top 10 feature importance")
    ax.set_ylabel("Scores")

    plt.savefig(figname, dpi=600, bbox_inches='tight', pad_inches=0.02)

def plot_scatter_mat(total_df, top_feat_df, figname):
	""" Function to plot scatter matrix of few most important features.
	
	Parameters:
	-----------
	total_df: distancs data frame
	top_feat_df: List of top features name
	figname: Name of figure 
	
	Returns:
	--------
	None
	"""
    # List of top 10 features name
    top_10_feat = top_feat_df
    z = total_df['label']

    scatter = scatter_matrix(top_10_feat, c=z, marker='.', s=40,
    figsize=(20,20), hist_kwds={'bins':15})

    plt.savefig(figname, dpi=450, bbox_inches='tight', pad_inches=0.02)

df = pd.DataFrame()
if __name_ == "__main__":
	sys = ['d_ox', 'd_red', 'b_ox', 'b_red', 'R121S', 'Y126F', 'P141G', 'With', 'WithOut']
	for Sys in sys:
		stride = 1 # To open the dataframe
		data = make_df(sys)
		

	
	
