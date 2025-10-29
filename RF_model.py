import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix

# Scikit-learn utilities
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

import pickle

R121S = '../CA_CA_Distance_MDAnalysis_data/R121S_CA_distances_MDAanalysis.pkl'
Y126F = '../CA_CA_Distance_MDAnalysis_data/Y126F_CA_distances_MDAanalysis.pkl'

def make_df(pkl1, pkl2, cut):
	with open(pkl1, 'rb') as f1:
		df1 = pickle.load(f1)
	f1.close()

	with open(pkl2, 'rb') as f2:
		df2 = pickle.load(f2)
	f2.close()

	df = pd.concat([df1, df2], ignore_index=True)
	print('Shape of the total dataframe: {}'.format(df.shape))

	df['label'] = 0

	df.loc[:30000, 'label'] = 0
	df.loc[30000:, 'label'] = 1

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

stride = 1 # To open the dataframe
data = make_df(R121S, Y126F, stride)

X = data.iloc[:, :-1]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#scaled_X_train = scaler.fit_transform(X_train)
#scaled_X_test = scaler.transform(X_test)

# # Random Forest Classifier Model
RF = RandomForestClassifier(max_depth=50, max_features=80, min_samples_leaf=2,
                           n_estimators=400)

RF.fit(X_train, y_train)
RF.score(X_test, y_test)

feature_importances = RF.feature_importances_
feature_names = data.columns

sorted_feature_imp = sorted(list(zip(feature_names, feature_importances.flatten())), 
                            key=lambda x:x[1], reverse=True)

Name = 'Feature_Importance_RF_R121S_Y126F.pkl'
with open(Name, "wb") as fp:
    pickle.dump(sorted_feature_imp, fp)

#with open("Feature_Importance_RF_5", "rb") as fp:
#    b = pickle.load(fp)


# Plot feature Importance
n_feat = 10
List = []

for i in range(0, n_feat):
    List.append(sorted_feature_imp[i][0])

top_10_feat = data[List]

plt.rcParams['font.size'] = 15
plt.rcParams['axes.linewidth'] = 1.5

plot_feat_Imp(sorted_feature_imp, n_feat, 'RF_Top20_Feature_Imp_R121S_Y126F.png')

plot_scatter_mat(data, top_10_feat, 'Scatter_Matrix_Top10_Feat_R121S_Y126F.png')

