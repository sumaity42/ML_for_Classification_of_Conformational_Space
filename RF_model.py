import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix

# Try Numba
from numba import njit, prange


# Load features
from Create_Dictionary import load_sys

# ML training and prediction
from ML_Pred import ML_Model

# Plotting function
#from Plot import plot_feat_Imp, plot_scatter_mat
from Plot_1 import plot_feat_Imp, plot_scatter_mat

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

import time
from joblib import Parallel, delayed, parallel_backend
#import dask.dataframe as dd

start_time = time.time()

############################################################
# Specify all the variables here
############################################################
stride = 1 # To open the dataframe
n_threads = 16
cutoff_th = 0.9
iterate = 50

SYSTEMS = ['d_ox', 'b_ox', 'b_red', 'd_red', 'R121S', 'Y126F', 'P141G', 'With', 'WithOut']

RF = RandomForestClassifier(max_depth=50, max_features=80, 
        min_samples_leaf=2, n_estimators=400)

# pkl filename to save imporatnce
save_Mean_Imp = 'Mean_feature_importance.pkl'

# Figure name for residue importance
figname_Imp = 'RF_Top20_Feature_Imp.png'

# Figure name for top 10 features scatter matrix
figname_scat = 'Scatter_Matrix_Top10_Feat.png'
############################################################

def make_df(pkl, target):

    with open(pkl, 'rb') as f:
        df = pickle.load(f)
    f.close()

    # Add target : active (1) or inactive (0)
    df['label'] = target
	
    return (df.iloc[::stride, :])

#@njit
def Numba_Corr(features):
    """Convert features dataframe to numpy array, Calculate correlation,
    and convert back numpy_array to dataframe.
    """
    feat_numpy = features.to_numpy()
    corr_numpy = np.corrcoef(feat_numpy, rowvar=False)
    abs_corr_numpy = np.abs(corr_numpy)

    # Convert to dataframe
    corr_numba = pd.DataFrame(abs_corr_numpy, index=features.columns, 
            columns=features.columns)

    return (corr_numba)

# Load multiple systems in parallel
def load_all_system():
    sys_list = SYSTEMS
    system = load_sys()

    jobs = [(system[Sys]['pkl'], system[Sys]['target'], Sys) for Sys in sys_list]
    print('Loading {} systems using {} parallel workers'.format(len(jobs), n_threads))

    def load_one(pkl, target, Sys):
        df = make_df(pkl, target)
        print('Loaded system: {}, Shape={}'.format(Sys, df.shape))

        return (df)

    # parallel loading
    dfs = Parallel(n_jobs=n_threads)(delayed(load_one)(pkl, target, Sys) for pkl, target, Sys in jobs)
    
    # Concatenate all systems
    df = pd.concat(dfs, ignore_index=True)
    print('Combined dataframe shape= {}'.format(df.shape))

    return (df)

def main():
    df = load_all_system()

    # Shuffle the data
    shuffle_df = shuffle(df, random_state=20)

    features = shuffle_df.iloc[:, :-1]
    
    #features = df.iloc[:, :-1]
    print('Shape of the total features: {}'.format(features.shape))

    corr_pre = Numba_Corr(features)
    print('Shape of the pre-coorelation matrix: {}'.format(corr_pre.shape))

    # Dataframe to the feature importance score
    importance = pd.DataFrame(columns=df.columns)

    for run in range(0, iterate):
        print('Iteration number: {}'.format(run))

        # Shuffle correlation matrix
        arr = np.arange(len(features.columns))
        print('Length of arr: {}'.format(len(arr)))

        np.random.shuffle(arr)
        corr_matrix = corr_pre.iloc[arr,arr]

        # Select upper triangluar part 
        upper_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        drop_col = [column for column in upper_corr.columns if any(upper_corr[column] > cutoff_th)]
        print('Number of features to drop: {}'.format(len(drop_col)))

        df_dropped = shuffle_df.drop(columns = drop_col)
        
        #df_dropped = df.drop(columns = drop_col)
        print('Shape of the features after drop: {}'.format(df_dropped.shape))

        X = df_dropped.iloc[:,:-1]
        y = df_dropped['label']

        X, y = X.to_numpy(), y.to_numpy()
        num_features = X.shape[1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        with parallel_backend('loky', n_jobs=n_threads):
            RF.fit(X_train, y_train)

        Score = RF.score(X_test, y_test)
        print('Accuracy score of the RF model: {}'.format(Score))

        coef = pd.DataFrame(np.reshape(RF.feature_importances_, (1, num_features)), 
            columns=df_dropped.columns[:-1])

        importance = pd.concat([importance, coef], ignore_index=True)

    # replace 'NaN' with '0'
    importance_filled = importance.fillna(0)
    mean_importance = abs(importance_filled.mean().to_frame().T)

    # Save mean importance
    with open(save_Mean_Imp, 'wb') as fp:
        pickle.dump(mean_importance, fp)

    ######################
    # Add a function to convert feature_importance to residue_importance
    # I'm working on it
    ######################

    # Plot scatter matrix of top 10 features
    plot_scatter_mat(df, mean_importance, 10, figname_scat)

    # Plot feature importance of top 20 features
    plot_feat_Imp(mean_importance, 20, figname_Imp)

if __name__ == "__main__":
    main()

end_time = time.time()
elaspsed_time = end_time - start_time
print('Time required : {} with {} threads'.format(elaspsed_time, n_threads))
