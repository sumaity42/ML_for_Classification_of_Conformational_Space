import numpy as np
import pandas as pd

# Load features
from Create_Dictionary import load_sys

# Scikit-learn utilities
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold, ShuffleSplit
import pickle

import time
from joblib import Parallel, delayed, parallel_backend

start_time = time.time()

############################################################
# Specify all the variables here
############################################################
stride = 1000 # To open the dataframe
n_threads = 16
cutoff_th = 0.9

SYSTEMS = ['d_ox', 'b_ox', 'b_red', 'd_red', 'R121S', 'Y126F', 'P141G', 'With', 'WithOut']

def make_df(pkl, target, Sys):

    with open(pkl, 'rb') as f:
        df = pickle.load(f)
    f.close()

    df_float16 = df[df.columns].astype(np.float16)
    mem_after = df_float16.memory_usage(deep=True).sum()/(1024*1024*1024)
    mem_before = df.memory_usage(deep=True).sum()/(1024*1024*1024)

    print(f'Memory of the {Sys} dataframe')
    print(f'Memory before: {mem_before:.2f} GB')
    print(f'Memory after: {mem_after:.2f} GB')
    print(' ')

    # Add target : active (1) or inactive (0)
    df_float16['label'] = target
    df_float16['label'].astype(np.int16)

    return (df_float16.iloc[::stride, :])

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
        df = make_df(pkl, target, Sys)
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
    print('Shape of the total features: {}'.format(features.shape))

    corr_pre = Numba_Corr(features)
    print('Shape of the pre-coorelation matrix: {}'.format(corr_pre.shape))

    # Shuffle correlation matrix
    arr = np.arange(len(features.columns))
    print('Length of arr: {}'.format(len(arr)))

    np.random.shuffle(arr)
    corr_matrix = corr_pre.iloc[arr,arr]

    # Select upper triangluar part 
    upper_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    drop_col = [column for column in upper_corr.columns if any(upper_corr[column] > cutoff_th)]
    print('Number of features to drop: {}'.format(len(drop_col)))

    #df_dropped = shuffle_df.drop(columns = drop_col)

    #print('Shape of the features after drop: {}'.format(df_dropped.shape))

    #drop_col.to_pickle('Correlated_feature_from_all_data.pkl')

    drop_df = pd.DataFrame(columns=drop_col)
    drop_df.to_pickle('Correlated_feature_from_all_data.pkl')

if __name__ == "__main__":
    main()

