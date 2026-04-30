import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix

# Load features
from Create_Dictionary import load_sys

# Plotting function
from Plot_1 import plot_feat_Imp, plot_scatter_mat

# Scikit-learn utilities
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold, ShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, classification_report

import pickle

import time
from joblib import Parallel, delayed, parallel_backend

start_time = time.time()

############################################################
# Specify all the variables here
############################################################
stride = 1 # To open the dataframe
n_threads = 16
cutoff_th = 0.9
iterate = 50

SYSTEMS = ['d_ox', 'b_ox', 'b_red', 'd_red', 'R121S', 'Y126F', 'P141G', 'With', 'WithOut']

ETC = ExtraTreesClassifier(n_estimators=400, max_depth=50, criterion='gini')

# pkl filename to save imporatnce
save_Mean_Imp = 'ETC_Mean_feature_importance.pkl'

# Figure name for residue importance
figname_Imp = 'ETC_Top20_Feature_Imp.png'

# Figure name for top 10 features scatter matrix
figname_scat = 'ETC_Scatter_Matrix_Top10_Feat.png'

# For train/test split
split_size = 0.2

# Correlated features names (columns)
with open ('Correlated_feature_from_all_data.pkl', 'rb') as f:
    DF = pickle.load(f)
f.close()
drop_col = DF.columns

############################################################
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

# Load multiple systems in parallel
def load_all_system():
    sys_list = SYSTEMS
    system = load_sys()
    jobs = [(system[Sys]['pkl'], system[Sys]['target'], Sys) for Sys in sys_list]
    print('Loading {} systems using {} parallel workers'.format(len(jobs), n_threads))

    def load_one(pkl, target, Sys):
        df = make_df(pkl, target, Sys)
        print('Loaded system: {}, Shape={}'.format(Sys, df.shape))

        # Drop correlated features
        df_dropped = df.drop(columns = drop_col)
        print('Dataset size after dropping correlated features: {}'.format(df_dropped.shape))

        # Shuffle the data before breaking into train and test set 
        shuffle_df_dropped = shuffle(df_dropped, random_state=20)

        training_size = int(shuffle_df_dropped.shape[0] * split_size)

        train_df = shuffle_df_dropped[:-training_size]
        tesT_df = shuffle_df_dropped[-training_size:]

        return (train_df, tesT_df)

    # parallel loading
    load = Parallel(n_jobs=n_threads)(delayed(load_one)(pkl, target, Sys) for pkl, target, Sys in jobs)
    training_DF, test_DF = zip(*load)

    Train_df = pd.concat(training_DF, ignore_index=True)
    Test_df = pd.concat(test_DF, ignore_index=True)

    print('Combined training dataframe shape= {}'.format(Train_df.shape))
    print('Combined testing dataframe shape= {}'.format(Test_df.shape))

    return (Train_df, Test_df)

def main():

    df = pd.concat([load_all_system()[0], load_all_system()[1]], ignore_index=True)
    print('Total size of the dataset: {}'.format(df.shape))

    # Dataframe to the feature importance score
    importance = pd.DataFrame(columns=df.columns[:-1])

    for run in range(0, iterate):
        print('Iteration number: {}'.format(run))

        training_df, test_df = load_all_system()

        # Shuffle the training and test data individually
        shuffle_train_df = shuffle(training_df, random_state=20)
        shuffle_test_df = shuffle(test_df, random_state=20)

        X_train = shuffle_train_df.iloc[:,:-1]
        y_train = shuffle_train_df['label']

        X_test = shuffle_test_df.iloc[:,:-1]
        y_test = shuffle_test_df['label']

        print('Shape of the training features: {}'.format(X_train.shape))
        print('Shape of the test features: {}'.format(X_test.shape))

        X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
        X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

        num_features = X_train.shape[1]

        with parallel_backend('loky', n_jobs=n_threads):
            ETC.fit(X_train, y_train)

        Score = ETC.score(X_test, y_test)
        print('Accuracy score of the ETC model: {}'.format(Score))

        coef = pd.DataFrame(np.reshape(ETC.feature_importances_, (1, num_features)),
            columns=df.columns[:-1])

        importance = pd.concat([importance, coef], ignore_index=True)
        
        print('classification report on train set')
        print(classification_report(y_train, ETC.predict(X_train)))

        print('classification report on test set')
        print(classification_report(y_test, ETC.predict(X_test)))

        print('Confusion matrix')
        print(multilabel_confusion_matrix(y_test, ETC.predict(X_test)))

    # replace 'NaN' with '0'
    importance_filled = importance.fillna(0)
    mean_importance = abs(importance_filled.mean().to_frame().T)

    # Save mean importance
    with open(save_Mean_Imp, 'wb') as fp:
        pickle.dump(mean_importance, fp)

    # Plot scatter matrix of top 10 features
    plot_scatter_mat(df, mean_importance, 10, figname_scat)

    # Plot feature importance of top 20 features
    plot_feat_Imp(mean_importance, 20, figname_Imp)

if __name__ == "__main__":
    main()

end_time = time.time()
elaspsed_time = end_time - start_time
print('Time required : {} with {} threads'.format(elaspsed_time, n_threads))

