import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix

# Load features
from Create_Dictionary import load_sys
from Plot_1 import plot_lc, plot_kde_1, plot_kde_2

# Scikit-learn utilities
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold, ShuffleSplit
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier

import pickle

import time
from joblib import Parallel, delayed, parallel_backend

start_time = time.time()

############################################################
# Specify all the variables here
############################################################
stride = 1 # To open the dataframe
n_threads = 16
split_size = 0.2

kde_1_figname = 'ETC_KDE_plot_1.png'
kde_2_figname = 'ETC_KDE_plot_2.png'
lc_figname = 'ETC_Learning_Curve.png'

SYSTEMS = ['d_ox', 'b_ox', 'b_red', 'd_red', 'R121S', 'Y126F', 'P141G', 'With', 'WithOut']

ETC = ExtraTreesClassifier()

parameter_set = {
        'n_estimators' : [400],
        'max_depth' : [50], # 20, 10], # default 
        'criterion' : ['gini'], # 'entropy'] #, 'log_loss'],
}

# Correlated features names (columns)
with open ('Correlated_feature_from_all_data.pkl', 'rb') as f:
    DF = pickle.load(f)
f.close()
drop_col = DF.columns

############################################################

def make_df(pkl, target, system_index, Sys):

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

    df_float16['system_index'] = system_index
    df_float16['system_index'].astype(np.int16)

    return (df_float16.iloc[::stride, :])

# Load multiple systems in parallel
def load_all_system():
    sys_list = SYSTEMS
    system = load_sys()
    jobs = [(system[Sys]['pkl'], system[Sys]['target'], system[Sys]['system_index'], Sys) for Sys in sys_list]
    print('Loading {} systems using {} parallel workers'.format(len(jobs), n_threads))

    def load_one(pkl, target, system_index, Sys):
        df = make_df(pkl, target, system_index, Sys)
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
    load = Parallel(n_jobs=n_threads)(delayed(load_one)(pkl, target, system_index, Sys) for pkl, target, system_index, Sys in jobs)
    training_DF, test_DF = zip(*load)
    
    Train_df = pd.concat(training_DF, ignore_index=True)
    Test_df = pd.concat(test_DF, ignore_index=True)

    print('Combined training dataframe shape= {}'.format(Train_df.shape))
    print('Combined testing dataframe shape= {}'.format(Test_df.shape))

    return (Train_df, Test_df)

def main():
    training_df, test_df = load_all_system()

    # Shuffle the training and test data individually
    shuffle_train_df = shuffle(training_df, random_state=20)
    shuffle_test_df = shuffle(test_df, random_state=20)

    X_train = shuffle_train_df.iloc[:,:-2]
    y_train = shuffle_train_df['label']
    train_grs = shuffle_train_df['system_index']

    X_test = shuffle_test_df.iloc[:,:-2]
    y_test = shuffle_test_df['label']
    test_grs = shuffle_test_df['system_index']

    print('Shape of the training features: {}'.format(X_train.shape))
    print('Shape of the test features: {}'.format(X_test.shape))

    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

    num_features = X_train.shape[1]
   
    X_train, y_train, train_grs = shuffle(X_train, y_train, train_grs, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for i, (train_idx, test_idx) in enumerate(skf.split(X_train, train_grs)):
        print(f'Fold: {i}')
        print('Number of dark_ox in training: {} in test:{}'.format(np.sum(train_grs[train_idx]==1), np.sum(train_grs[test_idx]==1)))
        print('Number of dark_red in training: {} in test:{}'.format(np.sum(train_grs[train_idx]==2), np.sum(train_grs[test_idx]==2)))
        print('Number of bright_ox in training: {} in test:{}'.format(np.sum(train_grs[train_idx]==3), np.sum(train_grs[test_idx]==3)))
        print('Number of bright_red in training: {} in test:{}'.format(np.sum(train_grs[train_idx]==4), np.sum(train_grs[test_idx]==4)))
        print('Number of R121S in training: {} in test:{}'.format(np.sum(train_grs[train_idx]==5), np.sum(train_grs[test_idx]==5)))
        print('Number of Y126F in training: {} in test:{}'.format(np.sum(train_grs[train_idx]==6), np.sum(train_grs[test_idx]==6)))
        print('Number of P141G in training: {} in test:{}'.format(np.sum(train_grs[train_idx]==7), np.sum(train_grs[test_idx]==7)))
        print('Number of With in training: {} in test:{}'.format(np.sum(train_grs[train_idx]==8), np.sum(train_grs[test_idx]==8)))
        print('Number of Without in training: {} in test:{}'.format(np.sum(train_grs[train_idx]==9), np.sum(train_grs[test_idx]==9)))

    grid_search = GridSearchCV(
            estimator=ETC,
            param_grid=parameter_set,
            n_jobs=-1,
            cv=skf,
            verbose=1
            )
    
    with parallel_backend('loky', n_jobs=n_threads):
        grid_search.fit(X_train, y_train)

    print('Best parameters found: {}'.format(grid_search.best_params_))
    print('Best cross-validation score: {}'.format(grid_search.best_score_))

    # Print results
    grid_search.cv_results_

    best_ETC_model = grid_search.best_estimator_
    y_pred = best_ETC_model.predict(X_test)

    Score = best_ETC_model.score(X_test, y_test)
    print('Accuracy score of the best ETC model: {}'.format(Score))

    print('classification report on train set')
    print(classification_report(y_train, best_ETC_model.predict(X_train)))

    print('classification report on test set')
    print(classification_report(y_test, y_pred))
    #print(confusion_matrix(y_test, y_pred))
    
    print('Confusion matrix')
    print(multilabel_confusion_matrix(y_test, y_pred))

    print('ROC accuracy on traing set: {}'.format(roc_auc_score(y_train, best_ETC_model.predict_proba(X_train)[:,1])))
    print('ROC accuracy on test set: {}'.format(roc_auc_score(y_test, best_ETC_model.predict_proba(X_test)[:,1])))

    # Kernel density plot for train and validation set
    plot_kde_1(best_ETC_model, X_train, y_train, skf, kde_1_figname)
    plot_kde_2(best_ETC_model, X_train, y_train, skf, kde_2_figname)

    plot_lc(best_ETC_model, X_train, y_train, skf, lc_figname)


if __name__ == "__main__":
    main()

end_time = time.time()
elaspsed_time = end_time - start_time
print('Time required : {} with {} threads'.format(elaspsed_time, n_threads))


