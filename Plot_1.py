import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
from matplotlib.colors import ListedColormap
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def Sort_Importance(Imp_dataframe):
    np_array = np.array([(col, Imp_dataframe[col].iloc[0]) for col in Imp_dataframe.columns])

    sorted_Imp = np_array[np_array[:,1].astype(float).argsort()[::-1]]

    return (sorted_Imp)

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
    Sorted_list = Sort_Importance(Important_feat_list)

    xtick_labels = []    

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(0,n_feat):
        #
        ax.bar(Sorted_list[i][0], Sorted_list[i][1].astype(float))
        xtick_labels.append(Sorted_list[i][0])

    #
    ax.set_xticks(np.arange(0,n_feat,1))
    ax.set_xticklabels(xtick_labels, va='center_baseline', 
            rotation=90, ha='right', rotation_mode='anchor')
    #ax.set_title("Top 10 feature importance")
    ax.set_ylabel("Importance score")

    plt.savefig(figname, dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close()

def plot_scatter_mat(total_df, Important_feat_list, n_feat, figname):
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
    
    Sorted_list = Sort_Importance(Important_feat_list)
    
    # List of top 10 features name
    top_10_feat = [Sorted_list[i][0] for i in range(10)]
    top_10_df = total_df[top_10_feat]
    
    labels = total_df['label']

    # Create colormap depending on number of classes
    classes = sorted(labels.unique())
    n_classes = len(classes)

    cmap = ListedColormap(plt.cm.tab10.colors[:n_classes])

    scatter = scatter_matrix(top_10_df, c=labels, marker='.', s=40,
    figsize=(20,20), hist_kwds={'bins':15}, cmap=cmap)

    # Create legend manually
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=cmap(k),
        label=f"Class {cls}") for k, cls in enumerate(classes)]

    plt.legend(handles=handles, bbox_to_anchor=(-4,-0.7), loc='lower center',
            ncol=2, fontsize=15)

    plt.savefig(figname, dpi=450, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def plot_lc(model, X_Train, Y_Train, splits, figname):

    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, test_scores = learning_curve(model, X_Train,
            Y_Train, scoring = 'accuracy', cv=splits, train_sizes=train_sizes)

    # Calculate training and test mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    print('Train mean: {} ans std: {}'.format(train_mean, train_std))
    print('test mean: {} ans std: {}'.format(test_mean, test_std))

    fig, ax = plt.subplots(1, figsize=(6,5))

    ax.set_xlabel('Training set size', fontsize=16)
    ax.set_ylabel('Accuracy score', fontsize=16)

    # Plot the learning curve
    ax.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training')
    ax.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    ax.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation')
    ax.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    ax.legend(loc='lower right', fontsize=12, frameon=False)

    plt.savefig(figname, bbox_inches='tight', dpi=450, pad_inches=0.03)
    plt.close()

def plot_kde_1(model, X_Train, Y_Train, splits, figname):
    train_erros = []
    val_erros = []

    for train_idx, val_idx in splits.split(X_Train, Y_Train):
        X_tr, X_val = X_Train[train_idx], X_Train[val_idx]
        y_tr, y_val = Y_Train[train_idx], Y_Train[val_idx]

        model.fit(X_tr, y_tr)

        train_probs = model.predict_proba(X_tr)[:,1]
        val_probs = model.predict_proba(X_val)[:,1]

        train_err = train_probs - y_tr
        val_error = val_probs - y_val

        train_erros.extend(train_err)
        val_erros.extend(val_error)

        print('ROC accuracy on traing set: {}'.format(roc_auc_score(y_tr, model.predict_proba(X_tr)[:,1])))
        print('ROC accuracy on validation set: {}'.format(roc_auc_score(y_val, model.predict_proba(X_val)[:,1])))


    train_erros = np.array(train_erros)
    val_erros = np.array(val_erros)

    sns.kdeplot(train_erros, label='Train', fill=True)
    sns.kdeplot(val_erros, label='Validation', fill=True)

    plt.xlabel("Prediction Error")
    plt.ylabel("Density")
    plt.legend(frameon=False)

    plt.savefig(figname, bbox_inches='tight', dpi=450, pad_inches=0.03)
    plt.close()

def plot_kde_2(model, X_Train, Y_Train, splits, figname):
    train_erros = []
    val_erros = []

    for train_idx, val_idx in splits.split(X_Train, Y_Train):
        X_tr, X_val = X_Train[train_idx], X_Train[val_idx]
        y_tr, y_val = Y_Train[train_idx], Y_Train[val_idx]

        model.fit(X_tr, y_tr)

        train_probs = model.predict_proba(X_tr)[:,1]
        val_probs = model.predict_proba(X_val)[:,1]

        train_err = np.abs(train_probs - y_tr)
        val_error = np.abs(val_probs - y_val)

        train_erros.extend(train_err)
        val_erros.extend(val_error)

    train_erros = np.array(train_erros)
    val_erros = np.array(val_erros)

    sns.kdeplot(train_erros, label='Train', fill=True)
    sns.kdeplot(val_erros, label='Validation', fill=True)

    plt.xlabel("Prediction Error")
    plt.ylabel("Density")
    plt.legend(frameon=False)

    plt.savefig(figname, bbox_inches='tight', dpi=450, pad_inches=0.03)
    plt.close()

