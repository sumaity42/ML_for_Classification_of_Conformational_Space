import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
from matplotlib.colors import ListedColormap

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

    scatter = scatter_matrix(top_10_df, c=z, marker='.', s=40,
    figsize=(20,20), hist_kwds={'bins':15}, cmap=cmap)

    # Create legend manually
    handels = [plt.Line2D([], [], marker='o', linestyle='', color=cmap(k),
        label=f"Class {cls}") for k, cls in enumerate(classes)]

    plt.legend(handels=handels, bbox_to_anchor=(-4,-0.7), loc='lower center',
            ncol=2, fontsize=15)

    plt.savefig(figname, dpi=450, bbox_inches='tight', pad_inches=0.02)
