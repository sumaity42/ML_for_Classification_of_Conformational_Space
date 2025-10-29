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
