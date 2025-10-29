import MDAnalysis as mda
from MDAnalysis.analysis import distances

import numpy as np
import pandas as pd
import pickle

import warnings
# suppress some MDAnalysis warnings when writing PDB files
warnings.filterwarnings('ignore')

def make_sorted_df(PDB, cutoff):

    u = mda.Universe(PDB)

    sel = u.select_atoms('name CA and resid 1-350')
    print('Total number of CA atoms: {}'.format(len(sel)))

    # Distances in initial pdb file
    dist_pdb = distances.distance_array(sel.positions, sel.positions)

    res_mat = np.zeros((len(sel), len(sel)), dtype='object')

    for i in range (0, len(sel)):
        for j in range(0, len(sel)):
            if i < 350 and j < 350:
                d = str(sel.resids[i]) + '_A_' + str(sel.resids[j]) + '_A'
                res_mat[i, j] = d
            elif i >= 350 and j < 350:
                d = str(sel.resids[i]) + '_B_' + str(sel.resids[j])+ '_A'
                res_mat[i, j] = d
            elif i < 350 and j >= 350:
                d = str(sel.resids[i]) + '_A_' + str(sel.resids[j])+ '_B'
                res_mat[i, j] = d
            else:
                d = str(sel.resids[i]) + '_B_' + str(sel.resids[j])+ '_B'
                res_mat[i, j] = d

    # Convert matrix to upper-triangular matrix > columns
    up_res_mat = np.triu(res_mat, k=1)
    up_res_mat = up_res_mat[up_res_mat != 0]
    up_res_flat = up_res_mat.flatten()

    up_dist_pdb = np.triu(dist_pdb, k=1)
    up_dist_pdb = up_dist_pdb[up_dist_pdb != 0]
    up_dist_pdb_conv = up_dist_pdb.reshape(1, -1)

    Initial_df = pd.DataFrame(data=up_dist_pdb_conv.astype(float),
            columns=up_res_flat.astype(str))

    # Sort distances based on "cutoff" value
    cols = Initial_df.columns
    sorted_col = []
    for col in cols:
        if Initial_df[col].item() <= cutoff:
            sorted_col.append(col)

    # Index for sorted columns
    Idx = [Initial_df.columns.get_loc(sorted_col[i]) for i in range(len(sorted_col))]

    # Sorted dataframe
    sorted_df = Initial_df.iloc[:, Idx]

    print('Total number of distance pairs: {}'.format(Initial_df.shape[1]))
    print('Sorted number of distance pairs: {}'.format(sorted_df.shape[1]))

    return (sorted_df)

