from Create_Dictionary import load_sys
from Create_Initial_Sorted_df import make_sorted_df
from Calculate_dist import calc_dist

import numpy as np
import pandas as pd
import pickle

df = pd.DataFrame()

if __name__ == "__main__":

    sys = ['d_ox', 'd_red', 'b_ox', 'b_red', 'R121S', 'Y126F', 'P141G', 'With', 'WithOut']
    cutoff = 15

    for Sys in sys:
        system = load_sys()[Sys]
        PDB_Name = system['pdb']
        print('pdb filename for {}: {}'.format(Sys, PDB_Name.split('/')[-1]))

        dist_df = make_sorted_df(PDB_Name, cutoff)
        #df.join(dist_df)

        df[dist_df.columns] = dist_df.values
        print('Shape of the df: {}'.format(dist_df.shape))
        print('Shape of the append_df: {}'.format(df.shape))
        print('  ')

unique_col = np.unique(df.columns)
unique_Idx = [df.columns.get_loc(unique_col[i]) for i in range(len(unique_col))]
unique_df = df.iloc[:, unique_Idx]

print('Shape of the sorted and unique dataframe: {}'.format(unique_df.shape))

unique_df.to_pickle('Unique_Sorted_15_CA_distances.pkl')

