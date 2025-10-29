from Create_Dictionary import load_sys
from Calculate_dist import calc_dist

import numpy as np
import pandas as pd
import pickle

with open('Unique_Sorted_15_CA_distances.pkl', 'rb') as f:
    data = pickle.load(f)

unique_col = data.columns
print('Shape of the sorted and unique dataframe: {}'.format(data.shape))

if __name__ == "__main__":

    sys = ['With', 'WithOut'] # 'R121S', 'Y126F', 'P141G', 'd_ox', 'd_red'] #, 'b_ox', 'b_red']

    for Sys in sys:
        system = load_sys()[Sys]
        PDB_Name = system['pdb']
        traj1 = system['dcd1']
        traj2 = system['dcd2']
        traj3 = system['dcd3']
        print('pdb filename for {}: {}'.format(Sys, PDB_Name.split('/')[-1]))

        dist_df = calc_dist(PDB_Name, traj1, traj2, traj3, unique_col)

        save_Name = Sys + '_CA_distances_MDAanalysis.pkl'
        dist_df.to_pickle(save_Name)

        print('  ')

