# ML_for_Classification_of_Conformational_Space
This repository contains the data and the scripts used for the manuscript "bPAC paper".

# Building conda environment
We suggest running the following command to create a conda environment called MLClassify:

conda env create -f environment.yml

# Repo details
## 1. Features:
If distances between Calpha pair is less than 15 Å, consider it as a feature.

## 2. Workflow:
The workflow is organized as follows:

i. Add pdb and trajectory files location in the Create_Dictionary.py file.

ii. Run Main_Script_residue.py for all systems to create unique Calpha pair list based on cutoff.

iii. Run Main_Script_dist.py and save dataframe only for unique pairs.

iv. Add distance dataframe and target lebel in the Create_Dictionary.py file.

v. Run RF_model.py and plot scatter_matrix, feature_importance
