# ML_for_Classification_of_Conformational_Space
This repository contains the data and the scripts used for the manuscript "bPAC paper".

# Building conda environment
We suggest running the following command to create a conda environment called MLClassify:

conda env create -f environment.yml

# Repo details
# 1. Features:
The features are distances between Calpha pair if its less than 15 Å.

# 2. Workflow:
The workflow is organized as follows:
Add pdb and trajectory files location in the Create_Dictionary.py file.
Run Main_Script_residue.py for all systems to create unique Calpha pair list based on cutoff.
Run Main_Script_dist.py and save dataframe only for unique pairs.
Run RF_model.py and plot scatter_matrix, feature_importance
