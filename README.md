# ML_for_Classification_of_Conformational_Space
This repository contains the data and the scripts used for the manuscript "Identification of Key Residues in Allosteric Signaling of Photoactivated Adenylyl Cyclase".

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

iv. Add distance dataframe ('pkl'), 'target' and 'system_index' in the Create_Dictionary.py file.

v. Run Calculate_feature_Correlation.py to calculate feature correlation and saved the correlated features name.

vi. Perform hyperparameter optimization: RF_Hyperparameter_optimization.py for Random Forest (RF) and ETC_Hyperparameter_optimization for Extratrees Classifier (ETC).

vii. With the optimized parameters train the model, make prediction and save average importance score: RF_model_final.py for RF and ETC_model_final.py for ETC.

viii. 

# Datasets
A subset of data will be uploaded shortly.
