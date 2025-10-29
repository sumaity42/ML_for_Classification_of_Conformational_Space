
def load_feat():
    """ Function to create a dictonary to upload pdb and trajectroy files
    for all systems
    
    Returns
    -------
    dict
        A dictionary where keys are features and targets are dictionaries with
        'pkl' and 'target' keys.
    """
    sys_load = {
        'd_ox': {
            'pkl': '../CA_CA_Distance_MDAnalysis_data/d_ox_CA_distances_MDAanalysis.pkl',
            'target': 0
            },
        
        'd_red': {
            'pkl': '../CA_CA_Distance_MDAnalysis_data/d_red_CA_distances_MDAanalysis.pkl',
            'target': 0
        },
        
        'b_ox': {
            'pkl': '../CA_CA_Distance_MDAnalysis_data/b_ox_CA_distances_MDAanalysis.pkl',
            'target': 1
        },
        
        'b_red': {
            'pkl': '../CA_CA_Distance_MDAnalysis_data/b_red_CA_distances_MDAanalysis.pkl',
            'target': 1
        },
        
        'R121S': {
            'pkl': '../CA_CA_Distance_MDAnalysis_data/R121S_CA_distances_MDAanalysis.pkl',
            'target': 1  
        },
        
        'Y126F': {
            'pkl': '../CA_CA_Distance_MDAnalysis_data/Y126F_CA_distances_MDAanalysis.pkl',                
            'target': 0
        },
        
        'P141G': {
            'pkl': '../CA_CA_Distance_MDAnalysis_data/P141G_CA_distances_MDAanalysis.pkl',
            'target': 0
        },

        'With': {
            'pkl': '../CA_CA_Distance_MDAnalysis_data/With_CA_distances_MDAanalysis.pkl',
            'target': 1
        },

        'WithOut': {
            'pkl':  '../CA_CA_Distance_MDAnalysis_data/WithOut_CA_distances_MDAanalysis.pkl',
            'target': 1
        }
    }
    
    return (sys_load)

