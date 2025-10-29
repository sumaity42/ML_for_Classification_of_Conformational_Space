
def load_sys():
    """ Function to create a dictonary to upload pdb and trajectroy files
    for all systems
    
    Returns
    -------
    dict
        A dictionary where keys are pdb file and dcd files path are dictionaries with
        'pdb', 'dcd1', 'dcd2', and 'dcd3' keys.
    """
    sys_dict = {
        'd_ox': {
            'pdb': '../../dark/simulation/MD-simulation/oxidized/build/dark_ox_apo_prot_only_initial.pdb',
            'dcd1': '../../dark/simulation/MD-simulation/oxidized/Production/aligned-dark-ox-prot-only-run1.dcd',
            'dcd2': '../../dark/simulation/MD-simulation/oxidized/Production/aligned-dark-ox-prot-only-run2.dcd',
            'dcd3': '../../dark/simulation/MD-simulation/oxidized/Production/aligned-dark-ox-prot-only-run3.dcd'
        },
        
        'd_red': {
            'pdb': '../../dark/simulation/MD-simulation/reduced/build/dark_red_apo_prot_only_initial.pdb',
            'dcd1': '../../dark/simulation/MD-simulation/reduced/Production/aligned-dark-red-prot-only-run1.dcd',
            'dcd2': '../../dark/simulation/MD-simulation/reduced/Production/aligned-dark-red-prot-only-run2.dcd',
            'dcd3': '../../dark/simulation/MD-simulation/reduced/Production/aligned-dark-red-prot-only-run3.dcd'
        },
        
        # bright state dcd(s) are not aligned and also for whole system
        'b_ox': {
            'pdb': '../../bright/simulation/MD-simulation/Corrected_System/oxidized/build/bright_ox_ionize.pdb',
            'dcd1': '../../bright/simulation/MD-simulation/Corrected_System/oxidized/Production/prod_bright_ox_NPT_run1.dcd',
            'dcd2': '../../bright/simulation/MD-simulation/Corrected_System/oxidized/Production/prod_bright_ox_NPT_run2.dcd',
            'dcd3': '../../bright/simulation/MD-simulation/Corrected_System/oxidized/Production/prod_bright_ox_NPT_run3.dcd'
        },
        
        'b_red': {
            'pdb': '../../bright/simulation/MD-simulation/Corrected_System/reduced/build/bright_red_ionize.pdb',
            'dcd1': '../../bright/simulation/MD-simulation/Corrected_System/reduced/Production/prod_bright_red_NPT_run1.dcd',
            'dcd2': '../../bright/simulation/MD-simulation/Corrected_System/reduced/Production/prod_bright_red_NPT_run2.dcd',
            'dcd3': '../../bright/simulation/MD-simulation/Corrected_System/reduced/Production/prod_bright_red_NPT_run3.dcd'
        },
        
        'R121S': {
            'pdb': '../../Mutation_analysis/R121S/dark/MD_simulation/oxidized/build/dark_ox_R121S_apo_prot_only_initial.pdb',
            'dcd1': '../../Mutation_analysis/R121S/dark/MD_simulation/oxidized/Production/aligned-dark-ox-prot-only-R121S-run1.dcd',
            'dcd2': '../../Mutation_analysis/R121S/dark/MD_simulation/oxidized/Production/aligned-dark-ox-prot-only-R121S-run2.dcd',
            'dcd3': '../../Mutation_analysis/R121S/dark/MD_simulation/oxidized/Production/aligned-dark-ox-prot-only-R121S-run3.dcd'  
        },
        
        'Y126F': {
            'pdb': '../../Mutation_analysis/Y126F/dark/MD_simulation/oxidized/build/Initial_prot_only_Y126F.pdb',                
            'dcd1': '../../Mutation_analysis/Y126F/dark/MD_simulation/oxidized/Production/aligned-dark-ox-prot-only-Y126F-run1.dcd',
            'dcd2': '../../Mutation_analysis/Y126F/dark/MD_simulation/oxidized/Production/aligned-dark-ox-prot-only-Y126F-run2.dcd',
            'dcd3': '../../Mutation_analysis/Y126F/dark/MD_simulation/oxidized/Production/aligned-dark-ox-prot-only-Y126F-run3.dcd'
        },
        
        'P141G': {
            'pdb': '../../Mutation_analysis/P141G/dark/MD_simulation/oxidized/build/dark_ox_P141G_apo_prot_only_initial.pdb',
            'dcd1': '../../Mutation_analysis/P141G/dark/MD_simulation/oxidized/Production/aligned-dark-ox-prot-only-P141G-run1.dcd',
            'dcd2': '../../Mutation_analysis/P141G/dark/MD_simulation/oxidized/Production/aligned-dark-ox-prot-only-P141G-run2.dcd',
            'dcd3': '../../Mutation_analysis/P141G/dark/MD_simulation/oxidized/Production/aligned-dark-ox-prot-only-P141G-run3.dcd'
        },

        'With': {
            'pdb': '../../Mutation_analysis/K78C_T115C_W/dark/MD_simulation/oxidized/build/dark_ox_K78C_T115C_W_apo_prot_only_initial.pdb',
            'dcd1': '../../Mutation_analysis/K78C_T115C_W/dark/MD_simulation/oxidized/Production/aligned-dark-ox-prot-only-K78C_T115C_W_run1.dcd',
            'dcd2': '../../Mutation_analysis/K78C_T115C_W/dark/MD_simulation/oxidized/Production/aligned-dark-ox-prot-only-K78C_T115C_W_run2.dcd',
            'dcd3': '../../Mutation_analysis/K78C_T115C_W/dark/MD_simulation/oxidized/Production/aligned-dark-ox-prot-only-K78C_T115C_W_run3.dcd'
        },

        'WithOut': {
            'pdb':  '../../Mutation_analysis/K78C_T115C_WO/dark/MD_simulation/oxidized/build/dark_ox_K78C_T115C_WO_apo_prot_only_initial.pdb',
            'dcd1': '../../Mutation_analysis/K78C_T115C_WO/dark/MD_simulation/oxidized/Production/aligned-dark-ox-prot-only-K78C_T115C_WO_run1.dcd',
            'dcd2': '../../Mutation_analysis/K78C_T115C_WO/dark/MD_simulation/oxidized/Production/aligned-dark-ox-prot-only-K78C_T115C_WO_run2.dcd',
            'dcd3': '../../Mutation_analysis/K78C_T115C_WO/dark/MD_simulation/oxidized/Production/aligned-dark-ox-prot-only-K78C_T115C_WO_run3.dcd'
        }

    }
    
    return (sys_dict)


