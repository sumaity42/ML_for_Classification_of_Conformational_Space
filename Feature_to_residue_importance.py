import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

plt.rcParams['font.size'] = 15
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.figsize'] = (8,5)

# Add all the residue importance
def sum_elements(imp_array):
    resid = []
    for i in imp_array:
        if i[0] not in resid:
            resid.append(i[0])

    import_sum = []
    for i in resid:
        su = 0
        for j in imp_array:
            if j[0] == i:
                su = float(su) + float(j[1])
                
        import_sum.append([i, su])
    return (import_sum)

def calc_residue_imp(datafile):
    with open(datafile, "rb") as fp:
        d = pickle.load(fp)

    Importance_array = np.array([(col, d[col].iloc[0]) for col in d.columns])

    print('Total number of features: {}'.format(len(Importance_array)))

    residue_importance = []

    for i in range(0, len(Importance_array)-1):
        importance = Importance_array[i][1].astype(float)
        res = Importance_array[i][0].split('_')
    
        res_1 = res[0] + '_' + res[1]
        res_2 = res[2] + '_' + res[3]
    
        residue_importance.append([res_1, res_2, importance])

    # Check for repetation
    for i in residue_importance:
        res1 = i[0] + '_' + i[1]
        for j in residue_importance:
            res2 = j[1] + '_' + j[0] 
            if res1 == res2:
                print(res1, res2)

    residue_list = []
    for i in residue_importance:
        residue_list.append([i[0], i[2]])
        residue_list.append([i[1], i[2]])

    # Total importance for each resideus
    Total_imp = sum_elements(residue_list)
    Total_imp = np.array(Total_imp)

    # Convert 'Resid_Segname' to 'Residue'
    Final_Residue_Imp = []

    for i in range(len(Total_imp)):
        res = Total_imp[i][0].split('_')
        importance = float(Total_imp[i][1])

        if res[1] == 'B':
            res_info = int(res[0]) + 350
        else:
            res_info = int(res[0])
    
        #print(res, res_info, type(res_info), importance)
        Final_Residue_Imp.append([res_info, importance])
    
    #Final_Residue_Imp = np.array(Final_Residue_Imp)
    print('Total number of residues: {}'.format(len(Final_Residue_Imp)))

    return (Final_Residue_Imp)

def calc_norm(data):
    # Normalize importance value 
    MIN = np.min(data.astype(float))
    MAX = np.max(data.astype(float))
    Norm_imp = (data.astype(float) - MIN) / (MAX - MIN)

    return (Norm_imp)
    
def imp_plot(x, y, figname):
    plt.plot(x[:350], y[:350], label='Chain A')
    plt.plot(x[350:], y[350:], label='Chain B')

    plt.xlabel("# residue")
    plt.ylabel('Importance score')     

    plt.xlim(0, 700)
    plt.ylim(0, 1)

    plt.legend(frameon=False)

    #plt.show()
    plt.savefig(figname, dpi=450, pad_inches=0.03, bbox_inches='tight')
    plt.close()

def save_norm_imp(n_res, imp_val, files):
    with open(files, 'w') as file:
        for i in range (700):
            res = int(n_res[i])
            val = '%.2f' %imp_val[i]
                  
            file.write(f'{res}\t {val}\n')

rf_imp = calc_residue_imp('RF_Mean_feature_importance.pkl') 
etc_imp = calc_residue_imp('ETC_Mean_feature_importance.pkl') 

rf_imp = np.array(rf_imp)
etc_imp = np.array(etc_imp)

rf_imp = rf_imp[rf_imp[:, 0].argsort()]
etc_imp = etc_imp[etc_imp[:, 0].argsort()]

x_val = rf_imp[:, 0].astype(float).astype(int)
rf_y_val = rf_imp[:, 1]
etc_y_val = etc_imp[:, 1]

norm_rf_imp = calc_norm(rf_y_val)
norm_etc_imp = calc_norm(etc_y_val)

imp_plot(x_val, norm_rf_imp, 'RF_Residue_Importance_using_function.png')

imp_plot(x_val, norm_etc_imp, 'ETC_Residue_Importance_using_function.png')    

# Plot RF and ETC values together
plt.plot(x_val, norm_rf_imp, label='RF')
plt.plot(x_val, norm_etc_imp, label='ETC')

plt.axvline(x=350, ymin=0, ymax=1, ls='--', c='black')

plt.xlabel("# residue")
plt.ylabel('Importance score')     
#plt.title("Random forest Model")

plt.xlim(0, 700)
plt.ylim(0, 1)

plt.text(15, 0.92, 'Chain A')
plt.text(365, 0.92, 'Chain B')

plt.legend(frameon=False)

#plt.show()
plt.savefig('RF_ETC_Residue_Importance_using_function.png', dpi=450, pad_inches=0.03, bbox_inches='tight')
plt.close()

# Save Normalize importance for RF
save_norm_imp(x_val, norm_rf_imp, 'RF.dat')

# Save Normalize importance for ETC
save_norm_imp(x_val, norm_etc_imp, 'ETC.dat')

