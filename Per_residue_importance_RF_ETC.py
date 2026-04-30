#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle


# In[2]:


plt.rcParams['font.size'] = 15
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.figsize'] = (8,5)


# ## Add all the residue importance

# In[3]:


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


# In[11]:


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


# In[47]:


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
    #plt.title("Random forest Model")

    plt.xlim(0, 700)
    plt.ylim(0, 1)

    plt.legend(frameon=False)

    #plt.show()
    plt.savefig(figname, dpi=450, pad_inches=0.03, bbox_inches='tight')
    plt.close()


# In[1]:


def save_norm_imp(n_res, imp_val, files):
    with open(files, 'w') as file:
        for i in range (700):
            res = int(n_res[i])
            val = '%.2f' %imp_val[i]
                  
            file.write(f'{res}\t {val}\n')


# In[48]:


rf_imp = calc_residue_imp('RF_Mean_feature_importance.pkl') 
etc_imp = calc_residue_imp('ETC_Mean_feature_importance.pkl') 


# In[49]:


rf_imp = np.array(rf_imp)
etc_imp = np.array(etc_imp)

rf_imp = rf_imp[rf_imp[:, 0].argsort()]
etc_imp = etc_imp[etc_imp[:, 0].argsort()]

x_val = rf_imp[:, 0].astype(float).astype(int)
rf_y_val = rf_imp[:, 1]
etc_y_val = etc_imp[:, 1]


# In[50]:


norm_rf_imp = calc_norm(rf_y_val)
norm_etc_imp = calc_norm(etc_y_val)


# In[51]:


imp_plot(x_val, norm_rf_imp, 'RF_Residue_Importance_using_function.png')


# In[52]:


imp_plot(x_val, norm_etc_imp, 'ETC_Residue_Importance_using_function.png')    


# In[55]:


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


# In[2]:


# Save Normalize importance for RF
save_norm_imp(x_val, norm_rf_imp, 'RF.dat')


# Save Normalize importance for ETC
save_norm_imp(x_val, norm_etc_imp, 'ETC.dat')


# In[ ]:





# In[ ]:





# In[59]:


joint_etc = []
with open('ETC_Residue_Importance_using_function.dat', 'w') as file:
    for i in range (700):
        res = int(x_val[i])
        val = '%.2f' %norm_etc_imp[i]

        joint_etc.append([res, val])
        #print('res: {}, imp: {}'.format(res, val))         
        file.write(f'{res}\t {val}\n')


# In[72]:


joint_etc = np.array(joint_etc)
sorted_joint_etc = joint_etc[joint_etc[:, 1].argsort()[::-1]]


# In[69]:


joint_rf = []
with open('RF_Residue_Importance_using_function.dat', 'w') as file:
    for i in range (700):
        res = int(x_val[i])
        val = '%.2f' %norm_rf_imp[i]

        joint_rf.append([res, val])
        #print('res: {}, imp: {}'.format(res, val))
                  
        file.write(f'{res}\t {val}\n')


# In[73]:


joint_rf = np.array(joint_rf)
sorted_joint_rf = joint_rf[joint_rf[:, 1].argsort()[::-1]]


# In[76]:


Sorted_imp = np.hstack((sorted_joint_etc, sorted_joint_rf))


# In[77]:


Sorted_imp


# In[80]:


df = pd.DataFrame(Sorted_imp, columns=['ETC_res', 'ETC_imp', 'RF_res', 'RF_imp'])


# In[81]:


df


# In[82]:


# Save to Excel
df.to_excel("output.xlsx") #, index=False, header=False)


# # Common residues between 1st 25 from ETC and RF model

# In[103]:


common_res = np.intersect1d(df['ETC_res'][:25], df['RF_res'][:25])


# In[104]:


print('Number of common residues: {}'.format(len(common_res)))


# In[105]:


common_res


# In[111]:


common_res[0]


# In[112]:


df['ETC_res'][:25][0]


# In[116]:


for i in range(25):
    print(df['ETC_res'][:25][i], common_res[i])
    if df['ETC_res'][:25][i] == common_res.any():
        print(df['ETC_res'][:25][i])


# In[ ]:




