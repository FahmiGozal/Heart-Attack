import math
import numpy as np
import pandas as pd

# FUNCTION 1 : Row Enhancement based on slope
def data_enhancement_slope_based(data, percentage):
    copy_data = data.copy()
    gen_data = pd.DataFrame()

    for slp_i in copy_data['slp'].unique():
        slp_data = copy_data[copy_data['slp']==slp_i]
        trtbps_std = slp_data['trtbps'].std()
        chol_std = slp_data['chol'].std()
        thalachh_std = slp_data['thalachh'].std()
        oldpeak_std = slp_data['oldpeak'].std()

        for i in range(slp_data.shape[0]):
            
            if np.random.randint(2) == 1:
                slp_data['trtbps'].values[i] += trtbps_std
            else:
                slp_data['trtbps'].values[i] -= trtbps_std
                
            if np.random.randint(2) == 1:
                slp_data['chol'].values[i] += chol_std
            else:
                slp_data['chol'].values[i] -= chol_std
                
            if np.random.randint(2) == 1:
                slp_data['thalachh'].values[i] += thalachh_std
            else:
                slp_data['thalachh'].values[i] -= thalachh_std
                
            if np.random.randint(2) == 1:
                slp_data['oldpeak'].values[i] += oldpeak_std
            else:
                slp_data['oldpeak'].values[i] -= oldpeak_std
            
        gen_data = pd.concat([gen_data, slp_data])

        extra_sample = gen_data.sample(math.floor(gen_data.shape[0]*percentage/100))
        row_enhanced_data = pd.concat([copy_data, extra_sample])
            
    return row_enhanced_data

#Function 2 : Row Enhancement based on cp then slope
def data_enhancement_cp_slope_based(data, percentage):
    copy_data = data.copy()
    gen_data = pd.DataFrame()

    for cp_i in copy_data['cp'].unique():
        cp_data = copy_data[copy_data['cp']==cp_i]
        
        for slp_i in copy_data['slp'].unique():
            slp_data = cp_data[cp_data['slp']==slp_i]
            trtbps_std = slp_data['trtbps'].std()
            chol_std = slp_data['chol'].std()
            thalachh_std = slp_data['thalachh'].std()
            oldpeak_std = slp_data['oldpeak'].std()

            for i in range(slp_data.shape[0]):
            
                if np.random.randint(2) == 1:
                    slp_data['trtbps'].values[i] += trtbps_std
                else:
                    slp_data['trtbps'].values[i] -= trtbps_std
                
                if np.random.randint(2) == 1:
                    slp_data['chol'].values[i] += chol_std
                else:
                    slp_data['chol'].values[i] -= chol_std
                
                if np.random.randint(2) == 1:
                    slp_data['thalachh'].values[i] += thalachh_std
                else:
                    slp_data['thalachh'].values[i] -= thalachh_std
                
                if np.random.randint(2) == 1:
                    slp_data['oldpeak'].values[i] += oldpeak_std
                else:
                    slp_data['oldpeak'].values[i] -= oldpeak_std
            
            gen_data = pd.concat([gen_data, slp_data])
            
            extra_sample = gen_data.sample(math.floor(gen_data.shape[0]*percentage/100))
            row_enhanced_data = pd.concat([copy_data, extra_sample])

    return row_enhanced_data