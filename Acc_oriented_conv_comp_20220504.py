import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from pytictoc import TicToc

t = TicToc()

body_dir = os.getcwd()
src_mtr0 = 'D:\OneDrive - HKNC\999_Proto'
# src_mtr_M = os.path.join(src_mtr0, '06_IMU_smart_tyre\Res_20220401_v2') # Mahony
# src_mtr_A = os.path.join(src_mtr0, '06_IMU_smart_tyre\Res_20220502_AccOrient','20220503_Acc_Oriented_v3') # AO res.

src_mtr_M = os.path.join(src_mtr0, '06_IMU_smart_tyre\Res_20220426_K_city','20220504_Convention_Res') # Mahony / S2
src_mtr_A = os.path.join(src_mtr0, '06_IMU_smart_tyre\Res_20220502_AccOrient','20220504_Acc_Oriented_S2_v1') # AO res. / S2

# os.chdir(src_mtr1)
f_names_M = os.listdir(src_mtr_M)
f_names_A = os.listdir(src_mtr_A)


def data_out(f_name, sheetname, row_stt, col_stt):
    df_1 = pd.read_excel(f_name, sheet_name=sheetname)
    df_arr = np.array(df_1)
    data = df_arr[row_stt:,col_stt:]
    
    return data

# d_AO_x_2d = data_out(f_names[1], sheetname='2d_ts_X', row_stt = 0, col_stt = 1)
# d_AO_y_2d = data_out(f_names[1], sheetname='2d_ts_Y', row_stt = 0, col_stt = 1)
# d_AO_z_2d = data_out(f_names[1], sheetname='2d_ts_Z', row_stt = 0, col_stt = 1)
os.chdir(src_mtr_A)
d_AO_x_3d = data_out(f_names_A[1], sheetname='3d_ts_X', row_stt = 0, col_stt = 1)
d_AO_y_3d = data_out(f_names_A[1], sheetname='3d_ts_Y', row_stt = 0, col_stt = 1)
d_AO_z_3d = data_out(f_names_A[1], sheetname='3d_ts_Z', row_stt = 0, col_stt = 1)

os.chdir(src_mtr_M)
d_M_x_2d = data_out(f_names_M[-2], sheetname='Mahony_mod_acc_X', row_stt=1, col_stt=1)
d_M_y_2d = data_out(f_names_M[-2], sheetname='Mahony_mod_acc_Y', row_stt=1, col_stt=1)
d_M_z_2d = data_out(f_names_M[-2], sheetname='Mahony_mod_acc_Z', row_stt=1, col_stt=1)

d_M_x_2d_103 = data_out(f_names_M[-1], sheetname='Mahony_mod_acc_X', row_stt=1, col_stt=1)
d_M_y_2d_103 = data_out(f_names_M[-1], sheetname='Mahony_mod_acc_Y', row_stt=1, col_stt=1)
d_M_z_2d_103 = data_out(f_names_M[-1], sheetname='Mahony_mod_acc_Z', row_stt=1, col_stt=1)

os.chdir(body_dir)
         
## 2 variables
def plot_comp2(data_1, data_2, title, end_val):
    rw, cl = data_1.shape
    for c in range(cl):

        plt.plot(data_1[:end_val,c], color='k',label = 'Mahony')
        plt.plot(data_2[:end_val,c], color='r', label ='AO v3')
        plt.legend()
        plt.title(title + '_col_num: ' +str(c))
        plt.savefig(title+'_col_num_' +str(c+1)+'_'+str(end_val)+'.png')
        plt.show()

t.tic()
plot_comp2(d_M_x_2d, d_AO_x_3d, 'Comparison X acc', end_val = 35000)
# plot_comp2(d_M_x_2d, d_AO_x_3d, 'Comparison X acc', end_val = 5000)
t.toc()

t.tic()
plot_comp2(d_M_y_2d, d_AO_y_3d, 'Comparison Y acc', end_val = 35000)
# plot_comp2(d_M_y_2d, d_AO_y_3d, 'Comparison Y acc', end_val = 5000)
t.toc()

t.tic()
plot_comp2(d_M_z_2d, d_AO_z_3d, 'Comparison Z acc', end_val = 35000)
# plot_comp2(d_M_z_2d, d_AO_z_3d, 'Comparison Z acc', end_val = 5000)
t.toc()


## 3 variables
def plot_comp3(data_1, data_2, data_3, title, end_val):
    rw, cl = data_1.shape
    for c in range(cl):
        plt.plot(data_1[:end_val,c], color='k',label = 'Mahony(Kp,Ki=0.5,0.0)')
        plt.plot(data_3[:end_val,c], color='r', label = 'AO v3')
        plt.plot(data_2[:end_val,c], color='b', label = 'Mahony(Kp,Ki=1.0,0.3)')
        plt.legend()
        plt.title(title + '_col_num: ' +str(c))
        plt.savefig(title+'_col_num_' +str(c+1)+'_'+str(end_val)+'.png')
        plt.show()

t.tic()
plot_comp3(d_M_x_2d, d_M_x_2d_103, d_AO_x_3d, 'Modified X acc', end_val = 35000)
t.toc()

t.tic()
plot_comp3(d_M_y_2d, d_M_y_2d_103, d_AO_y_3d, 'Modified Y acc', end_val = 35000)
t.toc()

t.tic()
plot_comp3(d_M_z_2d, d_M_z_2d_103, d_AO_z_3d, 'Modified Z acc', end_val = 35000)
t.toc()

# RMSE
def RMSE_matrix(data_1, data_2):
    # both must be in array with same shape
    from sklearn.metrics import mean_squared_error as mse

    RMSE = []
    RMSE_percent = []
    row, col = data_1.shape    
    for c in range(col):
        ind = np.where(data_1[:,c] == 0)
        if len(ind[0]) != 0:
            row_targ = min(ind[0])
        else:
            row_targ = len(data_1)
            
        rmse = np.sqrt(mse(data_1[:row_targ,c], data_2[:row_targ,c]))
        val_range = max(data_1[:row_targ,c]) - min(data_1[:row_targ,c])
        rmse_percent = rmse/val_range
        
        RMSE.append(rmse)
        RMSE_percent.append(rmse_percent)
        
    RMSE = np.vstack(RMSE)
    RMSE_percent = np.vstack(RMSE_percent)
    return RMSE, RMSE_percent
            
RMSE_x, RMSE_x_pec = RMSE_matrix(d_M_x_2d, d_AO_x_3d)
RMSE_y, RMSE_y_pec = RMSE_matrix(d_M_y_2d, d_AO_y_3d)
RMSE_z, RMSE_z_pec = RMSE_matrix(d_M_z_2d, d_AO_z_3d)

# X axis
a1 = RMSE_x.copy()
a2 = RMSE_x_pec.copy()
plt.plot(a1, color='k', label = 'RMSEs')
plt.axhline(y= np.mean(a1), color='r', label = 'mean RMSE: '+ str( round(np.mean(a1),2) ) )
plt.legend()
plt.title('RMSE of X direction / unit: [G]')
plt.savefig('S1_RMSE_Acc_X_mahony_AO.png')
plt.show()

plt.plot(a2, color='k', label = 'RMSEs')
plt.axhline(y= np.mean(a2), color='r', label = 'mean RMSE: '+ str( round(np.mean(a2),2) ) )
plt.legend()
plt.title('Perecentage RMSE of X direction / unit: [%]')
plt.savefig('S1_Per_RMSE_Acc_X_mahony_AO.png')
plt.show()
del a1, a2

# Y axis
a1 = RMSE_y.copy()
a2 = RMSE_y_pec.copy()
plt.plot(a1, color='k', label = 'RMSEs')
plt.axhline(y= np.mean(a1), color='r', label = 'mean RMSE: '+ str( round(np.mean(a1),2) ) )
plt.legend()
plt.title('RMSE of X direction / unit: [G]')
plt.savefig('S1_RMSE_Acc_Y_mahony_AO.png')
plt.show()

plt.plot(a2, color='k', label = 'RMSEs')
plt.axhline(y= np.mean(a2), color='r', label = 'mean RMSE: '+ str( round(np.mean(a2),2) ) )
plt.legend()
plt.title('Perecentage RMSE of X direction / unit: [%]')
plt.savefig('S1_Per_RMSE_Acc_Y_mahony_AO.png')
plt.show()
del a1, a2

# Z axis
a1 = RMSE_z.copy()
a2 = RMSE_z_pec.copy()
plt.plot(a1, color='k', label = 'RMSEs')
plt.axhline(y= np.mean(a1), color='r', label = 'mean RMSE: '+ str( round(np.mean(a1),2) ) )
plt.legend()
plt.title('RMSE of X direction / unit: [G]')
plt.savefig('S1_RMSE_Acc_Z_mahony_AO.png')
plt.show()

plt.plot(a2, color='k', label = 'RMSEs')
plt.axhline(y= np.mean(a2), color='r', label = 'mean RMSE: '+ str( round(np.mean(a2),2) ) )
plt.legend()
plt.title('Perecentage RMSE of X direction / unit: [%]')
plt.savefig('S1_Per_RMSE_Acc_Z_mahony_AO.png')
plt.show()
del a1, a2

# # # TEST: spm paired t-test
# import spm1d
# # ttest = spm1d.stats.ttest_paired(d_M_x_2d.T, d_AO_x_3d.T)
# # ti = ttest.inference(alpha=0.05, two_tailed=False)
# # ti.plot()

# ttest_x = spm1d.stats.ttest_paired(d_M_x_2d.T, d_AO_x_3d.T)
# ti_x = ttest_x.inference(alpha=0.05, two_tailed=False)
# ti_x.plot()

# ttest_y = spm1d.stats.ttest_paired(d_M_y_2d.T, d_AO_y_3d.T)
# ti_y = ttest_y.inference(alpha=0.05, two_tailed=False)
# ti_y.plot()

# ttest_z = spm1d.stats.ttest_paired(d_M_z_2d.T, d_AO_z_3d.T)
# ti_z = ttest_z.inference(alpha=0.05, two_tailed=False)
# ti_z.plot()

print('Commit test')
        
