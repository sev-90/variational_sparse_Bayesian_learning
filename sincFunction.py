
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from math import sqrt
import time

from VI_utils_v4_amb import *
# from VI_dataProcessing_utils import *
from amb_dataProcessing import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics.pairwise import (
    linear_kernel,
    rbf_kernel,
    polynomial_kernel
)

def _apply_kernel( x, y,kernel = 'rbf',coef1 =1 , bias_used=False):
       """Apply the selected kernel function to the data."""
       if kernel == 'linear':
            phi = linear_kernel(x, y)
       elif kernel == 'rbf':
            phi = rbf_kernel(x, y, coef1)
       # elif kernel == 'poly':
       #      phi = polynomial_kernel(x, y, degree, coef1, coef0)
       elif callable(kernel):
              phi = kernel(x, y)
              if len(phi.shape) != 2:
                     raise ValueError(
                    "Custom kernel function did not return 2D matrix"
                )
              if phi.shape[0] != x.shape[0]:
                raise ValueError(
                    "Custom kernel function did not return matrix with rows"
                    " equal to number of data points."""
                )
       else:
              raise ValueError("Kernel selection is invalid.")

       if bias_used:
              phi = np.append(phi, np.ones((phi.shape[0], 1)), axis=1)
       return phi

def genrate_data(x_range, sample_method, num_pnts, fun):
    if sample_method == 'linear':
        x = np.linspace(x_range[0], x_range[1], num_pnts)
        X = x
    elif sample_method == 'uniform':
        # x = np.random.uniform(x_range[0], x_range[1],num_pnts)
        X = []
        epsilon = 1e-8
        for c in range(num_pnts):
            x = np.random.uniform(x_range[0], x_range[1])
            while abs(x) < epsilon: x = np.random.uniform(x_range[0], x_range[1])
            X = np.append(X,x)
        inds= np.where(abs(X) < epsilon, 0,1)
        if len(inds[inds == 0 ]) > 0:
            print(X)
            raise ValueError
    else:
        raise NotImplementedError()
    x= np.sort(X)
    y = fun(x)
    return x, y

if __name__ == '__main__':
    fake = False
    hm_example = False #None#True
    kernel = False
    boston = True
    if boston == True:
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        x = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        y = raw_df.values[1::2, 2]
        kernelType = 'rbf' #'linear'#'rbf'
        z = x
        y = y.reshape(-1,1)
        #### standard scaler ####
        sc_x = StandardScaler()
        z = sc_x.fit_transform(z)
        sc_y = StandardScaler()
        y = sc_y.fit_transform(y)
        # #### Minmax scaler ####
        # sc_y = MinMaxScaler()
        # y = sc_y.fit_transform(y)
        # print('data min ', sc_y.data_min_)
        # print('data max ', sc_y.data_max_)
        # # print("sample variance is ", sc_y.var_)

        
    if kernel==True:
        ''' will do variational relevant vector machine on homework data'''
        s = 1
        # this is an NxD matrix, where N is number of items and D its dimensionalites
        set = 3
        data_path = '/home/sevin/Desktop/Fall_2020/courses/Byesian_ML_6720_John_Paisley/HWs/HW2/EECS6720_data_hw2/'
        x = pd.read_csv(data_path+'X_set{}.csv'.format(set), header = None)
        x = np.array(x)
        y = pd.read_csv(data_path+'y_set{}.csv'.format(set), header = None)
        y = np.array(y)
        z = pd.read_csv(data_path+'z_set{}.csv'.format(set), header = None)
        z = np.array(z)
        # pairwise_sq_dists = squareform(pdist(x, 'sqeuclidean'))
        # x = scipy.exp(-pairwise_sq_dists / s**2)
        t1 = time.time()
        phi = _apply_kernel( x, x,kernel = 'rbf',coef1 =1 , bias_used = False)
        t2 = time.time()

    elif fake == True:
        ''' will do variational relevant vector machine on generated data'''
        z_range = [-10, 10]
        sample_method = 'uniform'
        num_pnts = 100
        # noise_ratio = 0.2
        noise_mean_off = 0
        noise_std = 0.2#0.1 #0#0.1
        def fun(x):
            return np.sin(x)/x #10* np.tanh(x) #10 * np.sinc(x) #5 * np.tanh(x) #10 * np.sinc(x)
        z, y_gt = genrate_data(z_range, sample_method, num_pnts, fun)
        x_plot, y_gt_plot = genrate_data(z_range, sample_method, num_pnts = 500, fun = fun)

        # y_std = np.std(y_gt)
        # y_mean = np.mean(y_gt)
        #  np.random.seed(100)
        y = y_gt + np.random.normal(noise_mean_off, noise_std , y_gt.size) #noise_ratio * y_std
        kernelType = 'rbf'
        # plt.plot(x_plot, y_gt_plot,'-')
        # plt.plot(z, y, '.')
        # plt.show()
        # exit()
    print("data is created")
    # z = x

    rvs = {}
    rmses = {}
    noises = {}
    iteratables = [4] #[0.01, 0.05, 0.1,0.2, 0.4, 0.5, 0.6, 0.8, 1, 2,4, 5] #[0.6, 0.8, 1, 2,4, 5,6, 10]#[10**i for i in range(-5,5)] [0.01, 0.05, 0.1,0.2, 0.4, 0.5, 0.6, 0.8, 1, 2,4, 5]
    iteratables = [1./i for i in iteratables]
    for g in iteratables:
        print('gamma is ', g)
        tmp_rvs = []
        tmp_rmses = []
        tmp_noises = []
        k = 0
        for k in range(1):
            print(g, k)
            if boston:
                np.random.seed(5)
                indices = np.arange(506)  #np.arange(506)
                train_inds = np.random.choice(506, 481, replace = False)
                test_inds = np.delete(indices,train_inds) #np.delete(indices,train_inds)
                X_train, X_test = z[train_inds], z[test_inds]
                y_train, y_test = y[train_inds], y[test_inds]   

            elif fake:
                indices = np.arange(100)  #np.arange(506)
                train_inds = np.random.choice(indices, 50, replace=False) #np.random.choice(506,481,replace=False)
                test_inds = np.delete(indices,train_inds) #np.delete(indices,train_inds)
                X_train, X_test = z[train_inds], z[test_inds]
                y_train, y_test = y[train_inds], y[test_inds]
                X_train = X_train.reshape(-1,1)
                X_test = X_test.reshape(-1,1)
                y_train = y_train.reshape(-1,1)
                y_test = y_test.reshape(-1,1)
                
            else:
                raise NotImplementedError
            
            print(len(X_train), len(y_train), len(X_test), len(y_test))

            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            # plt.plot(x_plot, y_gt_plot,'-')
            # plt.plot(X_train, y_train, '.')
            # plt.show()
            ### instantiate class
            t = 0
            t1 = time.time()
            vi = VI_BLR(x = X_train, x_new = X_train, y = y_train, kernel=True, kernelType = kernelType, gamma_kernel=g)
            vi.initialization()
            numItr = 500
            nd = 0
            delta_L = np.NaN
            while(delta_L > 10e-5 or math.isnan(delta_L)):
                # for t in range(numItr):
                if t%10 == 0:
                    print(t)
                vi.update_q_alpha()
                vi.update_q_lambda()
                vi.update_q_w()
                vi.eval_l()
                if t>1:
                    delta_L = np.abs(vi.Lv[t]-vi.Lv[t-1])
                t += 1
                if t > 500 and math.isnan(delta_L):
                    print("oof")
                    vi.plot_ELBO(t, k, g, save=False)
                    break
                        # if t%100 == 0 and t>0:
                        #     print(t)
                        #     # vi.plot_ELBO(t+1)
                        #     vi._prune_while_training(prune_w_weight = True, weight_treshold = 0.0001)
                        #     # vi._prune_while_training(prune_w_alpha = True, alpha_treshold=10e8)
                        #     vi._add_newData(add_newData[nd]) 
                        #     nd += 1  
            vi.plot_ELBO(t, k, g, degree=None, save=False)
            # weight_treshold = 0.01#1e-3
            # alpha_treshold = 10e5
            # # vi.stemPlot_relevance(prune_w_weight=True, weight_treshold=weight_treshold)
            # vi._prune_while_training(prune_w_weight=True, weight_treshold=weight_treshold)
            # # vi.stemPlot_relevance(prune_w_alpha=True, alpha_treshold=alpha_treshold)
            # # vi._prune_while_training(prune_w_alpha=True, alpha_treshold=alpha_treshold)
            # rvs.append(len(vi.mu_p))
            tmp_noises.append(vi.f_p/vi.e_p)
            # print(vi.a_p/vi.b_p)
            alpha = pd.DataFrame(list(vi.a_p/vi.b_p), columns=['alpha'])
            mu = pd.DataFrame(list(vi.mu_p),columns=['mu'])
            df = pd.concat([alpha,mu],axis=1)
            print(df)
            y_predict_tr, var_predict_tr,y_predictn_tr, var_predictn_tr = vi.posterior_predictive(X_train)
            y_predict_tst, var_predict_tst,y_predictn_tst, var_predictn_tst = vi.posterior_predictive(X_test)

            y_predict_ = pd.DataFrame(list(y_predict_tst),columns = ['y_predict'])
            var_predict_ = pd.DataFrame(list(var_predict_tst),columns = ['var_predict'])
            y_predictn_ = pd.DataFrame(list(y_predictn_tst),columns = ['y_predictn'])
            var_predictn_ = pd.DataFrame(list(var_predictn_tst),columns = ['var_predictn'])
            y_true = pd.DataFrame(list(y_test), columns=['y_real_tst'])
            df = pd.concat([y_true, y_predict_, var_predict_, y_predictn_, var_predictn_], axis=1)
            print(df)
            rms_tr_sc = sqrt(mean_squared_error(y_train, list(y_predict_tr.squeeze())))
            rms_tst_sc = sqrt(mean_squared_error(y_test, list(y_predict_tst.squeeze())))
            ######### transfer back ####
            y_tr_rlScale = sc_y.inverse_transform(y_train)
            y_predict_tr_rlScale = sc_y.inverse_transform(y_predict_tr)
            y_tst_rlScale = sc_y.inverse_transform(y_test)
            y_predict_tst_rlScale = sc_y.inverse_transform(y_predict_tst)
            rms_tr_rlsc = sqrt(mean_squared_error(y_tr_rlScale, list(y_predict_tr_rlScale.squeeze())))
            rms_tst_rlsc = sqrt(mean_squared_error(y_tst_rlScale, list(y_predict_tst_rlScale.squeeze())))
            ##############################
            weight_treshold = 0.001 #1e-3
            vi._prune_while_training(prune_w_weight=True, weight_treshold = weight_treshold)
            tmp_rvs.append(len(vi.mu_p))
            rvs_cnt = len(vi.mu_p)
            print('gamma ', g)
            # plot_predictions(y_test_= y_train, y_pred_= y_predict, g = g, k = k,  error = None, n=len(y_train), save=False)
            
            plot_predictions_new(y_tr=y_tr_rlScale, y_pred_tr=y_predict_tr_rlScale,y_test = y_tst_rlScale, y_pred_tst= y_predict_tst_rlScale,g = g, k = k,  error = None, rms_tr=rms_tr_rlsc, rms_tst=rms_tst_rlsc, rvs = rvs_cnt, save=False)
            # plot_data(x_plot, y_gt_plot, X_train, y_train, X_test, y_test, vi.Xpruned, vi.ypruned)
            print('error train ',rms_tr_rlsc)
            print('error test ',rms_tst_rlsc)
            print('noise estimate ', vi.f_p/vi.e_p)
            tmp_rmses.append(rms_tst_rlsc)
            print('relevance vectors ',len(vi.mu_p))
            # alpha_treshold = 10e5
            # vi.stemPlot_relevance(prune_w_weight=True, weight_treshold=weight_treshold)
     
        rmses.update({g:np.mean(tmp_rmses)})
        noises.update({g:np.mean(tmp_noises)})
        rvs.update({g:np.mean(tmp_rvs)})
    print('errors ')
    print(rmses)
    print('relevance vectors ')
    print(rvs)
    print('noises ', noises)
    print(noises)




