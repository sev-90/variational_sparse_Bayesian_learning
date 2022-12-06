import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import geopandas as gpd
import math
import pickle
import re
from datetime import datetime
from VRVM_utils import *


class VRVM(object):
    def __init__(self, data_type = None, prune = False):
        self.data_type = data_type
        self.prune = prune
        self.VIs_file_name = '/home/sevin/Desktop/projects/TravelTime_Prediction/VRVM_clean/trained_models/{}/'.format(self.data_type) + 'VIs.pickle'
        print('****** loading data...')
        # self.split_data(how = 'atom_hr_batches')
        # # self.load_data()
        # # self.ensemble_predictions()

        for self.set in range(1):
            print('****** set ', self.set)
            self.load_data()
            self.prepare_model_input()
            self.call_vrvm()
            self.save_model()
            self.predict()
            
    def split_data(self, how = None): # 'atom_hr_batches
        data_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/VRVM_clean/data/modeling_data/{}/'.format(self.data_type)
        self.data = pd.read_csv(data_path + 'train.txt')
        b = 20
        train_dfBatch_list = [pd.DataFrame([])] * b
        # print(train_dfBatch_list[0])
        # exit()
        # counts = pd.DataFrame(data[['From','To']].value_counts(), columns=['counts'])
        data_df = pd.DataFrame(self.data).set_index(['Oatom','Datom'])
        counts_dic = dict(self.data[['Oatom','Datom']].value_counts())
        list_OD = list(counts_dic.keys())
        for od in list_OD:
            print(od)
            coun = counts_dic[od[0],od[1]]
            if coun>0:
                masked_df = data_df.loc[od[0],od[1]]
                for hr in range(24):
                    masked_df_df = masked_df[masked_df['hour']==hr].reset_index(drop=True)
                    if len(masked_df_df) > 0 :
                            
                        n = len(masked_df_df)
                        m = n // b
                        for ii in range(b):
                            train_dfBatch_list[ii] = train_dfBatch_list[ii].append(masked_df_df[ii*m:(ii+1)*m])
                        if n % b != 0:
                            alpha = n - m * b
                            rand_ind = random.randint(0,b-1)
                            train_dfBatch_list[rand_ind] = train_dfBatch_list[rand_ind].append(masked_df_df[-alpha:])
                    else:
                        continue
            else:
                continue
        for jj in range(len(train_dfBatch_list)):
            train_dfBatch_list[jj].to_csv(data_path + 'train_batch_{}.txt'.format(jj), index = False)
          

    def load_data(self):
        data_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/VRVM_clean/data/modeling_data/{}/'.format(self.data_type)
        # self.train_df = pd.read_csv(data_path + 'train_batch_{}.txt'.format(self.set)) #'train_wpreds.txt'  train_wpreds_1
        # self.train_df_sample = self.train_df.sample(n=5000) #random_state = 2342
        self.train_df_sample = pd.read_csv(data_path + 'train_batch_{}.txt'.format(self.set)).reset_index(drop=True)
        # print(self.train_df_sample)

        self.x_train = np.array([])
        # self.x_train = self.train_df[['lat_origin', 'long_origin', 'lat_destination', 'long_destination',  'hour', 'minute']].values
        # self.y_train = self.train_df['travel_time'].values 
        # self.y_train  = np.array(self.y_train).reshape(-1,1)
        self.test_df= pd.read_csv(data_path + 'train_batch_{}.txt'.format(self.set+1))
        # self.test_df= pd.read_csv(data_path + 'test_wpreds.txt')
        # print(self.test_df.head())
        self.x_test = np.array([])
        # self.x_test = self.test_df[['lat_origin', 'long_origin', 'lat_destination', 'long_destination',  'hour', 'minute']].values
        # self.y_test = self.test_df['travel_time'].values

        with open(data_path + 'scale_parameters.pickle', 'rb') as handle:
            scale_parameters = pickle.load(handle)
        self.variables_means = scale_parameters['means']
        self.variables_stds = scale_parameters['stds']
        self.mean_y = self.variables_means['travel_time']
        self.sigma_y = self.variables_stds['travel_time'] 
        # print(self.variables_means)
        # print(self.variables_stds)
        # exit()
    
    def prepare_model_input(self): 

        x_variables = ['highway_miles', 'street_miles', 'path_avg_degree', 'path_avg_bwness', 'lat_origin', 'long_origin',
                        'lat_destination', 'long_destination','sin_theta','cos_theta' ] #, 'adverse_weather'  'x_origin', 'y_origin', 'x_destination','y_destination',
        y_variable = ['travel_time'] # 'shortPath_mile',
        print('******* scaling inputs...')
        
        def scale_variable(variable, x):
            scaled_x = (x[variable] - self.variables_means[variable])/self.variables_stds[variable]
            return scaled_x
        
        for var in x_variables:
            self.x_train = np.append(self.x_train, scale_variable(var, self.train_df_sample))
        self.x_train = self.x_train.reshape(-1,len(x_variables))
        self.y_train = np.array(scale_variable(y_variable[0], self.train_df_sample)).reshape(-1,len(y_variable))

        for var in x_variables:
            self.x_test = np.append(self.x_test, scale_variable(var, self.test_df))
        self.x_test = self.x_test.reshape(-1,len(x_variables))
        self.y_test = np.array(scale_variable(y_variable[0], self.test_df)).reshape(-1,len(y_variable))

        
    
    def call_vrvm(self):
        t1 = time.time()
        # phi = rbf_kernel(self.x_train, self.x_train, 1)
        # dis = np.sqrt(-1 * np.log(phi))
        # g = np.quantile(dis,0.01)/1000
        # print(g)
        self.vi = VI_BLR(x = self.x_train, x_new = self.x_train,  y = self.y_train, kernel = 'rbf', multi_kernel = False, g = 0.15) #0.25
        print("***** kernel construction is done, {} sec****".format(time.time()-t1))
        t1 = time.time()
        self.vi.initialization()
        print("***** initialiation is done, {} sec******".format(time.time()-t1))

        t1 = time.time()
        self.vi.train_model(prune_threshold=None, prune = self.prune)
        print("***** training is done, {} sec ******".format(time.time()-t1))


    def save_model(self):
        self.VIs_file_name = '/home/sevin/Desktop/projects/TravelTime_Prediction/VRVM_clean/trained_models/{}/'.format(self.data_type) + 'VIs_set{}.pickle'.format(self.set)
        with open(self.VIs_file_name, 'wb' ) as file: #'ab'
            pickle.dump(self.vi, file)
    
    
    def predict(self, elbow=False):
        # self.set += 1
        vis = []
        with open(self.VIs_file_name, 'rb') as fr:
            try:
                while True:
                    vis.append(pickle.load(fr))
            except EOFError:
                pass 
        vi = vis[0]
        if elbow:
            elbo = vi.Lv
            numItr = len(elbo)
            x = np.arange(1,numItr+1,1).tolist()
                # print(self.Lv)
            plt.plot(x,elbo)
            ax = plt.axes()
            ax.yaxis.grid(True)
            plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
            plt.xlabel('Iteration')
            plt.ylabel('ELBO')
            plt.show()
        print(self.x_train)
        y_predict_tr, var_predict_tr = vi.posterior_predictive(self.x_train, prune=self.prune)
        y_predict_tst, var_predict_tst = vi.posterior_predictive(self.x_test, prune = self.prune)

        self.train_df_sample['predMean_VI_{}'.format(self.set)] = y_predict_tr * self.sigma_y + self.mean_y
        self.train_df_sample['predStd_VI_{}'.format(self.set)] = var_predict_tr * self.sigma_y**2

        self.test_df['predMean_VI_{}'.format(self.set)] = y_predict_tst * self.sigma_y + self.mean_y
        self.test_df['predStd_VI_{}'.format(self.set)] = var_predict_tst * self.sigma_y**2

        data_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/VRVM_clean/data/modeling_data/{}/'.format(self.data_type)
        self.train_df_sample.to_csv(data_path + 'train_wpreds_{}.txt'.format(self.set), index = False)
        self.test_df.to_csv(data_path + 'test_wpreds.txt', index = False)
        # print(self.test_df)

    def load_preds_data(self):
        pass

    def ensemble_predictions(self):
        data_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/VRVM_clean/data/modeling_data/{}/'.format(self.data_type)
        self.test_df= pd.read_csv(data_path + 'test_wpreds.txt')
        self.test_df['y_pred_ensemble'] = self.test_df.filter(like='predMean_VI').mean(axis=1)
        var_pred_cols = self.test_df.columns[self.test_df.columns.str.contains('predStd_VI', case=False)]
        ypred_cols = self.test_df.columns[self.test_df.columns.str.contains('predMean_VI', case=False)]
        # var_predt_cols = [col for col in var_predt_cols if re.split(r'(\d+)', col)[0]=='predStd_VI']
        self.test_df['var_pred_ensemble'] = self.test_df.apply(lambda x:np.mean(x[var_pred_cols]) + np.mean(x[ypred_cols]**2) -
                                             x['y_pred_ensemble']**2, axis=1)
        print(self.test_df[['travel_time','y_pred_ensemble']])

        fig, ax = plt.subplots(figsize=(6,6))
        ax.errorbar(x = self.test_df['travel_time']/60., y = self.test_df['y_pred_ensemble']/60., yerr= self.test_df['var_pred_ensemble'].map(np.sqrt)/60., fmt='r.', alpha=0.4)
        ax.plot([0,60],[0,60])
        plt.show()