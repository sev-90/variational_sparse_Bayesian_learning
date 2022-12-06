import numpy as np
import pandas as pd
from train_vrvm import *


if __name__ == '__main__':
    vi = VRVM(data_type='amb') ## final train vi model with relavent vectors (rvs)

    data_path = '/home/sevin/Desktop/projects/TravelTime_Prediction/VRVM_clean/data/modeling_data/{}/'.format('amb')

    # data_tst = pd.read_csv(data_path + 'test_wpreds.txt')
    data_tst = pd.read_csv(data_path + 'test_wpreds.txt')
    for set in range(1):
        data_tr = pd.read_csv(data_path + 'train_wpreds_{}.txt'.format(set))
        print(data_tr)

        fig, ax = plt.subplots(ncols = 2, figsize=(10,6))
        ax[0].errorbar(x = data_tr['travel_time']/60., y = data_tr['predMean_VI_{}'.format(set)]/60., yerr= data_tr['predStd_VI_{}'.format(set)].map(np.sqrt)/60., fmt='r.', alpha=0.4)
        ax[0].plot([0,60],[0,60])
        ax[1].errorbar(x = data_tst['travel_time']/60., y = data_tst['predMean_VI_{}'.format(set)]/60., yerr= data_tst['predStd_VI_{}'.format(set)].map(np.sqrt)/60., fmt='r.', alpha=0.4)
        ax[1].plot([0,60],[0,60])
        plt.show()


