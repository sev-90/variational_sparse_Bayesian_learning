from itertools import filterfalse
import numpy as np
import pickle
import pandas as pd
from VI_utils_v4_amb import *
from amb_dataProcessing import *
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from math import sqrt
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
import re



def plot_data_info(n_dataBatch, reg, TT_distr = None,kde=False, hr_distr = None, OD_distr = None, testSetID=None, atoms=None ,plot_selected_atoms=False):
    parent_folder =  '/home/sevin/Desktop/projects/TravelTime_Prediction/'
    project_name = 'V_RVM_amb'
    data = pd.read_csv(parent_folder + project_name + '/data/processed_data_SM/{}_{}/{}_weekdays_seg2-3_batch_2018-2019_total.txt'.format(reg,n_dataBatch, reg))
    data = clean_data(data)  
    data = data[data['To'].apply(lambda x: x not in ['WTLIB','WTELI','GOVIA'])]
    count_D = pd.DataFrame(data[['To']].value_counts(), columns=['count_D']).reset_index(drop=False)
    count_O = pd.DataFrame(data[['From']].value_counts(), columns=['count_O']).reset_index(drop=False)
    count = count_D.merge(count_O, left_on='To', right_on='From', how='outer')
    data_test = pd.read_csv(parent_folder + project_name + '/data/processed_data_SM/{}_{}/{}_weekdays_seg2-3_batch_{}.txt'.format(reg,n_dataBatch, reg,testSetID))
    del data['Unnamed: 0']
    del data['Unnamed: 0.1']
    print(data)
    x = data[['lat_origin','long_origin', 'lat_destination', 'long_destination' ]].values
    x_test = data_test[['lat_origin','long_origin', 'lat_destination', 'long_destination' ]].values

    ############################# plot travel time histograms #######################################
    if TT_distr:
        fig, ax = plt.subplots()
        if kde:
            import seaborn as sns
            sns.distplot(data['travel_time'], ax=ax, color='black', hist=False) #[data['travel_time']<=2500]
            sns.distplot(data_test['travel_time'],ax=ax, color='red', hist=False) #[data_test['travel_time']<=2500]
        else:
            ax.hist(data['travel_time'],bins=30, density=True, color='black',histtype='step') #slategray
            ax.hist(data_test['travel_time'],bins=30, density=True, color='red',histtype='step') #lightcoral
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.set_xlabel('Travel Time (s)',fontsize=14) #,fontweight="bold"
        plt.legend(['train', 'test'])
        plt.xticks(fontsize=14) #weight = 'bold'
        # plt.ylabel('Count')
        plt.show()
    ########################## plot how much data we have every hour(time distribution) ###############
    if hr_distr:
        fig, ax = plt.subplots()
        data['hour'].value_counts().sort_index().plot(kind='bar', color='slategray', ax=ax)
        data_test['hour'].value_counts().sort_index().plot(kind='bar', color='lightcoral', ax=ax)
        plt.legend(['train', 'test'])
        plt.xlabel('Hour')
        plt.ylabel('Count')
        plt.show()
    ######################## plo origin destination distribution #########################
    if OD_distr:
        plot_regions( X_train=x, X_test=x_test, atoms=atoms, region='Manhattan Central', count = count, x_rvs=None, ncol=2,plot_selected_atoms=plot_selected_atoms) #['013BA','013DA'] #atoms
    # exit()
#######################################################


def hist_error(errors, errors_tr=None, per_test=None, per_train=None, modelid=None, ax=None ,color=None, ensemble=False):
    # error = y_pred_ - y_test_
    ax.hist(errors, bins=50, density=False, color=color[1])
    if errors_tr is not None:
        ax.hist(errors_tr, bins=50,density=True, color=color[0], alpha=0.5)
        ax.legend(['Test {:.1f}'.format(per_test[0]),'Train {:.1f}'.format(per_train[0])], loc='upper left',fontsize=8)
    elif ensemble:
        ax.legend(['RMSE {:.1f}'.format(per_test[0])],loc='upper left',fontsize=8)
    else:
        ax.legend(['Test {:.1f}'.format(per_test[0])],loc='upper left',fontsize=8)
    ax.set_title(modelid, fontsize=7)#('Model {}'.format(modelid))
    

def scatter_plot_predictions(y_real=None, y_pred=None, y_train=None, y_pred_tr=None,errors=None,
                                 per_test=None, per_train=None, ax=None, modelid=None, n=100, color=None, ensemble=False, y_total=None):
    
    if y_train is None:
        ax.errorbar(y_real[:n], y_pred[:n],xerr = errors[:n], yerr=errors[:n], fmt='.', color=color[1], alpha=0.5, markersize=2) 
    else:
        ax.errorbar(y_real[:n], y_pred[:n],xerr =errors[1][:n],   yerr=errors[1][:n], fmt='.', color=color[1], alpha=0.5, markersize=2) #xerr=errors[1][:n],
    if y_train is not None:
        ax.errorbar(y_train[:n], y_pred_tr[:n],xerr =errors[0][:n],  yerr=errors[0][:n], fmt='.', color=color[0], alpha=0.5, markersize=2)
        mi = min(np.amin(y_real[:n]), np.amin(y_pred[:n]),np.amin(y_train[:n]), np.amin(y_pred_tr[:n]))#0
        ma = max(np.amax(y_real[:n]), np.amax(y_pred[:n]), np.amax(y_train[:n]), np.amax(y_pred_tr[:n]))#0
    if y_pred_tr is not None:
        ax.legend(['Test {:.2f}'.format(per_test[1]),'Train {:.2f}'.format(per_train[1])], loc='upper left', fontsize=8)
    elif ensemble:
        ax.text(0,1410,'MAPE: {:.2f}\nRMSE: {:.2f}'.format(per_test[1],per_test[0]),fontsize=12) #loc='upper left'
    else:
        ax.legend(['Test {:.2f}'.format(per_test[1])], loc='upper left',fontsize=8)

    if y_total is not None:
        mi = min(np.amin(y_total), np.amin(y_total))#0
        ma = max(np.amax(y_total), np.amax(y_total))#0
        lims = [mi,ma]
    else:
        mi = min(np.amin(y_real[:n]), np.amin(y_pred[:n]))#0
        ma = max(np.amax(y_real[:n]), np.amax(y_pred[:n]))#0
        lims = [mi,ma]

    ax.plot([0.5 * lims[0], 1.05 * lims[1]], [0.5 * lims[0], 1.05 * lims[1]])
    ax.grid(True)
    ax.set_aspect('equal')

def plot_relevances(keep_relevances, y_real, y_pred, n=None, ax=None): #relevances, y_train, y_pred_tr, n=len(y_train), ax=ax
    ax.scatter(y_real, y_pred, marker='.', color='blue', s=8) #, color=color
    y_real_ = y_real[keep_relevances[1:len(y_real)+1]]
    y_pred_ = y_pred[keep_relevances[1:len(y_pred)+1]]
    ax.scatter(y_real_, y_pred_, marker='o', facecolor='none', edgecolor='r', s=25, linewidth=1.5)
    mi = min(np.amin(y_real), np.amin(y_pred))#0
    ma = max(np.amax(y_real), np.amax(y_pred))#0
    lims = [mi,ma]
    ax.grid(True)
    ax.set_aspect('equal')
    # ax.set_title('Kernel 1',fontsize=12)
    ax.set_xlabel('True Values(s)',fontsize=12)
    ax.set_ylabel('Predictions(s)',fontsize=12)
    
    y_real_ = y_real[keep_relevances[len(y_real)+1:-1]]
    y_pred_ = y_pred[keep_relevances[len(y_real)+1:-1]]
    ax.scatter(y_real_, y_pred_, marker='o', facecolor='none', edgecolor='g', s=25,linewidth=1.5)
    ax.legend(['Training data','Relevance vectors1','Relevance vectors2'], loc='upper left', fontsize=10)
    ax.plot([0.5 * lims[0], 1.05 * lims[1]], [0.5 * lims[0], 1.05 * lims[1]])

    
def errorbar_plot_predictions(y_real,y_pred, errors, n=100, ax=None, title=None):

    ax.scatter(y_real, y_pred, marker='.') #, color=color
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    
    mi = min(np.amin(y_real), np.amin(y_pred))#0
    ma = max(np.amax(y_real), np.amax(y_pred))#0
    lims = [mi,ma]
    ax.plot([0.5 * lims[0], 1.05 * lims[1]], [0.5 * lims[0], 1.05 * lims[1]])
    ax.grid(True)
    ax.set_title(title)
    ax.set_aspect('equal')


def calculate_1std_error(var_tr, var_tst):
    errors_tr = [ math.sqrt(var_tr[i]) for i in range(len(var_tr))]
    errors_tst = [ math.sqrt(var_tst[i]) for i in range(len(var_tst))]
    return errors_tr, errors_tst

def ensemble_models(modelsIDs, testSetID,model_name,quant,gama, width_parameters, scaling_factors):

    sigma = scaling_factors['std_target']
    mean = scaling_factors['mean_target']
    fun1 = lambda x: x*sigma + mean
    fun2 = lambda x: x*sigma**2
    predictions_onTest = pd.DataFrame()
    predictions_onTrain = pd.DataFrame()
    count = 0  
    for i in range(len(modelsIDs)):
        modelID = modelsIDs[i]
        if width_parameters:
            quant = width_parameters[modelID][0]
            gama = width_parameters[modelID][1]
        else:
            quant=None
            gama=gama

        pred_onTest = pd.read_csv('./code/{}/{}/predictions_onTest_model{}_test{}_g{}_q{}.txt'.format(region, model_name, modelID, testSetID,gama,quant))
        pred_onTrain = pd.read_csv('./code/{}/{}/predictions_onTrain_model{}_g{}_q{}.txt'.format(region, model_name, modelID,gama,quant))
        del pred_onTest['Unnamed: 0']
        del pred_onTrain['Unnamed: 0']
        
        if scaling_factors is not None:
            pred_onTest['y_real_tst'] = fun1(pred_onTest['y_real_tst'])
            pred_onTest['y_predict'] = fun1(pred_onTest['y_predict'])
            pred_onTest['y_predictn'] = fun1(pred_onTest['y_predictn'])
            pred_onTest['var_predict'] = fun2(pred_onTest['var_predict'])
            pred_onTest['var_predictn'] = fun2(pred_onTest['var_predictn'])
            pred_onTrain['y_real_tr'] = fun1(pred_onTrain['y_real_tr'])
            pred_onTrain['y_predict'] = fun1(pred_onTrain['y_predict'])
            pred_onTrain['y_predictn'] = fun1(pred_onTrain['y_predictn'])
            pred_onTrain['var_predict'] = fun2(pred_onTrain['var_predict'])
            pred_onTrain['var_predictn'] = fun2(pred_onTrain['var_predictn'])
        pred_onTrain = pred_onTrain.rename(columns={'y_real_tst':'y_real_tr'})
        pred_onTest['model'] = 'Model_{}'.format(modelID)
        pred_onTrain['model'] = 'Model_{}'.format(modelID)
        count += len(pred_onTest)
        predictions_onTest = pred_onTest.join(predictions_onTest, lsuffix=modelID)
        predictions_onTrain = predictions_onTrain.append(pred_onTrain)

    ypredt_cols = predictions_onTest.columns[predictions_onTest.columns.str.contains('y_predict',
                                        case=False)]
    ypredt_cols = [col for col in ypredt_cols if re.split(r'(\d+)', col)[0]=='y_predict']
    predictions_onTest['y_predt_ensemble'] = predictions_onTest[ypredt_cols].mean(axis=1) #predictions.iloc[:,1:88:6].mean(axis=1)

    predictions_onTest['y_predn_ensemble'] =predictions_onTest.filter(like='y_predictn').mean(axis=1) #predictions.iloc[:,3:88:6].mean(axis=1)
    var_predt_cols = predictions_onTest.columns[predictions_onTest.columns.str.contains('var_predict',
                                        case=False)]
    var_predt_cols = [col for col in var_predt_cols if re.split(r'(\d+)', col)[0]=='var_predict']
    predictions_onTest['var_predt_ensemble'] = predictions_onTest.apply(lambda x:np.mean(x[var_predt_cols]) + np.mean(x[ypredt_cols]**2) - x['y_predt_ensemble']**2, axis=1)
    predictions_onTest['var_predn_ensemble'] = predictions_onTest.apply(lambda x:np.mean(x.filter(like='var_predictn')) + np.mean(x.filter(like='y_predictn')**2) - x['y_predn_ensemble']**2, axis=1)
    
    l = modelsIDs[0]
    predictions_onTest = predictions_onTest.rename(columns={'y_predict':'y_predict{}'.format(l), 'var_predict':'var_predict{}'.format(l),'y_predictn':'y_predictn{}'.format(l),'var_predictn':'var_predictn{}'.format(l)})

    predictions_onTrain.set_index('model', inplace=True)
    return predictions_onTest, predictions_onTrain

def plot_predictions_per_hour( predictions,modelID, testSetID, region, gama=None,quant=None,aggregated=False,scaling_factors=None): #modelID=0, testSetID = ['017AA',  '014CB'], region=region, gama=gamma[0],quant=quant

    atoms = testSetID
    parent_folder =  '/home/sevin/Desktop/projects/TravelTime_Prediction/'
    project_name = 'V_RVM_amb'
    if aggregated:
        data = pd.read_csv(parent_folder + project_name + '/data/processed_data_SM/{}_{}/{}_weekdays_seg2-3_batch_2018-2019_aggregated.txt'.format(region,'aggregated',region))
    else:
        data = pd.read_csv(parent_folder + project_name + '/data/processed_data_SM/{}_weekdays_seg2-3_batch_2018-2019_total.txt'.format(region))
    mask = (data['From']==atoms[0]) & (data['To']==atoms[1]) #'001AC') & (data['To']=='005AB'
    atm_lvl_test = data[mask].reset_index(drop=True)
    #####
    atm_lvl_test = clean_data(atm_lvl_test)
    ####

    print(atm_lvl_test)
    print(len(atm_lvl_test),len(predictions))
    assert len(atm_lvl_test)==len(predictions), "data is not same length"
    data_concat = pd.concat([atm_lvl_test,predictions], axis=1)
    
    data_concat['total_min'] = data_concat['hour'] * 60 + data_concat['minute']
    data_concat = data_concat.sort_values('total_min')
    try:
        print("predictions based on ensemble model")
        data_concat['std_pred'] = data_concat['var_predn_ensemble'].apply(lambda x: np.sqrt(x))
    except:
        data_concat['std_pred'] = data_concat['var_predictn'].apply(lambda x: np.sqrt(x))
    
    
    dict_resampled = {}
    dict_hrlySamples ={}
    for i in range(24):
        # mask = df['hour'] == i
        mask = data_concat['hour'] == i 
        d = data_concat[mask].reset_index(drop=True)
        n = len(d)
        
        if n > 0 :
            if aggregated:
                assert n==1
            avg_TT = d['y_real_tst'].mean()
            sample_std = d['y_real_tst'].std(ddof=1) #if sample std set ddof=1
            sample = d['y_real_tst'].values
            sample = sample[~np.isnan(sample)]

            try:
                avg_pred_TT = d['y_predn_ensemble'].mean()
                # var_pred = d['var_predn_ensemble'].mean() + np.mean(d['y_predn_ensemble']**2) - avg_pred_TT**2
                var_pred = d['var_predn_ensemble'].sum()/n**2
            except:

                avg_pred_TT = d['y_predictn'].mean()
                # var_pred = d['var_predn_ensemble'].mean() + np.mean(d['y_predn_ensemble']**2) - avg_pred_TT**2
                var_pred = d['var_predictn'].sum()/n**2
            dict_resampled.update({i:{'hour':i,'cnt':int(n),'sample_mean':avg_TT,'sample_std':sample_std, 'avg_pred_TT':avg_pred_TT, 'var_pred':var_pred, 'std_pred':np.sqrt(var_pred)}})
            dict_hrlySamples.update({i:sample})
        else:
            continue
    df_resampled = pd.DataFrame(dict_resampled).T
    
    fig, ax = plt.subplots(figsize=(20,5))
    markersize = np.array([ (df_resampled['cnt'].iloc[i])*10 for i in  range(len(df_resampled))])
    # c = df_resampled['cnt'].values
    pl2 = ax.scatter( df_resampled['hour'].values, df_resampled['sample_mean'].values, s=markersize,  marker='o',c='blue', edgecolor = 'none') #,cmap=cm.viridis c=c,
    pl3 = ax.errorbar(df_resampled['hour'].values, df_resampled['avg_pred_TT'], df_resampled['std_pred'],  fmt='D', markersize=6, color='r', alpha=0.8,zorder=0) #, alpha=0.5 capsize = 2,mfc='none',
    ticks = df_resampled['hour'].values
    plt.xticks(ticks, df_resampled['hour'].values.astype(int), rotation = 30, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Time of Day',fontsize=14)
    plt.ylabel('Travel Time(s)',fontsize=14)
    plt.legend([pl2,pl3],['Sample mean', 'prediction mean +/- prediction std'],fontsize=12) #+/-sample std #pl1["boxes"][0], 'Observed sample',
    plt.title('From atom {} to atom {}'.format(atoms[0],atoms[1]),fontsize=14)
    plt.show()

def plot_longShort_ensemble(predictions_train, predictions, modelsIDs, ncols=3, log=False ):
    model_noises = get_noises(region,model_name, num_ofModels, quant, gama, width_parameters, plot=False)
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(15, 5),sharex=True, sharey=True) #, sharex=True, sharey=True
    plt.setp(ax[:], xlabel='y true(s)')
    plt.setp(ax[0], ylabel='y prediction(s)')
    ensemble_noise = np.mean(list(model_noises.values()))
    y_true = predictions.y_real_tst
    y_pred = predictions.y_predn_ensemble
    var_pred = np.sqrt(predictions.var_predn_ensemble - ensemble_noise)

    mask_short = np.where(y_true<240)
    mask_medium = np.where((y_true>=240) & (y_true<=720))
    mask_long = np.where(y_true>=720)
    
    y_true_short = y_true[mask_short[0]]
    y_pred_short = y_pred[mask_short[0]]
    var_pred_short = var_pred[mask_short[0]]
    y_true_medium = y_true[mask_medium[0]]
    y_pred_medium = y_pred[mask_medium[0]]
    var_pred_medium = var_pred[mask_medium[0]]
    y_true_long = y_true[mask_long[0]]
    y_pred_long = y_pred[mask_long[0]]
    var_pred_long = var_pred[mask_long[0]]
    y_trues = [y_true_short,y_true_medium, y_true_long] #,y_true
    y_preds = [y_pred_short, y_pred_medium, y_pred_long] #,y_pred
    var_preds = [var_pred_short,var_pred_medium,  var_pred_long] #,var_pred
    trips = ['short','medium', 'long'] #,'total'
    for i, obj in enumerate(zip(trips,y_trues, y_preds, var_preds)):
        trip = obj[0]
        y_true_ = obj[1]
        y_pred_ = obj[2]
        var_pred_ = obj[3]
        rmse = sqrt(mean_squared_error(y_true_, y_pred_))
        mape = mean_absolute_percentage_error(y_true_, y_pred_)
        # errors = y_pred - y_true
        per_test = [ rmse, mape ]
        scatter_plot_predictions(y_true_, y_pred_, errors=var_pred_,  per_test = per_test, ax=ax[i], modelid=trip, n=len(y_true),
                                 color=['blue','red'], ensemble=True, y_total=y_true)
        # hist_error(errors=errors, per_test = per_test, modelid=None, ax=ax[1] ,color=['blue','red'], ensemble=True)
    plt.show()
def plot_ensemble_model(predictions_train, predictions, modelsIDs, ncols=4, log=False ):
    model_noises = get_noises(region,model_name, num_ofModels, quant, gama, width_parameters, plot=False)
    # print(predictions_train)
    # print(predictions)
    # exit()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5)) #, sharex=True, sharey=True
    # fig.subplots_adjust(hspace=0.5)
    # fig.subplots_adjust(wspace=0.5)
    # fig.tight_layout()
    plt.setp(ax[0], xlabel='y true(s)')
    plt.setp(ax[0], ylabel='y prediction(s)')
    plt.setp(ax[1], xlabel='prediction error(s)')
    plt.setp(ax[1], ylabel='count')
    rmses = []
    mapes = []
    for i in range(len(modelsIDs)): #ax.reshape(1,-1)[0][:-1]
        modelID = modelsIDs[i]
        print(modelID)
        mod = 'Model_{}'.format(modelID)
        y_train = predictions_train.loc[mod,'y_real_tr'].values
        y_pred_tr = predictions_train.loc[mod,'y_predictn'].values
        var_pred_tr = np.sqrt( predictions_train.loc[mod,'var_predictn'].values - model_noises[modelID]) 
        print(len(y_train), len(var_pred_tr))
        if log:
            y_train = np.exp(y_train)
            y_pred_tr = np.exp(y_pred_tr)
        # modelID = modelsIDs[i]
        y_true = predictions.y_real_tst
        y_pred = predictions['y_predictn{}'.format(modelID)]
        var_pred = np.sqrt(predictions['var_predictn{}'.format(modelID)] - model_noises[modelID] )
        print(len(y_true), len(var_pred))
        if log:
            y_true = np.exp(y_true)
            y_pred = np.exp(y_pred)
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)
        rmses.append(rmse)
        mapes.append(mape)
    ensemble_noise = np.mean(list(model_noises.values()))
    y_pred = predictions.y_predn_ensemble
    var_pred = np.sqrt(predictions.var_predn_ensemble - ensemble_noise)
    if log:
        y_pred = np.exp(y_pred)
    std = np.sqrt(predictions.var_predn_ensemble)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    errors = y_pred - y_true
    per_test = [rmse, mape ]
    scatter_plot_predictions(y_true, y_pred, errors=var_pred,  per_test = per_test, ax=ax[0], modelid=None, n=len(y_true), color=['blue','red'], ensemble=True)
    hist_error(errors=errors, per_test = per_test, modelid=None, ax=ax[1] , color=['blue','red'], ensemble=True)
    plt.show()
    return rmse, mape

def scatterPlots_predictions(predictions_train, predictions, modelsIDs, ncols=4, log=False ):
    model_noises = get_noises(region,model_name, num_ofModels, quant, gama, width_parameters, plot=False)
    # print(predictions_train)
    # print(predictions)
    # exit()
    count = len(modelsIDs) + 1
    # nrows = 4
    nrows = count//ncols
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows), sharex=True, sharey=True)
    # fig.subplots_adjust(hspace=0.5)
    # fig.subplots_adjust(wspace=0.5)
    # fig.tight_layout()
    plt.setp(ax[-1, :], xlabel='y true(s)')
    plt.setp(ax[:, 0], ylabel='y prediction(s)')
    for i, obj in enumerate(zip(modelsIDs, ax.flat), start=1): #ax.reshape(1,-1)[0][:-1]
        modelID = obj[0]
        print(modelID)
        mod = 'Model_{}'.format(modelID)
        y_train = predictions_train.loc[mod,'y_real_tr'].values
        y_pred_tr = predictions_train.loc[mod,'y_predictn'].values
        var_pred_tr = np.sqrt( predictions_train.loc[mod,'var_predictn'].values - model_noises[modelID]) 
        print(len(y_train), len(var_pred_tr))
        if log:
            y_train = np.exp(y_train)
            y_pred_tr = np.exp(y_pred_tr)
        # modelID = modelsIDs[i]
        y_true = predictions.y_real_tst
        y_pred = predictions['y_predictn{}'.format(modelID)]
        var_pred = np.sqrt(predictions['var_predictn{}'.format(modelID)] - model_noises[modelID] )
        print(len(y_true), len(var_pred))
        if log:
            y_true = np.exp(y_true)
            y_pred = np.exp(y_pred)
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)
        errors = y_pred - y_true
        per_test = [rmse,mape]
        errors_tr = y_train - y_pred_tr
        rmse_tr = sqrt(mean_squared_error(y_train, y_pred_tr))
        mape_tr = mean_absolute_percentage_error(y_train, y_pred_tr)
        per_train = [rmse_tr,mape_tr]
        scatter_plot_predictions(y_true, y_pred, y_train,y_pred_tr,errors=[var_pred_tr,var_pred], per_test=per_test, per_train=per_train, ax=obj[1], modelid =mod, n=len(y_true), color=['blue','red']) #slategray #'slategray','lightcoral'
    y_pred = predictions.y_predn_ensemble
    var_pred = np.sqrt(predictions.var_predn_ensemble - model_noises[modelID])
    if log:
        y_pred = np.exp(y_pred)
    std = np.sqrt(predictions.var_predn_ensemble)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    errors = y_pred - y_true
    per_test = [rmse, mape ]
    scatter_plot_predictions(y_true, y_pred, errors=var_pred,  per_test = per_test, ax=ax.reshape(1,-1)[0][-1], modelid='Ensemble', n=len(y_true), color=['blue','red'])
    plt.show()
    
def errorPlots_predictions(predictions_train, predictions, modelsIDs, ncols=4, log=False):
    count = len(modelsIDs) + 1
    nrows = count// ncols
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 5*nrows), sharex=True, sharey=True)
    plt.setp(ax[-1, :], xlabel='prediction error(s)')
    plt.setp(ax[:, 0], ylabel='count')
    for i, obj in enumerate(zip(modelsIDs, ax.flat)): #ax.reshape(1,-1)[0][:-1]
        modelID = obj[0]
        print(modelID)
        mod = 'Model_{}'.format(modelID)
        y_train = predictions_train.loc[mod,'y_real_tr'].values
        y_pred_tr = predictions_train.loc[mod,'y_predict'].values
        if log:
            y_train = np.exp(y_train)
            y_pred_tr = np.exp(y_pred_tr)
        # modelID = modelsIDs[i]
        y_true = predictions.y_real_tst
        y_pred = predictions['y_predictn{}'.format(modelID)]
        if log:
            y_true = np.exp(y_true)
            y_pred = np.exp(y_pred)
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)
        errors = y_pred - y_true
        per_test = [rmse,mape]
        errors_tr = y_train - y_pred_tr
        rmse_tr = sqrt(mean_squared_error(y_train, y_pred_tr))
        mape_tr = mean_absolute_percentage_error(y_train, y_pred_tr)
        per_train = [rmse_tr,mape_tr]
        hist_error(errors, errors_tr, per_test=per_test, per_train=per_train,modelid=mod, ax=obj[1] , color=['blue','red']) #'slategray','lightcoral'

    y_pred = predictions.y_predn_ensemble
    if log:
        y_pred = np.exp(y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    errors = y_pred - y_true
    per_test = [rmse,mape]
    hist_error(errors=errors, per_test = per_test,modelid='Ensemble', ax=ax.reshape(1,-1)[0][-1] ,color=['blue','red']) #'slategray','lightcoral'
    plt.show()

def check_pred_performance_single_model(modelID=None, testID=None,model_name=None,region=None, log=False,
                                         gama=None,quant=None, scaling_factors=None, show_relevances=False,weight_treshold=None):
    
    # y_true, y_pred,per_test = per_test, ax=ax.reshape(1,-1)[0][-1], rmse=rmse, mape=mape, modelid='Ensemble', n=500, color=['slategray','lightcoral']
    predictions_onTrain = pd.read_csv('/home/sevin/Desktop/projects/TravelTime_Prediction/V_RVM_amb/code/{}/{}/predictions_onTrain_model{}_g{}_q{}.txt'.format(region,model_name,modelID,gama,quant))
    predictions_onTest = pd.read_csv('/home/sevin/Desktop/projects/TravelTime_Prediction/V_RVM_amb/code/{}/{}/predictions_onTest_model{}_test{}_g{}_q{}.txt'.format(region,model_name, modelID, testID,gama,quant))
    # predictions_onTrain = predictions_onTest
    # print(predictions_onTrain)
    # exit()
    predictions_onTrain = predictions_onTrain.rename(columns={'y_real_tr':'y_real_tst'})
    print(predictions_onTrain)
    if scaling_factors is not None:
        y_train = predictions_onTrain.y_real_tst
        y_train = y_train * scaling_factors['std_target'] + scaling_factors['mean_target']
        y_pred_tr = predictions_onTrain.y_predictn
        y_pred_tr = y_pred_tr * scaling_factors['std_target'] + scaling_factors['mean_target']
    else:
        y_train = predictions_onTrain.y_real_tst
        y_pred_tr = predictions_onTrain.y_predictn
    if log:
        y_train = np.exp(y_train)
        y_pred_tr = np.exp(y_pred_tr)
    # modelID = modelsIDs[i]
    if scaling_factors is not None:
        y_true = predictions_onTest.y_real_tst
        y_true = y_true * scaling_factors['std_target'] + scaling_factors['mean_target']
        y_pred = predictions_onTest.y_predictn
        y_pred = y_pred * scaling_factors['std_target'] + scaling_factors['mean_target']
    else:
        y_true = predictions_onTest.y_real_tst
        y_pred = predictions_onTest.y_predictn
    plt.hist(y_true,bins=20)
    plt.hist(y_pred,bins=20)
    plt.legend(['true','pred'])
    plt.show()
    if log:
        y_true = np.exp(y_true)
        y_pred = np.exp(y_pred)
    try:
        rmse_tr = sqrt(mean_squared_error(y_train, y_pred_tr))
        mape_tr = mean_absolute_percentage_error(y_train, y_pred_tr)
    except:
        rmse_tr = None
        mape_tr = None
    
    per_train = [rmse_tr,mape_tr]
    try:
        rmse = sqrt(mean_squared_error(y_true,  y_pred))
        mape = mean_absolute_percentage_error(y_true,  y_pred)
    except:
        rmse=None
        mape=None
    per_test = [rmse,mape]
    modelid = 'Model{}'.format(modelID)

    #########
    file = '{}/{}/VI_g{}_q{}_{}.pickle'.format(region, model_name, gama,quant, modelID)
    vis = []
    with open(file, 'rb') as fr:
        try:
            while True:
                    vis.append(pickle.load(fr))
        except EOFError:
            pass 
    vi = vis[0] 
    if scaling_factors is not None:
        noise_var = vi.f_p/vi.e_p * (scaling_factors['std_target'] **2)
    else:
        noise_var = vi.f_p/vi.e_p
    print('noise_var ', np.sqrt(noise_var))
    
    ################ errorbar plot ################
    fig, ax = plt.subplots(ncols=2, figsize=(10, 3.5))
    distr = 'normal'
    if distr == 'normal':
        if scaling_factors is not None:
            var_tr = predictions_onTrain.var_predictn * scaling_factors['std_target'] **2 #- noise_var
            var_tst = predictions_onTest.var_predictn* scaling_factors['std_target'] **2 #-noise_var
            print(np.sqrt(var_tst))
        else:
            var_tr = predictions_onTrain.var_predictn - noise_var
            var_tst = predictions_onTest.var_predictn -noise_var
        error_tr, error_tst = calculate_1std_error(var_tr, var_tst)

    elif distr == 't':
        var_tr = predictions_onTrain.var_predict
        var_tst = predictions_onTest.var_predict
        error_tr, error_tst = calculate_1std_error(var_tr, var_tst)
 
    else:
        raise NotImplementedError
    if show_relevances:
        # fig, ax = plt.subplots(ncols=2, figsize=(10, 3.5))
        fig, ax = plt.subplots( figsize=(5, 5))
        print('distance coefficient ', vi.mu_p[-1])
        relevances = [True if np.abs(vi.mu_p[i]) > weight_treshold else False for i in range(len(vi.mu_p))]
        plot_relevances(relevances, y_train, y_pred_tr, n=len(y_train), ax=ax)
        plt.show()
        exit()

    errorbar_plot_predictions(y_train, y_pred_tr, errors = error_tr, n=1000, ax=ax[0], title='Trian {}'.format(mape_tr)) #len(predictions_onTrain)
    errorbar_plot_predictions(y_true, y_pred, errors = error_tst, n=1000, ax=ax[1], title='Test {}'.format(mape)) #len(predictions_onTest)
    plt.show()

def remove_duplicate_rows(data_batchID):
    parent_folder =  '/home/sevin/Desktop/projects/TravelTime_Prediction/'
    project_name = 'V_RVM_amb'
    coun = 0
    with open('scaling_factors_pre.pickle', 'rb') as handle:
                scaling_factors = pickle.load(handle)
    train_dataset = pd.read_csv(parent_folder + project_name + '/data/processed_data_SM/MNSouth_weekdays_seg2-3_batch_{}.txt'.format(data_batchID))
    # print(train_dataset)
    X_train = train_dataset[
            ['lat_origin', 'long_origin', 'lat_destination', 'long_destination',  'hour', 'minute']].values # 'day','ManhattanDistance', 'planeDistance',
    y_train = train_dataset['travel_time'].values
    x_tr, X_train_scaled = preprocessing_(X_train, scaling_factors)
    unq, count = np.unique(X_train_scaled, axis=0, return_counts=True)
    repeated_groups = unq[count>1]

    if len(repeated_groups)>0:
        for repeated_group in repeated_groups:
            repeated_idx = np.argwhere(np.all(X_train_scaled == repeated_group, axis=1))
            inds = repeated_idx.ravel()
            new_x = X_train_scaled[inds]
            new_x = np.mean(new_x, axis=0).reshape(1,-1)
            new_y = np.mean(y_train[inds])
            print(inds)
        print(len(X_train_scaled))
        X_train_scaled = np.delete(X_train_scaled,inds,axis=0)
        y_train = np.delete(y_train,inds,axis=0)

        X_train_scaled = np.concatenate([X_train_scaled,new_x], axis=0)
        y_train = np.append(y_train,new_y)

def plot_everything_inGrid(n_dataBatch, reg, boro= None, region = None):
    parent_folder =  '/home/sevin/Desktop/projects/TravelTime_Prediction/'
    project_name = 'V_RVM_amb'
    data = pd.read_csv(parent_folder + project_name + '/data/processed_data_SM/{}_{}/{}_weekdays_seg2-3_batch_2018-2019_total.txt'.format(reg,n_dataBatch, reg))
    data = data[data['To'].apply(lambda x: x not in ['WTLIB','WTELI','GOVIA'])]
    # path = "/home/sevin/Downloads/nycshapefile/"
    # street_map = gpd.read_file(path + 'geo_export_5c6a899d-8870-42e4-a1ab-2ab8b76541ee.shp')
    path_to_data = gpd.datasets.get_path("nybb")
    gdf_mn = gpd.read_file(path_to_data)
    gdf_mn = gdf_mn.to_crs({'init': 'epsg:4326'})
    path='/home/sevin/Documents/Research/Data/Data for Columbia/Shapefiles/'
    EMS_Atoms_cntrs = gpd.read_file( path+ 'Atom_Centroids_20190822.shp' )
    atomList = EMS_Atoms_cntrs[EMS_Atoms_cntrs['AGENCY'] == boro]['ATOM'].to_list()
    # print(len(atomList))
    atomList = [item for item in atomList if item not in ['WTLIB','WTELI','GOVIA']]
    print(len(atomList))
    # atomList = EMS_Atoms_cntrs['ATOM'].tolist()
    path='/home/sevin/Documents/Research/Data/Data for Columbia/Shapefiles/' 
    EMS_Atoms = gpd.read_file( path+ 'EMS_Atoms.shp', ignore_fields=['DISP_AREA', 'DIVISION', 'ATOM4'] )
    if region:
        EMS_Atoms = EMS_Atoms[EMS_Atoms['DISP_FREQ']==region]
    geodf = EMS_Atoms.to_crs({'init': 'epsg:4326'})
    
    dfAtoms = geodf[geodf['ATOM'].isin(atomList)]
    # print(dfAtoms['geometry'].bounds)
    # exit()
    count_D = pd.DataFrame(data[['To']].value_counts(), columns=['value']).reset_index(drop=False)
    df_cnt_D = dfAtoms.merge(count_D, left_on='ATOM',right_on='To')
    count_O = pd.DataFrame(data[['From']].value_counts(), columns=['value']).reset_index(drop=False)
    df_cnt_O = dfAtoms.merge(count_O, left_on='ATOM',right_on='From')
    count = count_D.merge(count_O, left_on='To', right_on='From', how='outer')
    avg_TT_O = data[['From','travel_time']].groupby('From').mean().rename(columns={'travel_time':'value'})
    df_avg_TT_O = dfAtoms.merge(avg_TT_O, left_on='ATOM',right_on='From')
    std_TT_O = data[['From','travel_time']].groupby('From').std().rename(columns={'travel_time':'value'})
    df_std_TT_O = dfAtoms.merge(std_TT_O, left_on='ATOM',right_on='From')
    avg_TT_D = data[['To','travel_time']].groupby('To').mean().rename(columns={'travel_time':'value'})
    df_avg_TT_D = dfAtoms.merge(avg_TT_D, left_on='ATOM',right_on='To')
    std_TT_D = data[['To','travel_time']].groupby('To').std().rename(columns={'travel_time':'value'})
    df_std_TT_D = dfAtoms.merge(std_TT_D, left_on='ATOM',right_on='To')
    dfs = [df_avg_TT_O, df_avg_TT_D, df_std_TT_O, df_std_TT_D, df_cnt_O, df_cnt_D]
    colors = ['Greens','Greens','Greens','Greens','Blues','Blues']
    # print(street_map)
    # exit()
    mi_c = min(np.amin(count['value_x']), np.amin(count['value_y']))#0
    ma_c = max(np.amax(count['value_x']), np.amax(count['value_y']))#0
    mi_t = min(np.amin(df_avg_TT_D['value']), np.amin(df_avg_TT_O['value']))#0
    ma_t = max(np.amax(df_avg_TT_D['value']), np.amax(df_avg_TT_O['value']))#0
    mi_st = min(np.amin(df_std_TT_O['value']), np.amin(df_std_TT_D['value']))#0
    ma_st = max(np.amax(df_std_TT_O['value']), np.amax(df_std_TT_D['value']))#0
    #############################
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
    fig.subplots_adjust(wspace=0.5, hspace=0)
    # fig.subplots_adjust(right=0.8)
    axs = axes.ravel()
    for i, obj in enumerate(zip(axes.flat, dfs[:2],colors[:2])):
        ax = obj[0]
        d = obj[1]
        c = obj[2]
        dp = d.plot(column='value', cmap='viridis', ax=ax, legend=0,vmin=mi_t, vmax=ma_t,linewidth=0.1) #, vmin=mi_t, vmax=ma_t
        ax.axis('off')
    im = plt.gca().get_children()[0]
    cax = fig.add_axes([0.3,0.25,0.4,0.01])
    cmap = mpl.cm.viridis
    bounds = [0, 200, 400, 600,800,1000]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    aa=fig.colorbar(im, cax=cax, orientation='horizontal', ticks=bounds, norm=norm, spacing='uniform')
    plt.title('Average Travel Time (s) per Atom', size=14,fontweight="bold")
    aa.ax.tick_params(labelsize=14) 
    # plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
    fig.subplots_adjust(wspace=0.5, hspace=0)
    # fig.subplots_adjust(right=0.8)
    axs = axes.ravel()
    for i, obj in enumerate(zip(axes.flat, dfs[2:4],colors[2:4])):
        ax = obj[0]
        d = obj[1]
        c = obj[2]
        d.plot(column='value', cmap='viridis', ax=ax, legend=0,vmin=mi_st, vmax=ma_st,linewidth=0.1) #, vmin=mi_t, vmax=ma_t #viridis
        ax.axis('off')
    im = plt.gca().get_children()[0]
    cax = fig.add_axes([0.3,0.25,0.4,0.01])
    cmap = mpl.cm.viridis
    bounds = [0, 200, 400, 600,800,1000] #[0, 100, 200, 300, 400, 500,]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    aa = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=bounds, norm=norm, spacing='uniform')
    plt.title('Std. of Travel Time (s) per Atom', size=14,fontweight="bold")
    aa.ax.tick_params(labelsize=14) 
    # plt.show()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
    fig.subplots_adjust(wspace=0.5, hspace=0)
    for i, obj in enumerate(zip(axes.flat, dfs[4:],colors[4:])):
        ax = obj[0]
        d = obj[1]
        c = obj[2]
        d.plot(column='value', cmap='viridis', ax=ax, legend=0, vmin=mi_c, vmax=ma_c)
        ax.axis('off')
    im = plt.gca().get_children()[0]
    cax = fig.add_axes([0.3,0.2,0.4,0.01])
    cmap = mpl.cm.viridis
    bounds = [0, 1000, 2000, 3000, 4000,5000]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    aa =fig.colorbar(im, cax=cax, orientation='horizontal', ticks=bounds, norm=norm, spacing='uniform')
    plt.title('Trip Count per Atom', size=14,fontweight="bold")
    aa.ax.tick_params(labelsize=14) 
    plt.show()
    # exit()
def k_fold():
    pass
def get_noises(region,model_name, num_ofModels,quant,gama, width_parameters, plot=False):

    noises = []
    models_noises = {}
    for k in range(0,num_ofModels):
        if width_parameters:
            quant = width_parameters[k][0]
            gama = width_parameters[k][1]
        else:
            quant=quant
            gama=gama
        # print(quant,gama)
        file = '{}/{}/VI_g{}_q{}_{}.pickle'.format(region, model_name, gama,quant, k)
        # print(file)
        vis = []
        with open(file, 'rb') as fr:
            try:
                while True:
                    vis.append(pickle.load(fr))
            except EOFError:
                pass 
        vi = vis[0]
        noise = vi.f_p/vi.e_p * (scaling_factors['std_target'] **2)
        noises.append(noise)
        models_noises.update({k:noise})
    print(np.min( np.sqrt(noises)),np.max( np.sqrt(noises)))
    if plot:
        noises_std = [np.sqrt(nois) for nois in noises]
        x = np.arange(num_ofModels).tolist()    
        plt.hist(np.sqrt(noises),bins=25,density=True,color='royalblue') #,rwidth=0.6
        plt.xlabel('Predictive Noise (s)',fontsize=12)
        plt.show()
    return models_noises
def plot_elbos(region,model_name, num_ofModels,quant,width_parameters):


    for k in range(0,num_ofModels):
        quant = width_parameters[k][0]
        gama = width_parameters[k][1]
        # print(quant,gama)
        file = '{}/{}/VI_g{}_q{}_{}.pickle'.format(region, model_name, gama,quant, k)
        # print(file)
        vis = []
        with open(file, 'rb') as fr:
            try:
                while True:
                    vis.append(pickle.load(fr))
            except EOFError:
                pass 
        vi = vis[0]
        elbo = vi.Lv
        numItr = len(elbo)
        x = np.arange(1,numItr+1,1).tolist()
        # print(self.Lv)
        plt.plot(x,elbo) #,linewidth=0.8
        ax = plt.axes()
        ax.yaxis.grid(True)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
        plt.xlabel('Iteration',fontsize=14)
        plt.ylabel('ELBO',fontsize=14)
        
    ax.legend(['m{}'.format(k) for k in range(0,num_ofModels)],loc='upper right',fontsize=8)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def stemPlot_relevance(modelid, quant, prune=False, prune_w_weight=False, weight_treshold=None, prune_w_alpha = False, alpha_treshold=None):
    quant = width_parameters[modelid][0]
    gama = width_parameters[modelid][1]
    # print(quant,gama)
    file = '{}/{}/VI_g{}_q{}_{}.pickle'.format(region, model_name, gama,quant,modelid)
    # print(file)
    vis = []
    with open(file, 'rb') as fr:
        try:
            while True:
                vis.append(pickle.load(fr))
        except EOFError:
            pass 
    vi = vis[0]
    
    # keep_mu_p = np.abs(vi.mu_p) > mu_treshold
    if prune:
        if prune_w_weight==True:
            relevance = [vi.mu_p[i] if np.abs(vi.mu_p[i]) > weight_treshold else None for i in range(len(vi.mu_p))]
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.stem(vi.mu_p)
            ax.stem(relevance,markerfmt='C2o')
            ax.set_xlabel('Input vectors')
            ax.set_ylabel('Weights')
            ax.legend(['Weights', 'Relevance vectors'])
            alpha = vi.a_p/vi.b_p#np.log10(self.a_p/self.b_p).squeeze()
            # print(alpha)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.hist(vi.a_p/vi.b_p, bins=50)
            plt.show()
        elif prune_w_alpha==True:
            relevance = [vi.a_p[i]/vi.b_p[i] if np.abs(vi.a_p[i]/vi.b_p[i]) < alpha_treshold else None for i in range(len(vi.a_p))]
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.stem(vi.a_p/vi.b_p)
            ax.stem(relevance,markerfmt='C2o')
            mup = vi.mu_p.squeeze()
            # print(alpha)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.hist(mup, bins=50)
            plt.show()
        else:
            raise NotImplementedError
    else:
        
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.stem(vi.mu_p)
        plt.show()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.hist(vi.mu_p, bins=50)
        # ax.stem(relevance, markerfmt='C2o')
        ax.set_xlabel('Input vectors')
        ax.set_ylabel('Weights')
        ax.legend(['Weights', 'Relevance vectors'])
        plt.show()
        alpha = np.log10(vi.a_p/vi.b_p).squeeze() #vi.a_p/vi.b_p#
        # print(alpha)
        print(vi.f_p/vi.e_p * scaling_factors['std_target'] **2)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.hist(alpha, bins=50) #vi.b_p/vi.a_p
        plt.show()

        

if __name__ =='__main__':
    # testSetID = ['013BA','013DA']#['001AC' , '005FA'] #['009GA',  '009AB']#['001AC' , '005FA'] #['013BA','013DA'] #testSetID = 0
    # testSetID = [('009GA',  '009AB','green'), ('013BA','013DA','purple'), ('001AC' , '005FA','yellow')]
    num_ofModels = 20#20#4
    region= 'MNCentral'#'UWS' #'UWS' #'MNCentral'
    n_dataBatch = 50 #10 #'aggregated'
    # modelsIDs = np.arange(1,num_ofModels) #[1,2,4,5,6,7,8]#np.arange(9)
    testSetID = 3
    # rem = [0,1]
    modelsIDs = np.arange(num_ofModels)
    modelsIDs = np.delete(modelsIDs, testSetID )
    model_name = 'ensembles_mulikernel_wDist_wexTime'#'ensembles_mulikernel_wDist_wexTime_v2'#'ensembles_mulikernel_wDist_wexTime'#'ensembles_mulikernel_wDist'

    if True:
        [['018MA', '014HA'],['018MA', '018GA'],['018MA', '018GB'],['018MA', '018HB'],
                    ['018MA', '018IA'],['018MA', '018KA'],['018MA', '020AA']]
        [['023FA','023CB'],['023FA','023HB'],['023FA','023HC'],['023FA','023IA'],['023FA','023KA']]
        
        # plot_everything_inGrid(n_dataBatch=50, reg = region, boro='MN', region='Manhattan Central')
        # atoms = [('010GB' , '014HA','blue'),('017AA', '014CB','green') , ('018MA' , '018HB','purple'),('020AA' , '018HB','yellow')]
        color = 'orchid'
        atoms = [('018MA', '014HA',color),('018MA', '018GB',color),('018MA', '018HB',color),
                    ('018MA', '018IA',color),('018MA', '020AA',color),('018MA', '020BA',color),
                    ('023FA','023CB',color),('023FA','023HB',color),('023FA','023HC',color),('023FA','023IA',color),('023FA','023KA',color)]
        atoms = [('023FA','023CB',color),('023FA','023HB',color),('023FA','023HC',color),
                    ]
        plot_data_info(n_dataBatch=50, reg = region, TT_distr = False, hr_distr = False, OD_distr = True, testSetID=testSetID, atoms=atoms,plot_selected_atoms=True ) #['001AC' , '005FA']
