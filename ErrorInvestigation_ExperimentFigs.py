import time
import os
import numpy as np
np.random.seed(0)
import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"]=10
import pandas as pd
import matplotlib as mpl
import scipy.interpolate
import scipy.stats

path=os.getcwd()

store_capture=[]
store_mean_capture=[]
store_capture_20=[]
store_mean_capture_20=[]
store_CI_sumsquares=[]
store_CI_20_sumsquares=[]
store_delta=[]
store_delta_20=[]
store_CRPS=[]
store_Reli=[]

sites=['Bdesh','Tanz']
costs=['mse','nse','kge','ai']
weights=np.arange(0,4,1)
vars=['frc','swot']
Ensemble_Size=200

exp_names=[]

for a in range (0,3):
    for b in range(0,5):
        for c in range (0,7):
            for d in range(0,2):
                exp=sites[a] + '_' + costs[b] + '_w' + str(weights[c]) + '_' + vars[d] + 'time'
                exp_names.append(exp)

'''
Test Lengths:
Bdesh FRC: 533
Bdesh SWOT: 244
Tanz FRC: 77
Tanz SWOT: 23
'''
sizes=np.array([[533,77,54],[243,23,54]])
test_lengths=[]

for a in range (0,3):
    for b in range(0,5):
        for c in range (0,7):
            for d in range(0,2):
                test_lengths=np.append(test_lengths,sizes[d,a])


for z in range (0, len(test_lengths)):
    Experiment = exp_names[z]
    test_len=int(test_lengths[z])

    X_test=np.load(path+"\\"+Experiment+"\\"+"X_test.npy")
    X_test=np.reshape(X_test,(Ensemble_Size,test_len))
    X_test=X_test[0,:]
    Y_test=np.load(path+"\\"+Experiment+"\\"+"Y_test.npy")
    Y_test=np.reshape(Y_test,(Ensemble_Size,test_len))
    Y_test=Y_test[0,:]
    Y_test_pred=np.load(path+"\\"+Experiment+"\\"+"Y_test_pred.npy")
    Y_test_pred=np.reshape(Y_test_pred,(Ensemble_Size,test_len))


    single=90/25.4
    half=140/25.4
    full=190/25.4

    #Percent Capture and CI Reliability

    capture_all=np.less_equal(Y_test,np.max(Y_test_pred,axis=0))*np.greater_equal(Y_test,np.min(Y_test_pred,axis=0))*1
    capture_99=np.less_equal(Y_test,np.percentile(Y_test_pred,99.5,axis=0))*np.greater_equal(Y_test,np.percentile(Y_test_pred,0.5,axis=0))*1
    capture_95=np.less_equal(Y_test,np.percentile(Y_test_pred,97.5,axis=0))*np.greater_equal(Y_test,np.percentile(Y_test_pred,2.5,axis=0))*1
    capture_90=np.less_equal(Y_test,np.percentile(Y_test_pred,95,axis=0))*np.greater_equal(Y_test,np.percentile(Y_test_pred,5,axis=0))*1
    capture_80=np.less_equal(Y_test,np.percentile(Y_test_pred,90,axis=0))*np.greater_equal(Y_test,np.percentile(Y_test_pred,10,axis=0))*1
    capture_70=np.less_equal(Y_test,np.percentile(Y_test_pred,85,axis=0))*np.greater_equal(Y_test,np.percentile(Y_test_pred,15,axis=0))*1
    capture_60=np.less_equal(Y_test,np.percentile(Y_test_pred,80,axis=0))*np.greater_equal(Y_test,np.percentile(Y_test_pred,20,axis=0))*1
    capture_50=np.less_equal(Y_test,np.percentile(Y_test_pred,75,axis=0))*np.greater_equal(Y_test,np.percentile(Y_test_pred,25,axis=0))*1
    capture_40=np.less_equal(Y_test,np.percentile(Y_test_pred,70,axis=0))*np.greater_equal(Y_test,np.percentile(Y_test_pred,30,axis=0))*1
    capture_30=np.less_equal(Y_test,np.percentile(Y_test_pred,65,axis=0))*np.greater_equal(Y_test,np.percentile(Y_test_pred,35,axis=0))*1
    capture_20=np.less_equal(Y_test,np.percentile(Y_test_pred,60,axis=0))*np.greater_equal(Y_test,np.percentile(Y_test_pred,40,axis=0))*1
    capture_10=np.less_equal(Y_test,np.percentile(Y_test_pred,55,axis=0))*np.greater_equal(Y_test,np.percentile(Y_test_pred,45,axis=0))*1

    capture_all_20=capture_all*np.less(Y_test,0.2)
    capture_99_20=capture_99*np.less(Y_test,0.2)
    capture_95_20=capture_95*np.less(Y_test,0.2)
    capture_90_20=capture_90*np.less(Y_test,0.2)
    capture_80_20=capture_80*np.less(Y_test,0.2)
    capture_70_20=capture_70*np.less(Y_test,0.2)
    capture_60_20=capture_60*np.less(Y_test,0.2)
    capture_50_20=capture_50*np.less(Y_test,0.2)
    capture_40_20=capture_40*np.less(Y_test,0.2)
    capture_30_20=capture_30*np.less(Y_test,0.2)
    capture_20_20=capture_20*np.less(Y_test,0.2)
    capture_10_20=capture_10*np.less(Y_test,0.2)

    length_20=np.sum(np.less(Y_test,0.2))
    capture_all_sum=np.sum(capture_all)
    capture_99_sum=np.sum(capture_99)
    capture_95_sum=np.sum(capture_95)
    capture_90_sum=np.sum(capture_90)
    capture_80_sum=np.sum(capture_80)
    capture_70_sum=np.sum(capture_70)
    capture_60_sum=np.sum(capture_60)
    capture_50_sum=np.sum(capture_50)
    capture_40_sum=np.sum(capture_40)
    capture_30_sum=np.sum(capture_30)
    capture_20_sum=np.sum(capture_20)
    capture_10_sum=np.sum(capture_10)

    capture_all_20_sum=np.sum(capture_all_20)
    capture_99_20_sum=np.sum(capture_99_20)
    capture_95_20_sum=np.sum(capture_95_20)
    capture_90_20_sum=np.sum(capture_90_20)
    capture_80_20_sum=np.sum(capture_80_20)
    capture_70_20_sum=np.sum(capture_70_20)
    capture_60_20_sum=np.sum(capture_60_20)
    capture_50_20_sum=np.sum(capture_50_20)
    capture_40_20_sum=np.sum(capture_40_20)
    capture_30_20_sum=np.sum(capture_30_20)
    capture_20_20_sum=np.sum(capture_20_20)
    capture_10_20_sum=np.sum(capture_10_20)

    #print("Total values capture: "+str(capture_all_sum)+" of "+str(test_len)+"; "+str(capture_all_sum/test_len)+" percent. Mean percent capture: "+str(capture_mean))
    #print("Total values sub 0.2 mg/L capture: "+str(capture_all_20_sum)+" of "+str(length_20)+"; "+str(capture_all_20_sum/length_20)+" percent. Mean percent capture: "+str(capture_20_mean))

    store_capture = np.append(store_capture,capture_all_sum/test_len)
    store_capture_20 = np.append(store_capture_20,capture_all_20_sum/length_20)


    x=[0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99,100]
    capture=[capture_10_sum/test_len,capture_20_sum/test_len,capture_30_sum/test_len,capture_40_sum/test_len,capture_50_sum/test_len,capture_60_sum/test_len,capture_70_sum/test_len,capture_80_sum/test_len,capture_90_sum/test_len,capture_95_sum/test_len,capture_99_sum/test_len,capture_all_sum/test_len]
    capture_20=[capture_10_20_sum/length_20,capture_20_20_sum/length_20,capture_30_20_sum/length_20,capture_40_20_sum/length_20,capture_50_20_sum/length_20,capture_60_20_sum/length_20,capture_70_20_sum/length_20,capture_80_20_sum/length_20,capture_90_20_sum/length_20,capture_95_20_sum/length_20,capture_99_20_sum/length_20,capture_all_20_sum/length_20]

    capture_sum_squares=(0.1-capture_10_sum/test_len)**2+(0.2-capture_20_sum/test_len)**2+(0.3-capture_30_sum/test_len)**2+(0.4-capture_40_sum/test_len)**2+(0.5-capture_50_sum/test_len)**2+(0.6-capture_60_sum/test_len)**2+(0.7-capture_70_sum/test_len)**2+(0.8-capture_80_sum/test_len)**2+(0.9-capture_90_sum/test_len)**2+(1-capture_all_sum/test_len)**2
    capture_20_sum_squares=(0.1-capture_10_20_sum/length_20)**2+(0.2-capture_20_20_sum/length_20)**2+(0.3-capture_30_20_sum/length_20)**2+(0.4-capture_40_20_sum/length_20)**2+(0.5-capture_50_20_sum/length_20)**2+(0.6-capture_60_20_sum/length_20)**2+(0.7-capture_70_20_sum/length_20)**2+(0.8-capture_80_20_sum/length_20)**2+(0.9-capture_90_20_sum/length_20)**2+(1-capture_all_20_sum/length_20)**2

    store_CI_sumsquares = np.append(store_CI_sumsquares, capture_sum_squares)
    store_CI_20_sumsquares = np.append(store_CI_20_sumsquares, capture_20_sum_squares)


    #print(capture_sum_squares)
    #print(capture_20_sum_squared)

    fig, (ax1,ax2)=plt.subplots(1,2,figsize=(full,single))
    ax1.set_xlim([0,1])
    ax1.plot(x,x,c='k')
    ax1.scatter(x,capture)
    ax1.set_xlabel("Ensemble Confidence Interval")
    ax1.set_ylabel("Percent Capture")
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    ax2.plot(x,x,c='k')
    ax2.scatter(x,capture_20)
    ax2.set_xlabel("Ensemble Confidence Interval")
    ax2.set_ylabel("Percent Capture (HH FRC <= 0.2 mg/L")
    ax2.set_ylim([0,1])
    plt.subplots_adjust(left=0.1,bottom=0.12, top=0.98)
    plt.savefig(path+"\\"+Experiment+"\\CI_Reliability.png")
    plt.close()

    #Rank Histogram
    rank=[]
    for a in range(0,len(Y_test)):
        n_lower=np.sum(np.greater(Y_test[a],Y_test_pred[:,a]))
        n_equal=np.sum(np.equal(Y_test[a],Y_test_pred[:,a]))
        deviate_rank=np.random.random_integers(0,n_equal)
        rank=np.append(rank,n_lower+deviate_rank)


    rank_hist=np.histogram(rank,bins=Ensemble_Size+1)
    fig=plt.figure(figsize=(full,single))
    plt.hist(rank,bins=Ensemble_Size+1,density=True)
    plt.xlabel('Rank')
    plt.ylabel('Probability')
    plt.savefig(path+"\\"+Experiment+"\\RankHistogram.png")
    plt.close()

    delta=np.sum((rank_hist[0]-(test_len/(Ensemble_Size+1)))**2)
    delta_0=210*test_len/(Ensemble_Size+1)
    #delta_score=delta/delta_0
    #print(delta_score)

    store_delta = np.append(store_delta,delta/delta_0 )
    #Stacked RH

    rank_lower=[]
    rank_upper=[]
    for a in range(0,len(Y_test)):
        if Y_test[a]<0.2:
            n_lower=np.sum(np.greater(Y_test[a],Y_test_pred[:,a]))
            n_equal=np.sum(np.equal(Y_test[a],Y_test_pred[:,a]))
            deviate_rank=np.random.random_integers(0,n_equal)
            rank_lower=np.append(rank_lower,n_lower+deviate_rank)
        if Y_test[a]>=0.2:
            n_lower = np.sum(np.greater(Y_test[a], Y_test_pred[:, a]))
            n_equal = np.sum(np.equal(Y_test[a], Y_test_pred[:, a]))
            deviate_rank = np.random.random_integers(0, n_equal)
            rank_upper = np.append(rank_upper, n_lower + deviate_rank)

    #stacked_rh_data=np.hstack((np.transpose(rank_lower),np.transpose(rank_upper)))
    data_labels=['HH FRC < 0.2 mg/L','HH FRC>= 0.2 mg/L']
    rank_hist_02=np.histogram(rank_lower,bins=Ensemble_Size+1)
    fig=plt.figure(figsize=(full,single))
    plt.hist([rank_lower,rank_upper],bins=Ensemble_Size+1,density=True,stacked=True,label=data_labels)
    plt.legend()
    plt.xlabel('Rank')
    plt.ylabel('Probability')
    plt.savefig(path+"\\"+Experiment+"\\Stacked_RankHistogram.png")
    plt.close()

    delta=np.sum((rank_hist_02[0]-(length_20/(Ensemble_Size+1)))**2)
    delta_0=Ensemble_Size*length_20/(Ensemble_Size+1)
    #delta_score=delta/delta_0
    #print(delta_score)
    store_delta_20 = np.append(store_delta_20,delta/delta_0)

    #CRPS

    alpha=np.zeros((len(Y_test),Ensemble_Size+1))
    beta=np.zeros((len(Y_test),Ensemble_Size+1))
    low_outlier=0
    high_outlier=0

    for a in range(0, len(Y_test)):
        observation = Y_test[a]
        forecast = np.sort(Y_test_pred[:, a])
        for b in range(1, Ensemble_Size):
            if observation > forecast[b]:
                alpha[a, b] = forecast[b] - forecast[b - 1]
                beta[a, b] = 0
            elif forecast[b] > observation > forecast[b - 1]:
                alpha[a, b] = observation - forecast[b - 1]
                beta[a, b] = forecast[b] - observation
            else:
                alpha[a, b] = 0
                beta[a, b] = forecast[b] - forecast[b - 1]
        # overwrite boundaries in case of outliers
        if observation < forecast[0]:
            beta[a, 0] = forecast[0] - observation
            low_outlier += 1
        if observation > forecast[Ensemble_Size-1]:
            alpha[a, Ensemble_Size] = observation - forecast[Ensemble_Size-1]
            high_outlier += 1

    alpha_bar=np.mean(alpha,axis=0)
    beta_bar=np.mean(beta,axis=0)
    g_bar=alpha_bar+beta_bar
    o_bar=beta_bar/(alpha_bar+beta_bar)

    if low_outlier > 0:
        o_bar[0] = low_outlier / test_len
        g_bar[0] = beta_bar[0] / o_bar[0]
    else:
        o_bar[0] = 0
        g_bar[0] = 0
    if high_outlier > 0:
        o_bar[Ensemble_Size] = high_outlier / test_len
        g_bar[Ensemble_Size]=alpha_bar[Ensemble_Size]/o_bar[Ensemble_Size]
    else:
        o_bar[Ensemble_Size] = 0
        g_bar[Ensemble_Size] = 0

    p_i=np.arange(0/Ensemble_Size,(Ensemble_Size+1)/Ensemble_Size,1/Ensemble_Size)

    CRPS=np.sum(g_bar*((1-o_bar)*(p_i**2)+o_bar*((1-p_i)**2)))

    store_CRPS = np.append(store_CRPS, CRPS)

    X_test=np.load(path+"\\"+Experiment+"\\"+"X_test.npy")
    Y_test=np.load(path+"\\"+Experiment+"\\"+"Y_test.npy")
    Y_test_pred=np.load(path+"\\"+Experiment+"\\"+"Y_test_pred.npy")

    #All of the above evaluated performance, this just provides a plot of the predictions and observations
    H, x_e, y_e=np.histogram2d(X_test, Y_test_pred, bins=[10,10], density=True)
    z_test=scipy.interpolate.interpn((0.5*(x_e[1:]+x_e[:-1]),0.5*(y_e[1:]+y_e[:-1])),H,np.transpose(np.vstack((X_test,Y_test_pred))),method='splinef2d',bounds_error=False)
    z_test[np.where(np.isnan(z_test))] = 0.0

    idx=z_test.argsort()
    X_test_sort,Y_test_pred,z_test=X_test[idx],Y_test_pred[idx],z_test[idx]

    fig, (ax1,ax2)=plt.subplots(1,2,figsize=(full,single))
    ax1.set_title('Training Predictions')
    ax1.scatter(X_test,Y_test,edgecolors='k',s=5,facecolors='',label='Observed')
    ax1.scatter(X_test_sort,Y_test_pred,c=z_test,s=15,edgecolors='',label='Predicted')

    ax1.set_xlabel("Tapstand FRC (mg/L)")
    ax1.set_ylabel("Household FRC (mg/L)")
    ax2.set_title('Testing Predictions')
    ax2.scatter(X_test,Y_test,edgecolors='k',s=5,facecolors='')
    ax2.scatter(X_test_sort,Y_test_pred,c=z_test,s=15,edgecolors='')
    ax2.set_xlim([0,1])
    ax2.set_ylim([0,0.2])
    ax2.set_xlabel("Tapstand FRC (mg/L)")
    ax2.set_ylabel("Household FRC (mg/L)")
    legend=fig.legend(bbox_to_anchor=(0.36, 0.895), shadow=False, fontsize='small',ncol=3,labelspacing=0.1, columnspacing=0.2, handletextpad=0.1)
    plt.subplots_adjust(left=0.135,bottom=0.12)
    plt.savefig(path+"\\"+Experiment+"\\predictions.png")
    plt.close()


df=pd.DataFrame({'Experiment':exp_names,'Capture Percent':store_capture,'Percent Capture (sub02)':store_capture_20,'CI Reliability Sum of Squares':store_CI_sumsquares,'CI Reliability Sum of Squares (sub02)':store_CI_20_sumsquares,'Delta':store_delta,'Delta (sub02)':store_delta_20,'CRPS':store_CRPS})
df.to_csv(path + '\\Super_Array_Results.csv', index=False)