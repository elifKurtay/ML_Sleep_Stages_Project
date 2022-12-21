import numpy as np
from abd_helper import *
from sklearn.metrics import balanced_accuracy_score
from helpers import *
from plot import *
import matplotlib.pyplot as plt
from constants import *
from helpers import *
from loader import *
from plot import *

#Method 0-1: This is not a ML method
    #applied per individual
#Give two weights for emfit and somnofy
#where weights are defined as: W and 100-W
#calculate: [emfit*W+somnofy*(100-W)]/100
#round the value to 0-1-2-3
#range over all w values (0-100) calculate accuracy

def M_01(SS_Emfit,SS_Somnofy,SS_PSG):
    accuracy_array=np.ones((11,2))
    #accuracy_array[0]=[0,balanced_accuracy_score(SS_PSG[:divide_ind],SS_Somnofy[:divide_ind])]
    #accuracy_array[10]=[100,balanced_accuracy_score(SS_PSG[:divide_ind],SS_Emfit[:divide_ind])]
    for a in [0,10,20,30,40,50,60,70,80,90,100]:
        weights=np.array([a,100-a])
        new_SS=build_new_ss(weights,SS_Emfit,SS_Somnofy)
        accuracy_array[int(a/10)]=[a,balanced_accuracy_score(SS_PSG,new_SS)]
    return accuracy_array

def mse_gd(ss_emfit,ss_somnofy,ss_psg, max_iters=150, gamma=0.005,w0=False):
    y,tx=build_y_tx(ss_emfit,ss_somnofy,ss_psg,w0)
    loss=0
    w = np.zeros((tx.shape[1],1), dtype=float) #initial_w
    for n_iter in range(max_iters):
        gradient = comp_grad(tx,comp_error(y,tx,w))
        loss = mse_loss(y, tx, w)
        w = w - gamma * gradient
    return w, loss

""" def grid_search_sleep_stages(ss_emfit,ss_somnofy,ss_psg, max_iters=150, gamma=0.005):
    y,tx=build_y_tx(ss_emfit,ss_somnofy,ss_psg,False)
    loss=0
    w = np.zeros((4), dtype=float) #initial_w:(0,0,0,0)
    w=np.c_[w,w] #w=[[0,0],[0,0],[0,0],[0,0]]
    for n_iter in range(max_iters):
        gradient = comp_grad(tx,comp_error_sleep_stages(y,tx,w))
        loss = mse_loss(y, tx, w)
        w = w - gamma * gradient
    return w, loss """

#Method X:
#Devide data set into:
#emfit value - somnofy value:
#0-1,0-2,0-3
#1-0,1-2,1-3
#2-0,2-1,2-3
#3-0,3-1,3-2
#give 3 or 2 weights per data group
#use mse - find optimal w's
#using w's find new sleep stage
#for agreement use what is agreed on

def method_x(sleep_stages, max_iters=150, gamma=0.005):
    weights=np.zeros((12,2))
    counter=0
    for i in [1,2,3,4]:
        for j in [1,2,3,4]:
            if i != j:
                ss=sleep_stages.loc[(sleep_stages["sleep_stage_num_emfit"]==i) &
                    (sleep_stages["sleep_stage_num_somnofy"]==j)]
                if not ss.empty:
                    ss_emfit=ss["sleep_stage_num_emfit"].to_numpy()
                    ss_somnofy=ss["sleep_stage_num_somnofy"].to_numpy()
                    ss_psg=ss["sleep_stage_num_psg"].to_numpy()
                    w,loss=mse_gd(ss_emfit,ss_somnofy,ss_psg, max_iters, gamma)
                    weights[counter]=w.reshape(1,2)
                counter+=1
    return weights

#shift everything by 1


""" def accur_app(test_list,w0=False):
    max_len=0
    for i in range(len(test_list)):
        subjectID = test_list[i]
        sleep_stages = read_patient_data(subjectID)
        len=len(sleep_stages["sleep_stage_num_psg"])
        if max_len<len:
            max_len=len

    ss_emfit=np.zeros
    ss_somnofy=sleep_stages["sleep_stage_num_somnofy"].to_numpy()
    ss_psg=sleep_stages["sleep_stage_num_psg"].to_numpy()
    subjectID = test_list[i]
    sleep_stages = read_patient_data(subjectID)
    ss_emfit=sleep_stages["sleep_stage_num_emfit"].to_numpy()
    ss_somnofy=sleep_stages["sleep_stage_num_somnofy"].to_numpy()
    ss_psg=sleep_stages["sleep_stage_num_psg"].to_numpy()
    if w0 is True:
        w=[0,0,0]
    else:
        w=[0,0] """
        
