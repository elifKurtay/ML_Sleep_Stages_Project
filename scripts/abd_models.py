import numpy as np
from abd_helper import *
from sklearn.metrics import balanced_accuracy_score
from helpers import *
from plot import *
import matplotlib.pyplot as plt

#Method 0-1: This is not a ML method
    #applied per individual
#Give two weights for emfit and somnofy
#where weights are defined as: W and 100-W
#calculate: [emfit*W+somnofy*(100-W)]/100
#round the value to 0-1-2-3
#range over all w values (0-100) calculate accuracy

def M_01(SS_Emfit,SS_Somnofy,SS_PSG):
    size=len(SS_Emfit)
    divide_ind=int(size*.7)
    accuracy_array=np.ones((11,2))
    #accuracy_array[0]=[0,balanced_accuracy_score(SS_PSG[:divide_ind],SS_Somnofy[:divide_ind])]
    #accuracy_array[10]=[100,balanced_accuracy_score(SS_PSG[:divide_ind],SS_Emfit[:divide_ind])]
    for a in [0,10,20,30,40,50,60,70,80,90,100]:
        weights=np.array([a,100-a])
        new_SS=build_new_ss(weights,SS_Emfit,SS_Somnofy)
        accuracy_array[int(a/10)]=[a,balanced_accuracy_score(SS_PSG[:divide_ind],new_SS[:divide_ind])]
    return accuracy_array

def mse_gd(ss_emfit,ss_somnofy,ss_psg, max_iters=150, gamma=0.005):
    y,tx=build_y_tx(ss_emfit,ss_somnofy,ss_psg)
    loss=0
    w = np.zeros((tx.shape[1],), dtype=float) #initial_w
    for n_iter in range(max_iters):
        gradient = comp_grad(tx,comp_error(y,tx,w))
        loss = mse_loss(y, tx, w)
        w = w - gamma * gradient
    return w, loss

#Method 2:
    #applied per individual
#take 80% of the data randomly
#Give two weights for each sleeping stage of emfit and somnofy
#where weights are defined as: [(W0,100-W0),(W1,100-W1),(W2,100-W2),(W3,100-W3)]
#calculate: [emfit*W+somnofy*(100-W)]/100
#round the value to 0-1-2-3
#change values of weights according to the overall accuracy