import numpy as np

def value_rounded(value):
    if 0 <= value <0.5:
        return 0
    elif 0.5 <= value <1.5:
        return 1
    elif 1.5<=value<2.5:
        return 2
    elif 2.5<=value<=3:
        return 3

def build_new_ss(weights,ss_emfit,ss_somnofy):
    new_SS=np.ones(len(ss_emfit))
    if len(weights) == 2:
        #weights not specified for each sleep stage
        for i in range(len(ss_emfit)):
            new_SS[i]=value_rounded((ss_emfit[i]*weights[0]+ss_somnofy[i]*weights[1])/100)
    elif len(weights) == 4:
        #4 weight arrays given, all including 2 values - total of 8 weights
        for i in range(len(ss_emfit)):
            new_SS[i]=value_rounded((ss_emfit[i]*weights[ss_emfit[i]][0]+ss_somnofy[i]*weights[ss_somnofy[i]][1])/100)
    return new_SS

def build_y_tx(ss_emfit,ss_somnofy,ss_psg):
    print(ss_psg.shape)
    y=ss_psg.reshape((1,len(ss_psg)))
    print(y.shape)
    y=y.T
    print(y.shape)
    tx=np.c_(ss_emfit,ss_somnofy)
    return y,tx

def comp_error(y,tx,w):
    return y-tx@w
    
def mse_loss(y, tx, w):
    e=comp_error(y,tx,w)
    return 1/(2*(np.shape(tx)[0]))*e.T*e

def comp_grad(tx,error):
    return -1/(np.shape(tx)[0])*tx.T*error



    


