import numpy as np

def value_rounded(value,Shifted):
    if Shifted:
        if value <1.5:
            return 0
        elif 1.5 <= value <2.5:
            return 1
        elif 2.5<=value<3.5:
            return 2
        elif 3.5<=value:
            return 3
    else:
        if value <0.5:
            return 0
        elif 0.5 <= value <1.5:
            return 1
        elif 1.5<=value<2.5:
            return 2
        elif 2.5<=value:
            return 3

def build_new_ss(weights,ss_emfit,ss_somnofy,Shifted=False):
    new_SS=np.ones(len(ss_emfit))
    if len(weights) == 2:
        for i in range(len(ss_emfit)):
            new_SS[i]=value_rounded((ss_emfit[i]*weights[0]+ss_somnofy[i]*weights[1]),Shifted)
    elif len(weights) == 3:
        for i in range(len(ss_emfit)):
                new_SS[i]=value_rounded(weights[0]+ss_emfit[i]*weights[1]+ss_somnofy[i]*weights[2],Shifted)         
    return new_SS

def build_new_ss_methodx(weights,ss_emfit,ss_somnofy):
    new_SS=np.ones(len(ss_emfit))
    for i in range(len(ss_emfit)):
        if ss_emfit[i]==1:
            if ss_somnofy[i]==1:
                new_SS[i]=ss_emfit[i]-1
            elif ss_somnofy[i]==2:
                new_SS[i]=value_rounded((ss_emfit[i]*weights[0][0]+ss_somnofy[i]*weights[0][1]),True)
            elif ss_somnofy[i]==3:
                new_SS[i]=value_rounded((ss_emfit[i]*weights[1][0]+ss_somnofy[i]*weights[1][1]),True)
            elif ss_somnofy[i]==4:
                new_SS[i]=value_rounded((ss_emfit[i]*weights[2][0]+ss_somnofy[i]*weights[2][1]),True)
        elif ss_emfit[i]==2:
            if ss_somnofy[i]==1:
                new_SS[i]=value_rounded((ss_emfit[i]*weights[3][0]+ss_somnofy[i]*weights[3][1]),True)
            elif ss_somnofy[i]==2:
                new_SS[i]=ss_emfit[i]-1
            elif ss_somnofy[i]==3:
                new_SS[i]=value_rounded((ss_emfit[i]*weights[4][0]+ss_somnofy[i]*weights[4][1]),True)
            elif ss_somnofy[i]==4:
                new_SS[i]=value_rounded((ss_emfit[i]*weights[5][0]+ss_somnofy[i]*weights[5][1]),True)
        elif ss_emfit[i]==3:
            if ss_somnofy[i]==1:
                new_SS[i]=value_rounded((ss_emfit[i]*weights[6][0]+ss_somnofy[i]*weights[6][1]),True)
            elif ss_somnofy[i]==2:
                new_SS[i]=value_rounded((ss_emfit[i]*weights[7][0]+ss_somnofy[i]*weights[7][1]),True)
            elif ss_somnofy[i]==3:
                new_SS[i]=ss_emfit[i]-1
            elif ss_somnofy[i]==4:
                new_SS[i]=value_rounded((ss_emfit[i]*weights[8][0]+ss_somnofy[i]*weights[8][1]),True)
        elif ss_emfit[i]==4:
            if ss_somnofy[i]==1:
                new_SS[i]=value_rounded((ss_emfit[i]*weights[9][0]+ss_somnofy[i]*weights[9][1]),True)
            elif ss_somnofy[i]==2:
                new_SS[i]=value_rounded((ss_emfit[i]*weights[10][0]+ss_somnofy[i]*weights[10][1]),True)
            elif ss_somnofy[i]==3:
                new_SS[i]=value_rounded((ss_emfit[i]*weights[11][0]+ss_somnofy[i]*weights[10][1]),True)
            elif ss_somnofy[i]==4:
                new_SS[i]=ss_emfit[i]-1
    return new_SS


def build_y_tx(ss_emfit,ss_somnofy,ss_psg,w0):
    y=ss_psg.reshape((1,len(ss_psg)))
    y=y.T
    if w0:
        tx=np.c_[np.ones(len(ss_psg)),ss_emfit,ss_somnofy]
    else:
        tx=np.c_[ss_emfit,ss_somnofy]
    return y,tx

def comp_error(y,tx,w):
    return y-tx@w

def mse_loss(y, tx, w):
    e=comp_error(y,tx,w)
    return 1/(2*(np.shape(tx)[0]))*e.T@e

def comp_grad(tx,error):
    return -1/(np.shape(tx)[0])*tx.T@error

def just_loss(ss_smthng,ss_psg):
    y=ss_psg.reshape((len(ss_psg),1))
    x=ss_smthng.reshape((len(ss_psg),1))
    e=y-x
    loss=1/(2*(len(ss_psg)))*np.transpose(e)@e
    return loss


    


