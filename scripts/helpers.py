from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score
import warnings
from constants import *
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from dateutil.relativedelta import relativedelta


#------------------------ PREPROCESSING ------------------------------------------------------------
def augment_data(sleep_stages, divided=False):
    """
    Augment sleep stages data to reach a fixed length
    :param divided:
    :param sleep_stages: processed sleep stages data of a single patient with 2 sensor and 1 psg
    :return: sleep_stages data expended/decreased from the start and end to be on the MEAN_SIZE
    """
    if not divided:
        augment = MEAN_SIZE > sleep_stages.shape[0]
        mean = MEAN_SIZE
    elif MEAN_SIZE > sleep_stages.shape[0]:
        augment = SMALL_MEAN > sleep_stages.shape[0]
        mean = SMALL_MEAN
    else:
        augment = BIG_MEAN > sleep_stages.shape[0]
        mean = BIG_MEAN

    ind_diff = abs(mean - sleep_stages.shape[0]) // 2
    interval = relativedelta(seconds=30)
    if augment:
        start_date = pd.to_datetime(sleep_stages.index[0]) - interval * ind_diff
        end_date = pd.to_datetime(sleep_stages.index[-1]) + interval * (ind_diff + int(sleep_stages.shape[0] % 2 == 0))
    else:
        start_date = pd.to_datetime(sleep_stages.index[0]) + interval * ind_diff
        end_date = pd.to_datetime(sleep_stages.index[-1]) - interval * (ind_diff + int(sleep_stages.shape[0] % 2 == 0))

    # print("OLD ", pd.to_datetime(sleep_stages.index[0]), pd.to_datetime(sleep_stages.index[-1]))
    # print("NEW ", start_date, end_date)
    expended_df = pd.DataFrame(index=pd.date_range(start_date, end_date, freq="30s")
                               .strftime("%Y-%m-%d %H:%M:%S+00:00"))
    merged = pd.merge(expended_df, sleep_stages, left_index=True, right_index=True, how='left')

    if augment:
        merged.iloc[0] = 0.
        merged.iloc[-1] = 0.
        merged = merged.interpolate(option='time').round()
    # merged.plot()
    return merged


def scale_data_bycolumn( rawpoints, high=1.0, low=0.0):
    scaler = MinMaxScaler()
    # transform data
    return scaler.fit_transform(rawpoints)

#------------------------ RESULT ------------------------------------------------------------
def overall_balanced_accuracy(x, y):
    """ computes the overall balanced accuracy"""
    warnings.warn("deprecated", DeprecationWarning)
    radar_score = balanced_accuracy_score(y, x[:, 0])
    mat_score = balanced_accuracy_score(y, x[:, 1])
    print("Radar:   Acc = ", radar_score)
    print("Mat:   Acc = ", mat_score)


#------------------------ PLOTS ------------------------------------------------------------
def plot_missing_sleepstages(df):
    color_map = [("sleep_stage_num_somnofy", "blue"), ("sleep_stage_num_emfit", "orange"), ("sleep_stage_num_psg", "green")]
    truth_vals = df["sleep_stage_num_psg"].to_numpy()
    for ind, (title, color) in enumerate(color_map):
        missing_ind = np.array(np.where(df[title].isnull())[0])
        y = np.ones(len(missing_ind))
        if ind == 1:
            y[:] = 2
        elif ind == 2:
            y[:] = 3
        else:
            print("Somnofy is missing where psg has the following values:")
            unique_elements, counts_elements = np.unique(truth_vals[missing_ind], return_counts=True)
            print(np.asarray((unique_elements, counts_elements)))
        plt.scatter(missing_ind, y, color=color)
    plt.show()


def plot_conf_matrix(preds, labels, normalize = False):
    if normalize:
        conf_matrix = confusion_matrix(labels, preds, normalize = 'all')
    else:
        conf_matrix = confusion_matrix(labels, preds)
    acc_per_class = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    print("Accuracy per class:", acc_per_class)
    df_cm = pd.DataFrame(conf_matrix, range(4), range(4))
    ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}, fmt=".3g")
    ax.set(xlabel='Actual', ylabel='Prediction')
    plt.show()

#------------------------ PREDICTION ------------------------------------------------------------
def get_predictions(weights,emfits,somnofys):
    new_ss=np.ones((emfits.shape[0],emfits.shape[1]))
    if len(weights) == 2:
        for i in range(new_ss.shape[1]):
            new_ss[:,i]=(emfits[:,i]*weights[0]+somnofys[:,i]*weights[1])
    elif len(weights) == 3:
        for i in range(new_ss.shape[1]):
                new_ss[:,i]=(emfits[:,i]*weights[0]+somnofys[:,i]*weights[1])+weights[0]
    new_ss=np.round(new_ss)
    return new_ss


def mse_gd(ss_emfit,ss_somnofy,ss_psg, max_iters=150, gamma=0.005,w0=False):
    y,tx=build_y_tx(ss_emfit,ss_somnofy,ss_psg,w0)
    loss=0
    w = np.zeros((tx.shape[1],1), dtype=float)  # initial_w
    for n_iter in range(max_iters):
        gradient = comp_grad(tx,comp_error(y,tx,w))
        loss = mse_loss(y, tx, w)
        w = w - gamma * gradient
    return w, loss


def build_y_tx(ss_emfit, ss_somnofy, ss_psg, w0):
    y = ss_psg.reshape((1, len(ss_psg)))
    y = y.T
    if w0:
        tx = np.c_[np.ones(len(ss_psg)), ss_emfit, ss_somnofy]
    else:
        tx = np.c_[ss_emfit, ss_somnofy]
    return y, tx


def comp_error(y, tx, w):
    return y - tx @ w


def mse_loss(y, tx, w):
    e = comp_error(y, tx, w)
    return 1 / (2 * (np.shape(tx)[0])) * e.T @ e


def comp_grad(tx, error):
    return -1 / (np.shape(tx)[0]) * tx.T @ error
