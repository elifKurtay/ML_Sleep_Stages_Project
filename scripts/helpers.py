from constants import *
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta


def augment_data(sleep_stages, divided=False):
    """
    Augment sleep stages data to reach a fixed length
    :param divided:
    :param sleep_stages: processed sleep stages data of a single patient with 2 sensor and 1 psg
    :return: sleep_stages data expended/decreased from the start and end to be on the MEAN_SIZE
    """
    if divided:
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


def impute_data(sleep_stages):
    """ impute all columns of patient data
    1. impute using interpolate for middle values
    2. for the beginning and ending values perform both bfill and ffill

    return type: array of size 2
    array[0] = new dataframe with imputed values
    array[1] = current nan values of each column in order
    """
    for column in sleep_stages.columns:
        if sleep_stages[column].isna().any():
            sleep_stages[column] = sleep_stages[column].interpolate(option='time').round().bfill().ffill()
    nan_count = sleep_stages.isna().sum()
    return sleep_stages, nan_count

def scale_data_bycolumn(rawpoints, high=1.0, low=0.0):
    """scale data by column"""
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)
