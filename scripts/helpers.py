from constants import *
import pandas as pd
from dateutil.relativedelta import relativedelta


def augment_data(sleep_stages):
    """
    Augment sleep stages data to reach a fixed length
    :param sleep_stages: processed sleep stages data of a single patient with 2 sensor and 1 psg
    :return: sleep_stages data expended/decreased from the start and end to be on the MEAN_SIZE
    """
    augment = MEAN_SIZE > sleep_stages.shape[0]
    ind_diff = abs(MEAN_SIZE - sleep_stages.shape[0]) // 2
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


def impute_data(subjectID, sleep_stages):
    """ impute all columns of patient data
    1. impute using interpolate for middle values
    2. for the beginning and ending values perform both bfill and ffill

    return type: array of size 2
    array[0] = new dataframe with imputed values
    array[1] = current nan values of each column in order
    """
    sleep_stages["sleep_stage_num_somnofy"] = sleep_stages["sleep_stage_num_somnofy"] \
        .interpolate(option='time').round().bfill().ffill()
    sleep_stages["sleep_stage_num_emfit"] = sleep_stages["sleep_stage_num_emfit"] \
        .interpolate(option='time').round().bfill().ffill()
    sleep_stages["sleep_stage_num_psg"] = sleep_stages["sleep_stage_num_psg"] \
        .interpolate(option='time').round().bfill().ffill()
    nan_count = sleep_stages.isna().sum()
    if nan_count[2] == sleep_stages.shape[0]:
        print("NO PSG FOR PATIENT " + subjectID)
    return sleep_stages, (nan_count[0], nan_count[1], nan_count[2])


def impute_all():
    """ impute all columns of all patient data
    1. impute using interpolate for middle values
    2. for the beginning and ending values perform both bfill and ffill

    return type: array of size 2
    array[0] = new dataframe with imputed values
    array[1] = current nan values of each column in order
    """
    dfs = []
    for subjectID in PARTICIPANT_IDS:
        sleep_stages, nan_count = impute_data(subjectID)
        dfs.append([sleep_stages, (nan_count[0], nan_count[1], nan_count[2])])
    return dfs
