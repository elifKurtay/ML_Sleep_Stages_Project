from constants import *


def impute_data(subjectID, sleep_stages):
    """ impute all columns of patient data
    1. impute using interpolate for middle values
    2. for the beginning and ending values perform both bfill and ffill

    return type: array of size 2
    array[0] = new dataframe with imputed values
    array[1] = current nan values of each column in order
    """
    sleep_stages["sleep_stage_num_somnofy"] = sleep_stages["sleep_stage_num_somnofy"]\
        .interpolate(option='time').round().bfill().ffill()
    sleep_stages["sleep_stage_num_emfit"] = sleep_stages["sleep_stage_num_emfit"]\
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
