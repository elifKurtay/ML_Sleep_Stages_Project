import numpy as np
import pandas as pd
from constants import *
from loader import read_patient_data


def test_imputing():
    """ testing different imputing methods on somnofy bc it is found to have most missing values"""
    votes = np.zeros(3)
    diff = np.zeros(3)
    df = pd.DataFrame()

    for subjectID in PARTICIPANT_IDS:
        sleep_stages = read_patient_data(subjectID)
        df["somnofy"] = sleep_stages["sleep_stage_num_somnofy"]
        df["truth"] = sleep_stages["sleep_stage_num_psg"]
        df["Forward Fill"] = df["somnofy"].ffill()
        df["Backward Fill"] = df["somnofy"].bfill()
        df["Interpolate Time"] = df["somnofy"].interpolate(option='time').round()
        diff[0] = np.linalg.norm(df["Interpolate Time"].replace(np.nan, 0) - df["truth"].ffill().replace(np.nan, 0))
        diff[1] = np.linalg.norm(df["Forward Fill"].replace(np.nan, 0) - df["truth"].ffill().replace(np.nan, 0))
        diff[2] = np.linalg.norm(df["Backward Fill"].replace(np.nan, 0) - df["truth"].ffill().replace(np.nan, 0))
        try:
            votes[np.nanargmin(diff)] += 1
        except ValueError:
            print("all nan in participant " + subjectID)
    return votes


def impute_data(subjectID):
    """ impute all columns of patient data
    1. impute using interpolate for middle values
    2. for the beginning and ending values perform both bfill and ffill

    return type: array of size 2
    array[0] = new dataframe with imputed values
    array[1] = current nan values of each column in order
    """
    sleep_stages = read_patient_data(subjectID)
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
