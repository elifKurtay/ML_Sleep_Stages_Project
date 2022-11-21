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