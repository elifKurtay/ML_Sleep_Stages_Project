from loader import *
import pandas as pd


if __name__ == '__main__':
    print('This will be run.py')
    path = get_write_path()
    for subjectID in PARTICIPANT_IDS:
        data = get_sleepstages(subjectID)
        if data is not None:
            data.to_csv(path + "/SMS_" + subjectID + ".csv", index=True)

    pd.set_option('display.max_columns', None)
    print(data.head())
    print(data.isna().sum())


