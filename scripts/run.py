from loader import *
import pandas as pd


if __name__ == '__main__':
    print('This will be run.py')
    data = get_sleepstages(PARTICIPANT_IDS[0])

    pd.set_option('display.max_columns', None)
    print(data.head())
    print(data.isna().sum())


