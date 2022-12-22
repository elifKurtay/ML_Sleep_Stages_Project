from helpers import *

import pandas as pd
import os
import datetime

path_ = get_read_path()


def read_patient_data(subjectID, raw=False):
    """read patient sleep stages from data/processed"""
    dir_path = get_write_path(raw=raw)
    path = dir_path + "/SMS_" + subjectID + ".csv"
    if os.path.exists(path):
        data = pd.read_csv(path, index_col='timestamp_local')
        return data
    return None


def write_data(raw=False):
    """creates all patient sleep stages by sensor
    and writes in data/processed"""
    path = get_write_path(raw=raw)
    for subjectID in PARTICIPANT_IDS:
        data = get_sleepstages(subjectID, raw=raw)
        if data is None:
            return print(path + "/SMS_" + subjectID + ".csv CANNOT BE GENERATED")
        imputed, _ = impute_data(data)
        imputed.to_csv(path + "/SMS_" + subjectID + ".csv", index=True)
    return True


def get_nn_patients(raw=False, divided=False):
    """
    creates all patient input for neural networks in fixed size
    :return: 3 items:
        - radars: numpy of all patients radar values in size MEAN_SIZE
        - mats: numpy of all patients emfit mat values in size MEAN_SIZE
        - patients: numpy of all patients augmented values in size MEAN_SIZE
    """
    x_small = []
    x_big = []
    y_small = []
    y_big = []
    for subjectId in PARTICIPANT_IDS:
        sleep_stages = read_patient_data(subjectId, raw=raw)
        augmented = augment_data(sleep_stages, divided=divided)

        if divided and raw and augmented.shape[0] > MEAN_SIZE:
            x_big.append(
                scale_data_bycolumn((augmented.drop("sleep_stage_num_psg", axis=1).to_numpy()), high=1.0, low=0.0))
        elif divided and augmented.shape[0] > MEAN_SIZE:
            x_big.append(augmented.drop("sleep_stage_num_psg", axis=1).to_numpy())
        elif raw:
            standartised = scale_data_bycolumn((augmented.drop("sleep_stage_num_psg", axis=1).to_numpy()), high=1.0, low=0.0)
            x_small.append(standartised)
        else:
            x_small.append(augmented.drop("sleep_stage_num_psg", axis=1).to_numpy())

        if divided and augmented.shape[0] > MEAN_SIZE:
            y_big.append(augmented["sleep_stage_num_psg"].to_numpy())
        else:
            y_small.append(augmented["sleep_stage_num_psg"].to_numpy())
    return np.array(x_small), np.array(y_small), np.array(x_big), np.array(y_big)


def get_sleepstages(subjectID, inner=True, raw=False):
    radar = get_sleepstages_radar(subjectID, path_, raw=raw)
    psg = get_sleep_class_psg(subjectID, path_)
    mat = get_EMFIT_sleep_stages_file(subjectID, raw=raw)
    if radar[0] and psg[0] and mat[0]:
        if inner:
            merged = pd.merge(radar[1], mat[1], left_index=True, right_index=True)
            data = pd.merge(merged, psg[1], left_index=True, right_index=True)
        else:
            data = radar[1].join(mat[1]).join(psg[1])
            print(data.shape[0])
        return data
    else:
        print("PROBLEM WITH EXTRACTING A DATA")


def get_EMFIT_sleep_stages_file(subjectID, emfitID="001505", _path=path_, raw=False, shift="0s"):
    """Returns a Tuple:
        [0]:boolean --> True, if request was sucessefull
        [1]: DataFrame with LOCAL timestamps as Index and the rest as Columne entries
    """
    path_to_subject_root = f'{_path}/SMS_{subjectID}'
    path_to_subject_psg = f'{path_to_subject_root}/EMFIT_{emfitID}'

    if os.path.exists(path_to_subject_psg):

        file_list = os.listdir(path_to_subject_psg)
        sleep_file = f'SMS_{subjectID}-psg-EMFIT_{emfitID}-processed_sleepclasses.csv'
        raw_file = f'SMS_{subjectID}-psg-EMFIT_{emfitID}-processed.csv'

        if sleep_file in file_list and raw_file in file_list:

            data_EMFIT = pd.read_csv(f'{path_to_subject_psg}/{sleep_file}', index_col=0)

            data_EMFIT = data_EMFIT.rename(columns={'timestamp': 'timestamp_local'})
            data_EMFIT.index = pd.to_datetime(data_EMFIT.timestamp_local, unit="s")
            data_EMFIT = data_EMFIT.tz_localize("UTC")
            data_EMFIT = data_EMFIT.tz_convert("Europe/Zurich")

            EMFIT_SLEEPCLASS_MAP = {
                1: 3,
                2: 2,
                3: 1,
                4: 0
            }
            data_EMFIT["sleep_stage_num_emfit"] = data_EMFIT.sleep_class.map(EMFIT_SLEEPCLASS_MAP)
            data_EMFIT_resampled = data_EMFIT.resample("30s").median().ffill()
            data_EMFIT_resampled = data_EMFIT_resampled.drop(["sleep_class", "timestamp_local"], axis=1)

            if raw:
                raw_EMFIT = pd.read_csv(f'{path_to_subject_psg}/{raw_file}', index_col=0)
                raw_EMFIT = raw_EMFIT.rename(columns={'timestamp': 'timestamp_local'})
                raw_EMFIT.index = pd.to_datetime(raw_EMFIT.timestamp_local, unit="s")
                raw_EMFIT = raw_EMFIT.tz_localize("UTC")
                raw_EMFIT = raw_EMFIT.tz_convert("Europe/Zurich")
                raw_EMFIT = raw_EMFIT.resample("30s").median().ffill()
                raw_EMFIT.drop(["timestamp_local"], axis=1, inplace=True)
                emfit = pd.merge(data_EMFIT_resampled, raw_EMFIT, left_index=True, right_index=True)
                return True, emfit[["sleep_stage_num_emfit", "hr", "rr", "act"]]

            return True, data_EMFIT_resampled["sleep_stage_num_emfit"]

        else:
            print(f'ERROR: no sleep_stages file in folder {path_to_subject_psg}')
            return False,
    else:
        print(f'No EMFIT_{emfitID} Data for Participant: {subjectID}')
        return False,


def get_somnofy_data(subjectID, _path, shift="0s"):
    """Returns a Tuple:
        [0]:boolean --> True, if request was sucessefull
        [1]: DataFrame with LOCAL timestamps as Index and the rest as Columne entries
    """
    path_to_subject_root = f'{_path}/SMS_{subjectID}'
    path_to_subject_psg = f'{path_to_subject_root}/Somnofy'

    if os.path.exists(path_to_subject_psg):
        file_list = os.listdir(path_to_subject_psg)
        if len(file_list) != 1:
            print(f'ERROR: more than 1 or no file in folder {path_to_subject_psg}')
            return (False,)
        else:
            data_somnofy = pd.read_csv(f'{path_to_subject_psg}/{file_list[0]}')

            data_somnofy["sleep_stage"].replace(
                {1: "3_Stage_Deep", 2: "2_Stage_Light", 3: "1_Stage_REM", 4: "0_Stage_Wake", 5: np.NaN}, inplace=True)
            data_somnofy["sleep_stage_num_somnofy"] = data_somnofy["sleep_stage"]
            data_somnofy["sleep_stage_num_somnofy"].replace(
                {"3_Stage_Deep": 3, "2_Stage_Light": 2, "1_Stage_REM": 1, "0_Stage_Wake": 0}, inplace=True)

            data_somnofy['timestamp_local'] = pd.to_datetime(data_somnofy['timestamp_local']).dt.tz_localize(
                'Europe/Paris')

            data_somnofy['timestamp_local'] = data_somnofy['timestamp_local'] + pd.Timedelta('30s')

            data_somnofy.set_index('timestamp_local', inplace=True)
            data_somnofy_resampled = data_somnofy.resample('30s').median().drop("Unnamed: 0", axis=1)
            data_somnofy_resampled["sleep_stage_num_somnofy"] = data_somnofy_resampled["sleep_stage_num_somnofy"].round(
                decimals=0)
        return (True, data_somnofy_resampled)
    else:
        print(f'No Somnofy Data for Participant: {subjectID}')
        return (False,)


def get_sleepstages_radar(subjectID, _path, raw=False):
    """    Function to get Sleepstages from Somnofy Report. the stages are coded the following way
    (the first column is the value now)
    0 = Awake = 4
    1 = REM   = 3
    2 = Light = 2
    3 = Deep  = 1
    nan= ...  = 5 <-- No Sleep Stage classified, due to (movement) artefacts"""
    _d = get_somnofy_data(subjectID, _path)
    if _d[0] and not raw:
        return True, _d[1][['sleep_stage_num_somnofy']]
    elif _d[0] and raw:
        return True, _d[1][['sleep_stage_num_somnofy', "distance_mean", "movement_mean", "respiration_rate_mean",
                            "signal_quality_mean"]]
    else:
        print("Sleep stages could no be extracted")
        return False,


def get_sleep_class_psg(subjectID, path_):
    """gets truth values for sleep stage of a subject
        either from RemLogic or Somnomedics in 30s intervals as index
        Returns a Tuple:
        [0]:boolean --> True, if request was successful
        [1]: DataFrame with LOCAL timestamps as Index and Sleepstages as Column entries ("sleep_stage_num_psg")
    """
    path_to_file = os.path.join(path_, "SMS_" + subjectID, "somnomedics")
    if os.path.exists(path_to_file):
        #  read Somnomedics
        psg_data = get_sleepstages_psg_somnomedics(subjectID, path_)
    else:
        #  read RemLogic
        psg_data = get_sleepstages_psg_remlogic(subjectID, path_)
    return psg_data


# REMLOGIC FUNCTIONS
def get_remlogic_tsv_file(participant_id: str, file_type: str, _path: str):
    """Valid entries for file_type is "scans" or "events"
    example: get_psg_tsv_file("002", "scans")"""
    try:
        file = pd.read_csv(os.path.join(_path, "SMS_" + participant_id, "RemLogic", "sub_%s.tsv" % file_type),
                           sep="\t")
        return file
    except Exception as e:
        print("No file %s" % str(e))


def get_sleepstages_psg_remlogic(subjectID, _path):
    """ Function to get Sleepstages from PSG.
    Returns a Tuple:
        [0]:boolean --> True, if request was sucessefull
        [1]: DataFrame with LOCAL timestamps as Index and Sleepstages as Column entries
    The stages are coded the following way:
    0 = Awake = SLEEP-S0
    1 = REM   = SLEEP-REM
    2 = Light = SLEEP-S1 and  SLEEP-S2
    3 = Deep  = SLEEP-S3"""
    acq_time = get_remlogic_tsv_file(subjectID, "scans", _path)
    if acq_time is None:
        return (False,)
    acq_time = pd.to_datetime(acq_time.columns[1])

    event_data = get_remlogic_tsv_file(subjectID, "events", _path)
    sleep_data = event_data[(event_data['trial_type'] == "SLEEP-S0") |
                            (event_data['trial_type'] == "SLEEP-S1") |
                            (event_data['trial_type'] == "SLEEP-S2") |
                            (event_data['trial_type'] == "SLEEP-S3") |
                            (event_data['trial_type'] == "SLEEP-REM")].reset_index().drop("index", axis=1)

    sleep_data["time"] = pd.Series(acq_time).repeat(len(sleep_data)).values + pd.to_timedelta(sleep_data.onset,
                                                                                              unit='s')

    sleep_data['trial_type'].replace({"SLEEP-S0": "0_Stage_Wake",
                                      "SLEEP-S1": "2_Stage_Light",
                                      "SLEEP-S2": "2_Stage_Light",
                                      "SLEEP-S3": "3_Stage_Deep",
                                      "SLEEP-REM": "1_Stage_REM"}, inplace=True)

    sleep_data["sleep_stage_num_psg"] = sleep_data["trial_type"]
    sleep_data["sleep_stage_num_psg"].replace(
        {"3_Stage_Deep": 3., "2_Stage_Light": 2., "1_Stage_REM": 1., "0_Stage_Wake": 0.}, inplace=True)
    sleep_data['timestamp_local'] = pd.to_datetime(sleep_data['time']).dt.tz_localize('Europe/Paris')
    sleep_data.set_index('timestamp_local', inplace=True)
    return (True, sleep_data[['sleep_stage_num_psg']])


# Somnomedics FUNCTIONS
def add_date(data, start_date):
    if "\n" in start_date:
        start_date = start_date[:10]

    if len(start_date) == 9:
        second_day = datetime.datetime.strptime(start_date, "%d-%b-%y").date()
    elif len(start_date) == 11:
        second_day = datetime.datetime.strptime(start_date, "%d-%b-%Y").date()
    else:
        second_day = datetime.datetime.strptime(start_date, "%d.%m.%Y").date()

    second_day += datetime.timedelta(days=1)
    second_day = second_day.strftime("%d.%m.%Y")
    # print(start_date + " - " + second_day)

    current_day = start_date
    for i in range(len(data)):
        if data[i][0] == "00:00:00,000":
            current_day = second_day
        data[i][0] = current_day + " " + data[i][0]
    return data


def get_sleepstages_psg_somnomedics(subjectID, _path):
    """    Function to get Sleepstages from Somnomedics Report. the stages are coded the following way
    (the first column is the value now)
    0 = Awake = 4
    1 = REM   = 3
    2 = Light = 2
    3 = Deep  = 1
    nan= ...  = 5 <-- No Sleep Stage classified, due to (movement) artefacts"""
    path_to_english_file = os.path.join(_path, "SMS_" + subjectID, "somnomedics", "Sleep profile.txt")
    path_to_german_file = os.path.join(_path, "SMS_" + subjectID, "somnomedics", "Schlafprofil.txt")
    if os.path.exists(path_to_english_file):
        path_to_file = path_to_english_file
    elif os.path.exists(path_to_german_file):
        path_to_file = path_to_german_file
    else:
        print(f'No Somnomedics Data for Participant: {subjectID}')
        return (False,)

    with open(path_to_file) as f:
        lines = f.readlines()
    if len(lines) <= 7:  # intro lines
        print(f'No Somnomedics Data for Participant: {subjectID}')
        return (False,)
    data = lines[7:]
    data = [a[:-1].split("; ") for a in data]
    # checking for date in datetime
    if len(data[0][0]) < 23:
        data = add_date(data, lines[1].split(" ")[2])
    data_somnomedics = pd.DataFrame(data, columns=['timestamp_local', 'sleep_stage'])
    data_somnomedics["sleep_stage"].replace({"N3": "3_Stage_Deep", "N2": "2_Stage_Light", "N1": "2_Stage_Light",
                                             "REM": "1_Stage_REM", "Rem": "1_Stage_REM", "Wake": "0_Stage_Wake",
                                             "Artefact": np.NaN, "A": np.NaN, "Artefakt": np.NaN,
                                             "Wach": "0_Stage_Wake"}, inplace=True)
    data_somnomedics["sleep_stage_num_psg"] = data_somnomedics["sleep_stage"]
    data_somnomedics["sleep_stage_num_psg"].replace({"3_Stage_Deep": 3., "2_Stage_Light": 2., "1_Stage_REM": 1.,
                                                     "0_Stage_Wake": 0.}, inplace=True)
    try:
        data_somnomedics['timestamp_local'] = pd.to_datetime(data_somnomedics['timestamp_local'],
                                                             format="%d.%m.%Y %H:%M:%S,%f").dt.tz_localize(
            'Europe/Paris')
    except ValueError:
        print(f'Date format wrong for Participant: {subjectID}')
        data_somnomedics['timestamp_local'] = pd.to_datetime(data_somnomedics['timestamp_local']).dt.tz_localize(
            'Europe/Paris')

    data_somnomedics['timestamp_local'] = data_somnomedics['timestamp_local'] + pd.Timedelta('30s')

    data_somnomedics.set_index('timestamp_local', inplace=True)
    del data_somnomedics['sleep_stage']
    return (True, data_somnomedics)


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
