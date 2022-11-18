from constants import *
import pandas as pd
import os
import numpy as np

path_ = get_read_path()


def write_data():
    """074 and 058 has problem with Somnomedics reading"""
    path = get_write_path()
    for subjectID in PARTICIPANT_IDS:
        data = get_sleepstages(subjectID)
        if data is not None:
            data.to_csv(path + "/SMS_" + subjectID + ".csv", index=True)
    return True


def get_sleepstages(subjectID):
    radar = get_sleepstages_radar(subjectID, path_)
    psg = get_sleep_class_psg(subjectID)
    mat = get_EMFIT_sleep_stages_file(subjectID, "001505")
    if radar[0] and psg[0] and mat[0]:
        data = radar[1].join(mat[1]).join(psg[1])
        return data
    else:
        print("PROBLEM WITH EXTRACTING A DATA")


def get_EMFIT_sleep_stages_file(subjectID, emfitID, _path=path_, shift="0s"):
    """Returns a Tuple:
        [0]:boolean --> True, if request was sucessefull
        [1]: DataFrame with LOCAL timestamps as Index and the rest as Columne entries
    """
    path_to_subject_root = f'{_path}/SMS_{subjectID}'
    path_to_subject_psg = f'{path_to_subject_root}/EMFIT_{emfitID}'

    if os.path.exists(path_to_subject_psg):

        file_list = os.listdir(path_to_subject_psg)
        file = f'SMS_{subjectID}-psg-EMFIT_{emfitID}-processed_sleepclasses.csv'

        if file in file_list:

            data_EMFIT = pd.read_csv(f'{path_to_subject_psg}/{file}')
            data_EMFIT = data_EMFIT.rename(columns={'timestamp': 'timestamp_local'})
            data_EMFIT.index = pd.to_datetime(data_EMFIT.timestamp_local, unit="s")
            data_EMFIT = data_EMFIT.tz_localize("UTC")
            data_EMFIT = data_EMFIT.tz_convert("Europe/Zurich")

            EMFIT_SLEEPCLASS_MAP = {
                1: 0,
                2: 1,
                3: 2,
                4: 3
            }

            data_EMFIT["sleep_stage_num_emfit"] = data_EMFIT.sleep_class.map(EMFIT_SLEEPCLASS_MAP)
            data_EMFIT_resampled = data_EMFIT.resample("30s").median(numeric_only=False).ffill()  # should be an int

            return (True, data_EMFIT_resampled["sleep_stage_num_emfit"])

        else:
            print(f'ERROR: no sleep_stages file in folder {path_to_subject_psg}')
            return (False,)

    else:
        print(f'No EMFIT_{emfitID} Data for Participant: {subjectID}')

        return (False,)


def get_somnofy_data(subjectID, _path, shift = "0s"):
    """Returns a Tuple:
        [0]:boolean --> True, if request was sucessefull
        [1]: DataFrame with LOCAL timestamps as Index and the rest as Columne entries
    """
    path_to_subject_root= f'{_path}/SMS_{subjectID}'
    path_to_subject_psg= f'{path_to_subject_root}/Somnofy'

    if os.path.exists(path_to_subject_psg):
        file_list= os.listdir(path_to_subject_psg)
        if len(file_list) != 1:
            print(f'ERROR: more than 1 or no file in folder {path_to_subject_psg}')
            return (False,)
        else:
            data_somnofy =  pd.read_csv(f'{path_to_subject_psg}/{file_list[0]}')

            data_somnofy["sleep_stage"].replace({1:"3_Stage_Deep", 2:"2_Stage_Light", 3:"1_Stage_REM", 4: "0_Stage_Wake", 5: np.NaN}, inplace=True)
            data_somnofy["sleep_stage_num_somnofy"]= data_somnofy["sleep_stage"]
            data_somnofy["sleep_stage_num_somnofy"].replace({"3_Stage_Deep":3, "2_Stage_Light":2, "1_Stage_REM":1, "0_Stage_Wake":0}, inplace=True)

            data_somnofy['timestamp_local'] =pd.to_datetime(data_somnofy['timestamp_local']).dt.tz_localize('Europe/Paris')

            data_somnofy['timestamp_local']=data_somnofy['timestamp_local']+ pd.Timedelta('30s')

            data_somnofy.set_index('timestamp_local', inplace = True)
            data_somnofy_resampled = data_somnofy.resample('30s').median(numeric_only=False).drop("Unnamed: 0",axis= 1)
            data_somnofy_resampled["sleep_stage_num_somnofy"] = data_somnofy_resampled["sleep_stage_num_somnofy"].round(decimals=0)
        return (True,data_somnofy_resampled)
    else:
        print(f'No Somnofy Data for Participant: {subjectID}')
        return (False,)


def get_sleepstages_radar(subjectID, _path):
    """    Function to get Sleepstages from Somnofy Report. the stages are coded the following way
    (the first column is the value now)
    0 = Awake = 4
    1 = REM   = 3
    2 = Light = 2
    3 = Deep  = 1
    nan= ...  = 5 <-- No Sleep Stage classified, due to (movement) artefacts"""
    _d = get_somnofy_data(subjectID, _path)
    if _d[0]:
        return (True, _d[1][['sleep_stage_num_somnofy']])
    else:
        print("Sleep stages could no be extracted")
        return (False,)


def get_sleep_class_psg(subjectID):
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
    acq_time = get_remlogic_tsv_file(subjectID, "scans",_path)
    if acq_time is None:
        return (False, )
    acq_time = pd.to_datetime(acq_time.columns[1])

    event_data = get_remlogic_tsv_file(subjectID, "events",_path)
    sleep_data= event_data[(event_data['trial_type']=="SLEEP-S0")|
                      (event_data['trial_type']=="SLEEP-S1")|
                      (event_data['trial_type']=="SLEEP-S2")|
                      (event_data['trial_type']=="SLEEP-S3")|
                      (event_data['trial_type']=="SLEEP-REM")].reset_index().drop("index",axis=1)

    sleep_data["time"]= pd.Series(acq_time).repeat(len(sleep_data)).values + pd.to_timedelta(sleep_data.onset, unit='s')

    sleep_data['trial_type'].replace({"SLEEP-S0":"0_Stage_Wake",
             "SLEEP-S1":"2_Stage_Light",
             "SLEEP-S2":"2_Stage_Light",
             "SLEEP-S3":"3_Stage_Deep",
             "SLEEP-REM":"1_Stage_REM"}, inplace=True)

    sleep_data["sleep_stage_num_psg"]= sleep_data["trial_type"]
    sleep_data["sleep_stage_num_psg"].replace({"3_Stage_Deep":3, "2_Stage_Light":2, "1_Stage_REM":1, "0_Stage_Wake":0}, inplace=True)
    sleep_data['timestamp_local'] =pd.to_datetime(sleep_data['time']).dt.tz_localize('Europe/Paris')
    sleep_data.set_index('timestamp_local', inplace= True)
    return (True, sleep_data[['sleep_stage_num_psg']])


# Somnomedics FUNCTIONS
def get_sleepstages_psg_somnomedics(subjectID, _path):
    """    Function to get Sleepstages from Somnomedics Report. the stages are coded the following way
    (the first column is the value now)
    0 = Awake = 4
    1 = REM   = 3
    2 = Light = 2
    3 = Deep  = 1
    nan= ...  = 5 <-- No Sleep Stage classified, due to (movement) artefacts"""
    path_to_file = os.path.join(_path, "SMS_" + subjectID, "somnomedics", "Sleep profile.txt")
    if os.path.exists(path_to_file):
        with open(path_to_file) as f:
            lines = f.readlines()
        if len(lines) <= 7:  # intro lines
            print(f'No Somnomedics Data for Participant: {subjectID}')
            return (False,)
        data = lines[7:]
        data = [a[:-1].split(";") for a in data]
        data_somnomedics = pd.DataFrame(data, columns = ['timestamp_local', 'sleep_stage'])
        data_somnomedics["sleep_stage"].replace({" N3":"3_Stage_Deep", " N2":"2_Stage_Light", " N1":"2_Stage_Light",
                                                 " REM":"1_Stage_REM", " Wake": "0_Stage_Wake", " Artefact": np.NaN, " A": np.NaN}, inplace=True)
        data_somnomedics["sleep_stage_num_psg"] = data_somnomedics["sleep_stage"]
        data_somnomedics["sleep_stage_num_psg"].replace({"3_Stage_Deep":3, "2_Stage_Light":2, "1_Stage_REM":1, "0_Stage_Wake":0}, inplace=True)

        data_somnomedics['timestamp_local'] =pd.to_datetime(data_somnomedics['timestamp_local']).dt.tz_localize('Europe/Paris')

        data_somnomedics['timestamp_local']=data_somnomedics['timestamp_local']+ pd.Timedelta('30s')

        data_somnomedics.set_index('timestamp_local', inplace = True)
        del data_somnomedics['sleep_stage']
        return (True, data_somnomedics)
    else:
        print(f'No Somnomedics Data for Participant: {subjectID}')
        return (False,)

