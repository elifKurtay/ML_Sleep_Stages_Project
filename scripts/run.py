from loader import *

if __name__ == '__main__':
    print('This will be run.py')
    trials = ["058", "074", "084", "087", "092", "098", "103", "106", "107", "111",
              "114", "117", "122", "125", "126", "129", "133"]
    path = get_write_path()
    for subjectID in trials:
        data = get_sleepstages(subjectID)
        if data is not None:
            data.to_csv(path + "/SMS_" + subjectID + ".csv", index=True)
