import os

PATH = "../data/"
RAW_PATH = PATH + "raw/"
PROCESSED_PATH = PATH + "processed/"


def get_read_path():
    dirname = os.path.dirname(__file__)[:-7]
    abs_path = os.path.join(dirname, "data", "raw")
    if os.path.exists(RAW_PATH):
        return RAW_PATH
    elif os.path.exists(abs_path):
        return abs_path
    else:
        print("Data path not found.")


def get_write_path():
    dirname = os.path.dirname(__file__)[:-7]
    abs_path = os.path.join(dirname, "data", "processed")
    if os.path.exists(PROCESSED_PATH):
        return PROCESSED_PATH
    elif os.path.exists(abs_path):
        return abs_path
    else:
        print("Data path not found.")


ALL_PARTICIPANT_IDS = ["001", "002", "003", "008", "009", "011", "012", "013",
                   "015", "016", "022", "029", "032", "036", "037", "038",
                   "042", "043", "044", "045", "046", "047", "051", "053",
                   "054", "055", "057", "058", "060", "062", "063", "065",
                   "071", "072", "074", "076", "081", "084", "087", "091",
                   "092", "093", "094", "098", "101", "103", "106", "107",
                   "111", "114", "117", "122", "125", "126", "129", "133"]

PARTICIPANT_IDS = ["001", "002", "003", "008", "009", "011", "012", "013",
                   "015", "016", "022", "029", "032", "036", "037", "038",
                   "042", "043", "044", "045", "046", "047", "051", "053",
                   "054", "055", "057", "060", "062", "063", "065",
                   "071", "072", "076", "081", "091",
                    "093", "094", "101"]
