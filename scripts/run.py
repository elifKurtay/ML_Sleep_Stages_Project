from loader import *


if __name__ == '__main__':
    print('This will be run.py')
    data = read_patient_data("001")
    print(data.head())
    print(data.isna().sum())


