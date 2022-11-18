from loader import *


if __name__ == '__main__':
    print('This will be run.py')
    data = get_sleepstages("058")
    print(data.shape)
    print(data.isna().sum())


