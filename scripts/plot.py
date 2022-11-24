import numpy as np
import matplotlib.pyplot as plt


def plot_missing_sleepstages(df):
    color_map = [("sleep_stage_num_somnofy", "blue"), ("sleep_stage_num_emfit", "orange"), ("sleep_stage_num_psg", "green")]
    truth_vals = df["sleep_stage_num_psg"].to_numpy()
    for ind, (title, color) in enumerate(color_map):
        missing_ind = np.array(np.where(df[title].isnull())[0])
        y = np.ones(len(missing_ind))
        if ind == 1:
            y[:] = 2
        elif ind == 2:
            y[:] = 3
        else:
            print("Somnofy is missing where psg has the following values:")
            unique_elements, counts_elements = np.unique(truth_vals[missing_ind], return_counts=True)
            print(np.asarray((unique_elements, counts_elements)))
        plt.scatter(missing_ind, y, color=color)
    plt.show()
