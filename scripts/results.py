import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
import warnings

from constants import PARTICIPANT_IDS
from loader import read_patient_data


def overall_balanced_accuracy():
    """
    gives overall balanced accuracy results for raw data, KNN, and NB
    :return:
    """
    warnings.warn("deprecated", DeprecationWarning)
    radar_scores = []
    mat_scores = []
    knn_scores = []
    nb_scores = []
    for subjectID in PARTICIPANT_IDS:
        sleep_stages = read_patient_data(subjectID)
        labels = sleep_stages["sleep_stage_num_psg"]
        features = sleep_stages.drop(columns="sleep_stage_num_psg")
        size = sleep_stages.shape[0]
        divide_ind = int(size * .7)
        # accuracy for radar and mat alone
        radar_scores.append(
            balanced_accuracy_score(labels[:divide_ind], sleep_stages["sleep_stage_num_somnofy"][:divide_ind]))
        mat_scores.append(
            balanced_accuracy_score(labels[:divide_ind], sleep_stages["sleep_stage_num_emfit"][:divide_ind]))
        # accuracy for KNN
        x_tr, y_tr = features[:divide_ind], labels[:divide_ind]
        x_te, y_te = features[divide_ind:], labels[divide_ind:]
        knn_classifier = KNeighborsClassifier(n_neighbors=10)
        knn_classifier.fit(x_tr, y_tr)
        preds = knn_classifier.predict(x_te)
        knn_scores.append(balanced_accuracy_score(y_te, preds))
        # accuracy for NB
        cnb_classifier = CategoricalNB()
        cnb_classifier.fit(x_tr, y_tr)
        preds = cnb_classifier.predict(x_te)
        nb_scores.append(balanced_accuracy_score(y_te, preds))

    print("Radar:   Acc = ", np.average(radar_scores), "St. Dv. = ", np.std(radar_scores))
    print("Mat:   Acc = ", np.average(mat_scores), "St. Dv. = ", np.std(mat_scores))
    print("kNN:   Acc = ", np.average(knn_scores), "St. Dv. = ", np.std(knn_scores))
    print("NB:   Acc = ", np.average(nb_scores), "St. Dv. = ", np.std(nb_scores))
