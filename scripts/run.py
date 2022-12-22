from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB

from loader import *

if __name__ == '__main__':
    print("The best model for our data is scikit-learn's CategoricalNB classifier.")
    # load processed data
    x, y, _, _ = get_nn_patients()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=False, stratify=None)
    train_size, test_size = x_train.shape[0] * x_train.shape[1], x_test.shape[0] * x_test.shape[1]
    x_tr, x_te, y_tr, y_te = x_train.reshape(train_size, 2), x_test.reshape(test_size, 2), y_train\
        .ravel(), y_test.ravel()

    # train CategoricalNB classifier
    cnb_classifier = CategoricalNB()
    cnb_classifier.fit(x_tr, y_tr)
    preds = cnb_classifier.predict(x_te)

    # observe results
    print(classification_report(y_te, preds))
    print("Balanced accuracy:", balanced_accuracy_score(y_te, preds))
    print("Accuracy:", accuracy_score(y_te, preds))
    plot_conf_matrix(preds, y_te, normalize=True)
