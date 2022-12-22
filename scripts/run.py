import keras as keras
from keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from loader import *

if __name__ == '__main__':
    print("The best model for our data is our sequential CNN model.")
    # load processed data and model
    x, y, _, _ = get_nn_patients()
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.30, shuffle=False, stratify=None)
    train_size, test_size = x_tr.shape[0] * x_tr.shape[1], x_te.shape[0] * x_te.shape[1]
    trainX = x_tr.reshape(train_size, 1, 2)
    trainy = y_tr.reshape(train_size)
    testX = x_te.reshape(test_size, 1, 2)
    y_te = y_te.reshape(test_size)

    trainy = to_categorical(trainy)
    testy = to_categorical(y_te)
    model = keras.models.load_model('path/to/location')


    #print(classification_report(true, preds))
    #print("Balanced accuracy:", balanced_accuracy_score(true, preds))
    #print("Accuracy:", accuracy_score(true, preds))
    #plot_conf_matrix(true, y_te, normalize=True)
