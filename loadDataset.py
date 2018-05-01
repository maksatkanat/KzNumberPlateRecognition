from keras.utils import np_utils
import scipy.io
import numpy as np

#load dataSet from npz file for Training (train_cnn_improved.py)

def load_data_2d():
    rows, cols = 28, 28
    nb_classes = 35

    letters = np.load('syntez-new.npz')

    X = letters['X']
    Y = letters['Y']
    #print(X.shape)

    X = X.reshape(X.shape[0], rows, cols, 1)
    X = X.astype('float32')
    X /= 255.0
    #Y = Y-1
    Y = Y.astype(int)
    Y = np_utils.to_categorical(Y, nb_classes)

    # Divide into test and train sets
    perm = np.random.permutation(X.shape[0])

    train_size = int(X.shape[0]*0.8)   # total training examples: 118000

    X_train = X[perm[:train_size]]
    X_test = X[perm[train_size:]]

    Y_train = Y[perm[:train_size]]
    Y_test = Y[perm[train_size:]]


    return (X_train, Y_train, X_test, Y_test)
