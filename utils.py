import pandas as pd
import numpy as np
import csv
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.layers import BatchNormalization

from keras.optimizers import SGD, Adam

from scipy.interpolate import splev, splprep


def get_skeleton(file_path, dim_joints=3):
    n = []
    n_samples = 0
    n_frames = 0
    with open(file_path, 'r') as f:
        data = csv.reader(f, delimiter=' ')
        for d in data:
            n_samples += 1
            n_frames = 0
            for j in d[0].split(','):
                n.append(float(j))
                n_frames += 1
    # convert data in numpy array
    n = np.asarray(n)
    n = np.reshape(n, (n_samples, n_frames // (22 * dim_joints), 22, dim_joints))
    return n


def read_csv_infos(file, test=False):
    idx = []
    labels = []
    sizesequences = []

    with open(file, 'r') as f:
        data = csv.reader(f, delimiter=' ')
        for d in data:
            j = d[0].split(',')
            idx.append(int(j[0]))
            if test:
                sizesequences.append(int(j[1]))
            else:
                labels.append(int(j[1]))
                sizesequences.append(int(j[2]))

    return idx, labels, sizesequences


def normalize_position(n, joint=0):
    N = n.copy()
    nb_samples = N.shape[0]
    for i in range(nb_samples):
        N[i, :, :, :] -= N[i, 0, joint, :]
    return N


def get_vectors(n):
    nb_samples, nb_frames = n.shape[:2]
    N = np.zeros((nb_samples, nb_frames - 1, 22, 3))
    for i in range(nb_samples):
        for j in range(nb_frames - 1):
            N[i, j, :, :] = n[i, j + 1, :, :] - n[i, j, :, :]
    return N


def reset_zeros(n, infos):
    N = n.copy()
    nb_samples, nb_frames = N.shape[:2]
    for i in range(nb_samples):
        N[i, infos[i] - 1:, :, :] = np.zeros((nb_frames - infos[i] + 1, 22, 3))
    return N


def to_image(n, k, min, max):
    sample = n[k]
    # min = np.array([np.min(n[:, :, :, 0]), np.min(n[:, :, :, 1]), np.min(n[:, :, :, 2])])
    # max = np.array([np.max(n[:, :, :, 0]), np.max(n[:, :, :, 1]), np.max(n[:, :, :, 2])])
    im = (max - sample) / (max - min)
    return im


def interpolate(n, sequence_sizes, final_size):
    N = np.zeros((n.shape[0], final_size, n.shape[2], n.shape[3]))
    for j in range(n.shape[0]):
        for i in range(22):
            sample = n[j, :sequence_sizes[j], i, :]
            okay = np.where(
                np.abs(np.diff(sample[:, 0])) + np.abs(np.diff(sample[:, 1])) + np.abs(np.diff(sample[:, 2])) > 0)
            okay = np.append(okay, okay[0][-1] + 1)
            sample = sample[okay].T
            tck, u = splprep([*sample], s=0, k=2)
            X, Y, Z = splev(np.linspace(0, 1, final_size), tck)
            new_points = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)])
            N[j, :, i, :] = new_points
    return N


def get_train_test(test_size, seed=None):
    list_image = []
    for i in range(1960):
        im = mpimg.imread(f"image/train/{i}.png")
        list_image.append(np.copy(im)[:, :, :3])
    _, labels, _ = read_csv_infos('infos_train.csv')
    labels = to_categorical(np.array(labels) - 1)
    X_train, X_test, y_train, y_test = train_test_split(list_image, labels, test_size=test_size, random_state=seed)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def run_test_harness():
    trainX, testX, trainY, testY = get_train_test(0.1, seed=1)
    # define model
    model = define_model()
    # fit model
    history = model.fit(trainX, trainY, epochs=35, batch_size=10, validation_data=(testX, testY), verbose=1)
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)
    return model


def summarize_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.close()


def define_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(162, 22, 3), padding='same', strides=1, activation='relu'))
    model.add(BatchNormalization(trainable=True))
    model.add(Conv2D(64, (3, 3), padding='same', strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', strides=1, activation='relu'))
    model.add(BatchNormalization(trainable=True))
    model.add(Conv2D(128, (3, 3), padding='same', strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding='same', strides=1, activation='relu'))
    model.add(BatchNormalization(trainable=True))
    model.add(Conv2D(256, (3, 3), padding='same', strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding='same', strides=1, activation='relu'))
    model.add(BatchNormalization(trainable=True))
    model.add(Conv2D(512, (3, 3), padding='same', strides=1, activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.4))
    model.add(Dense(28, activation='softmax'))
    opt = Adam(learning_rate=0.005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
