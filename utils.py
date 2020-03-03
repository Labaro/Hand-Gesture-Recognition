import pandas as pd
import numpy as np
import csv


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


def read_csv_infos(file):
    idx = []
    labels = []
    sizesequences = []

    with open(file, 'r') as f:
        data = csv.reader(f, delimiter=' ')
        for d in data:
            j = d[0].split(',')
            idx.append(int(j[0]))
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


def to_image(sample):
    min = np.array([np.min(sample[:, :, 0]), np.min(sample[:, :, 1]), np.min(sample[:, :, 2])])
    max = np.array([np.max(sample[:, :, 0]), np.max(sample[:, :, 1]), np.max(sample[:, :, 2])])
    im = (max - sample) / (max - min)
    return im
