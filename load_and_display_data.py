# Script made by Theo Voillemin

import numpy as np
from scipy import misc
import os
import matplotlib.pyplot as plt
import csv


# Function to load a csv skeleton file sequence
# dim_joint = 2 if joints coordinates expressed in the depth image
# dim_joint = 3 if joints coordinates expressed in 3D world spaces.
def read_csv_skeleton(file, dim_joints):
    n = []
    n_samples = 0
    n_frames = 0
    with open(file, 'r') as f:
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


# Function to load a csv skeleton file sequence
# It returns idx, sequence label and sizesequence (with non zero padding frames)
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


# function to read sampleSubmission csv file
def read_csv_results():
    idx = []
    labels = []

    with open('results_test.csv', 'r') as f:
        data = csv.reader(f, delimiter=' ')
        for d in data:
            j = d[0].split(',')
            idx.append(int(j[0]))
            labels.append(int(j[1]))

    return idx, labels


# Function to display a sequence of 2d skeleton
def diplay_skeleton(skeletons_image, size):
    ####

    # Idx of the bones in the hand skeleton to display it.

    bones = np.array([
        [0, 1],
        [0, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [1, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [1, 10],
        [10, 11],
        [11, 12],
        [12, 13],
        [1, 14],
        [14, 15],
        [15, 16],
        [16, 17],
        [1, 18],
        [18, 19],
        [19, 20],
        [20, 21]
    ]
    );

    skeletons_image = np.reshape(skeletons_image,
                                 (skeletons_image.shape[0], skeletons_image.shape[1] * skeletons_image.shape[2]))

    pngDepthFiles = np.zeros([size, 480, 640])
    skeletons_display = np.zeros([size, 2, 2, 21])

    for id_image in range(0, size):
        # pngDepthFiles[id_image,:] = misc.imread(path_gesture+str(id_image)+'_depth.png')

        x = np.zeros([2, bones.shape[0]])
        y = np.zeros([2, bones.shape[0]])

        ske = skeletons_image[id_image, :]

        for idx_bones in range(0, bones.shape[0]):
            joint1 = bones[idx_bones, 0]
            joint2 = bones[idx_bones, 1]

            pt1 = ske[joint1 * 2:joint1 * 2 + 2]
            pt2 = ske[joint2 * 2:joint2 * 2 + 2]

            x[0, idx_bones] = pt1[0]
            x[1, idx_bones] = pt2[0]
            y[0, idx_bones] = pt1[1]
            y[1, idx_bones] = pt2[1]

        skeletons_display[id_image, 0, :, :] = x
        skeletons_display[id_image, 1, :, :] = y

    for id_image in range(0, size):
        plt.clf()
        plt.imshow(pngDepthFiles[id_image, :])
        plt.plot(skeletons_display[id_image, 0, :, :], skeletons_display[id_image, 1, :, :], linewidth=2.5)
        plt.pause(0.01)


if __name__ == '__main__':
    skeletons = read_csv_skeleton('skeletons_image_test.csv', 2)
    idx, labels, sequences = read_csv_infos('infos_train.csv')
    diplay_skeleton(skeletons[6], sequences[6])
