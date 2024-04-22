#!/usr/bin/env python

#######################################################
# Copyright (c) 2024, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import os
import sys
# sys.path.insert(0, '../common')
from examples.common.idxio import read_idx

import arrayfire as af


def classify(arr, k, expand_labels):
    ret_str = ''
    if expand_labels:
        vec = af.cast(arr[:, k], af.f32)
        h_vec = vec.to_list()
        data = []

        for i in range(vec.size):
            data.append((h_vec[i], i))

        data = sorted(data, key=lambda pair: pair[0], reverse=True)

        ret_str = str(data[0][1])

    else:
        ret_str = str(int(af.cast(arr[k], af.float32).scalar()))

    return ret_str


def setup_mnist(frac, expand_labels):
    root_path = os.path.dirname(os.path.abspath(__file__))
    file_path = root_path + '/../../assets/examples/data/mnist/'
    idims, idata = read_idx(file_path + 'images-subset')
    ldims, ldata = read_idx(file_path + 'labels-subset')

    idims.reverse()
    numdims = len(idims)
    images = af.Array(idata, af.float32, tuple(idims))

    R = af.randu((10000, 1));
    cond = R < min(frac, 0.8)
    train_indices = af.where(cond)
    test_indices = af.where(~cond)

    train_images = af.lookup(images, train_indices, 2) / 255
    test_images = af.lookup(images, test_indices, 2) / 255


    num_classes = 10
    num_train = train_images.shape[2]
    num_test = test_images.shape[2]


    if expand_labels:
        train_labels = af.constant(0, (num_classes, num_train))
        test_labels = af.constant(0, (num_classes, num_test))

        h_train_idx = train_indices.copy()
        h_test_idx = test_indices.copy()

        ldata = list(map(int, ldata))

        for i in range(num_train):
            ldata_ind = ldata[h_train_idx[i].scalar()]
            train_labels[ldata_ind, i] = 1

        for i in range(num_test):
            ldata_ind = ldata[h_test_idx[i].scalar()]
            test_labels[ldata_ind, i] = 1
    
    else:
        labels = af.Array(idata, af.float32, tuple(idims))
        train_labels = labels[train_indices]
        test_labels = labels[test_indices]

    return (num_classes,
            num_train,
            num_test,
            train_images,
            test_images,
            train_labels,
            test_labels)


def display_results(test_images, test_output, test_actual, num_display, expand_labels):
    for i in range(num_display):
        print('Predicted: ', classify(test_output, i, expand_labels))
        print('Actual: ', classify(test_actual, i, expand_labels))

        img = af.cast((test_images[:, :, i] > 0.1), af.u8)
        img = af.moddims(img, (img.size,)).to_list()
        for j in range(28):
            for k in range(28):
                print('\u2588' if img[j * 28 + k] > 0 else ' ', end='')
            print()
        input()