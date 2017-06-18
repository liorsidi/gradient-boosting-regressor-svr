# By Lior Sidi and Hadar Klein
# based on Ordinal Classifier implementation
# Frank, Eibe, and Mark Hall. "A simple approach to ordinal classification." European Conference on Machine Learning. Springer Berlin Heidelberg, 2001.
# chapter 2

import numpy as np
import os
import csv
from sklearn.datasets.base import Bunch

def load_facebook(return_X_y=False, y_type='all'):
    data_file_name = 'datasets\Facebook_metrics\dataset_Facebook.csv'
    file_path = os.path.join('datasets\Facebook_metrics', 'dataset_Facebook.csv')
    with open(file_path) as f:
        data_file = csv.reader(f, delimiter=';')

        # temp = next(data_file)
        temp = next(data_file)

        n_features = 11
        # n_samples = sum(1 for row in data_file)
        # data = np.empty((n_samples, n_features))
        # target = np.empty((n_samples,))
        feature_names = np.array(temp)

    data = []
    target = []
    with open(file_path) as f:
        data_file = csv.reader(f, delimiter=';')
        i = 0
        firstline = True
        for d in data_file:
            if firstline:
                firstline = False
            else:
                d = np.array(d)
                d[d == 'Photo'] = 3.0
                d[d == 'Status'] = 1.0
                d[d == 'Link'] = 2.0
                d[d == 'Video'] = 4.0
                d[d == ''] = 0.0
                d = np.array(d, dtype='float')
                d.astype(float)
                data.append(d[:11])
                if y_type == 'all':
                    target.append(d[-1])
                elif y_type == 'comment':
                    target.append(d[12])
                elif y_type == 'like':
                    target.append(d[13])
                elif y_type == 'share':
                    target.append(d[14])
                i += 1
    data = np.array(data)
    target = np.array(target)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names)

def load_bike_rental(return_X_y=False):
    file_path = os.path.join('datasets\Bike-Sharing-Dataset', 'day.csv')
    with open(file_path) as f:
        data_file = csv.reader(f, delimiter=';')

        # temp = next(data_file)
        temp = next(data_file)

        n_features = 14
        # n_samples = sum(1 for row in data_file)
        # data = np.empty((n_samples, n_features))
        # target = np.empty((n_samples,))
        feature_names = np.array(temp)

    data = []
    target = []
    with open(file_path) as f:
        data_file = csv.reader(f, delimiter=',')
        i = 0
        firstline = True
        for d in data_file:
            if firstline:
                firstline = False
            else:
                d = np.array(d[2:], dtype='float')
                d.astype(float)
                data.append(d[:-1])
                target.append(d[-1])
                i += 1
    data = np.array(data).astype(float)
    target = np.array(target).astype(float)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names)

def load_OnlineNewsPopularity(return_X_y=False):
    file_path = os.path.join('datasets\OnlineNewsPopularity','OnlineNewsPopularity.csv')
    with open(file_path) as f:
        data_file = csv.reader(f, delimiter=';')

        # temp = next(data_file)
        temp = next(data_file)

        n_features = 59
        # n_samples = sum(1 for row in data_file)
        # data = np.empty((n_samples, n_features))
        # target = np.empty((n_samples,))
        feature_names = np.array(temp)

    data = []
    target = []
    with open(file_path) as f:
        data_file = csv.reader(f, delimiter=',')
        i = 0
        firstline = True
        for d in data_file:
            if firstline:
                firstline = False
            else:
                d = np.array(d[1:], dtype='float')
                d.astype(float)
                data.append(d[:-1])
                target.append(d[-1])
                i += 1
    data = np.array(data[:1000])
    target = np.array(target[:1000])

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names)

def load_student_grades(return_X_y=False, y_type='G3'):
    strings = {
        'at_home': 1,
        'health': 2,
        'other': 5,
        'services': 3,
        'teacher': 4,
        'GP': 1,
        'MS': 2,
        'course': 1,
        'home': 2,
        'other': 3,
        'reputation': 4,

        'father': 2,
        'mother': 1,
        'other': 3,
        'F': 1,
        'M': 0,
        'yes': 1,
        'no': 0
    }
    file_path = os.path.join('datasets\student', 'student-por.csv')
    with open(file_path) as f:
        data_file = csv.reader(f, delimiter=';')

        # temp = next(data_file)
        temp = next(data_file)

        n_features = 30
        # n_samples = sum(1 for row in data_file)
        # data = np.empty((n_samples, n_features))
        # target = np.empty((n_samples,))
        feature_names = np.array(temp)

    data = []
    target = []
    with open(file_path) as f:
        data_file = csv.reader(f, delimiter=';')
        i = 0
        firstline = True
        for d in data_file:
            if firstline:
                firstline = False
            else:
                d = np.array(d)
                d = np.delete(d, 3, 0)
                d = np.delete(d, 3, 0)
                d = np.delete(d, 3, 0)
                for str, val in strings.iteritems():
                    d[d == str] = val


                #data.append(d[1:-3])
                data.append(np.array(d[1:-3], dtype='float'))

                d.astype(float)
                if y_type == 'G3':
                    target.append(d[-1])
                elif y_type == 'G2':
                    target.append(d[-2])
                elif y_type == 'G1':
                    target.append(d[-3])
                i += 1
    data = np.array(data[:1000])
    target = np.array(target[:1000]).astype(float)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names)
