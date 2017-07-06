# Create by Lior Sidi April 2017
# evaluation and tuning and test for 3 type of models (on 5 diffrent datasets):
# 1. Original Gradient Boosting Regressor (GBR)
# 2. GradientBoostingRegressorSVR - update original scikit learrn  GradientBoostingRegressor to set any estimator (update the original gradient_boositng.py in scikit learn)
# 3. GradientBoostingRegressorSVRSimpleWrapper - wrap GradientBoostingRegressor to run DecisionTreeSVRRegressor as it's estimators (only at test)

import csv
import itertools

import time
from sklearn.datasets import make_hastie_10_2
import numpy as np

from sklearn import ensemble
from sklearn import datasets
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

from DecisionTreeSVRRegressor import DecisionTreeSVRRegressor
from GradientBoostingRegressorSVRSimpleWrapper import GradientBoostingRegressorSVRSimpleWrapper
from Utils import load_facebook, load_bike_rental, load_OnlineNewsPopularity, load_student_grades
import collections


def write_line(filename, dict, is_first=False):
    dict = collections.OrderedDict(sorted(dict.items()))
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=dict.keys())
        if is_first:
            writer.writeheader()
        writer.writerow(dict)


# return a list of experimet with all the relevant parameters
def get_index_product(params):
    i = 0
    params_index = {}
    for k, v in params.items():
        params_index[k] = i
        i += 1
    params_list = [None] * len(params_index.values())
    for name, loc in params_index.items():
        params_list[loc] = params[name]

    params_product = list(itertools.product(*params_list))
    params_product_dicts = []
    for params_value in params_product:
        params_dict = {}
        for param_name, param_index in params_index.items():
            params_dict[param_name] = params_value[param_index]
        params_product_dicts.append(params_dict)

    return params_product_dicts


## before running the code make sure you rplaced the gradient_boosting.py file in scikit learn
def main():
    params_gbrs = {
        'n_estimators': [100, 500],
        'max_depth': [3, 4, 6, 16],
        'min_samples_split': [2, 4],
        'learning_rate': [0.1, 0.01],
        'loss': ['ls'],
        'criterion': ['friedman_mse', 'mse'],
    }

    params_gbrs = {
        'n_estimators': [100 ],
        'max_depth': [16],
        'min_samples_split': [2, 4, 8, 16, 32, 64],
        'learning_rate': [0.01],
        'loss': ['ls'],
        'criterion': ['friedman_mse'],
    }

    experiments_params_gbrs = get_index_product(params_gbrs)
    params_svrs = {
        'kernel': ['rbf', 'poly'],
        'epsilon': [0.1, 0.2]
    }

    params_svrs = {
        'kernel': ['rbf'],
        'epsilon': [0.1]
    }
    #generate all combination of experiments with diffrent parameters
    experiments_params_svrs = get_index_product(params_svrs)

    is_first = True
    dataset_experiment = {
        'boston': datasets.load_boston(),
        'facebook': load_facebook(),
        'load_bike_rental': load_bike_rental(),
        'load_OnlineNewsPopularity': load_OnlineNewsPopularity(),
        'load_student_grades': load_student_grades()
    }

    # run experiment for each of the parameters
    for name_ds, ds in dataset_experiment.iteritems():
        X, y = shuffle(ds.data, ds.target, random_state=13)
        X = X.astype(np.float32)

        stats = {}
        stats['dataset'] = name_ds

        print name_ds
        for params_gbr in experiments_params_gbrs:
            for params_svr in experiments_params_svrs:
                gbr_svr_wrapper = GradientBoostingRegressorSVRSimpleWrapper(params_gbr, params_svr, name_ds)
                if gbr_svr_wrapper.is_model_in_disk():
                    print 'skipping'
                    break

                k_fold = KFold(n_splits=5)

                #evaluate each model and average in the end
                gbr_time = []
                gbr_mse = []
                gbr_mae = []
                gbr_svr_wrapper_time = []
                gbr_svr_wrapper_mse = []
                gbr_svr_wrapper_mae = []
                gbr_svr_time = []
                gbr_svr_mse = []
                gbr_svr_mae = []

                for train_indices, test_indices in k_fold.split(X):
                    # original GBR
                    X_train, y_train = X[train_indices], y[train_indices]
                    X_test, y_test = X[test_indices], y[test_indices]

                    gbr = ensemble.GradientBoostingRegressor(**params_gbr)
                    start_time = time.time()
                    gbr.fit(X_train, y_train)
                    gbr_time.append(round((time.time() - start_time) / 60.0, 3))
                    gbr_mse.append(mean_squared_error(y_test,
                                                      gbr.predict(
                                                          X_test)))
                    gbr_mae.append(mean_absolute_error(y_test,
                                                       gbr.predict(
                                                           X_test)))

                    # just predict with SVR in leaves based GBR
                    gbr_svr_wrapper = GradientBoostingRegressorSVRSimpleWrapper(params_gbr, params_svr, name_ds)
                    start_time = time.time()
                    gbr_svr_wrapper.fit(X_train, y_train, False, gbr)
                    gbr_svr_wrapper_time.append(round((time.time() - start_time) / 60.0, 3) + gbr_time[-1])
                    gbr_svr_wrapper_mse.append(mean_squared_error(y_test,
                                                                  gbr_svr_wrapper.predict(
                                                                      X_test)))
                    gbr_svr_wrapper_mae.append(mean_absolute_error(y_test,
                                                                   gbr_svr_wrapper.predict(
                                                                       X_test)))

                    # SVR in leaves based GBR
                    params_gbr_svr = params_gbr
                    params_gbr_svr['tree_constructor'] = DecisionTreeSVRRegressor
                    params_gbr_svr['extra_args'] = params_svr
                    gbr_svr = ensemble.GradientBoostingRegressor(**params_gbr_svr)
                    start_time = time.time()
                    gbr_svr.fit(X_train, y_train)
                    gbr_svr_time.append(round((time.time() - start_time) / 60.0, 3))
                    gbr_svr_mse.append(mean_squared_error(y_test, gbr_svr.predict(X_test)))
                    gbr_svr_mae.append(mean_absolute_error(y_test, gbr_svr.predict(X_test)))

                    del gbr
                    del gbr_svr_wrapper
                    del gbr_svr

                stats['gbr_time'] = np.mean(gbr_time)
                stats['gbr_mse'] = np.mean(gbr_mse)
                stats['gbr_mae'] = np.mean(gbr_mae)
                stats['gbr_svr_wrapper_time'] = np.mean(gbr_svr_wrapper_time)
                stats['gbr_svr_wrapper_mse'] = np.mean(gbr_svr_wrapper_mse)
                stats['gbr_svr_wrapper_mae'] = np.mean(gbr_svr_wrapper_mae)
                stats['gbr_svr_time'] = np.mean(gbr_svr_time)
                stats['gbr_svr_mse'] = np.mean(gbr_svr_mse)
                stats['gbr_svr_mae'] = np.mean(gbr_svr_mae)

                stats['gbr_time_std'] = np.std(gbr_time)
                stats['gbr_mse_std'] = np.std(gbr_mse)
                stats['gbr_mae_std'] = np.std(gbr_mae)
                stats['gbr_svr_wrapper_time_std'] = np.std(gbr_svr_wrapper_time)
                stats['gbr_svr_wrapper_mse_std'] = np.std(gbr_svr_wrapper_mse)
                stats['gbr_svr_wrapper_mae_std'] = np.std(gbr_svr_wrapper_mae)
                stats['gbr_svr_time_std'] = np.std(gbr_svr_time)
                stats['gbr_svr_mse_std'] = np.std(gbr_svr_mse)
                stats['gbr_svr_mae_std'] = np.std(gbr_svr_mae)

                stats.update(params_gbr_svr)
                print stats
                write_line('results.csv', stats, is_first)
                is_first = False


if __name__ == '__main__':
    main()
