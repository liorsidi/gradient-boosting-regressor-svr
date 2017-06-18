# Create by Lior Sidi April 2017
# evaluation and tuning and test for 3 type of models (on 5 diffrent datasets):
# 1. Original Gradient Boosting Regressor (GBR)
# 2. GradientBoostingRegressorSVR - update original scikit learrn  GradientBoostingRegressor to set any estimator (update the original gradient_boositng.py in scikit learn)
# 3. GradientBoostingRegressorSVRSimpleWrapper - wrap GradientBoostingRegressor to run DecisionTreeSVRRegressor as it's estimators (only at test)

import csv

from sklearn.datasets import make_hastie_10_2
import numpy as np

from sklearn import ensemble
from sklearn import datasets

from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

from DecisionTreeSVRRegressor import DecisionTreeSVRRegressor
from GradientBoostingRegressorSVRSimpleWrapper import GradientBoostingRegressorSVRSimpleWrapper
from Utils import load_facebook, load_bike_rental, load_OnlineNewsPopularity, load_student_grades

def write_line(filename, dict, is_first=False):
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=dict.keys())
        if is_first:
            writer.writeheader()
        writer.writerow(dict)

def main():
    n_estimators = [100, 500]
    max_depths = [3, 4]
    min_samples_split = [2, 4]
    learning_rates = [0.1, 0.01]
    losses = ['ls']  # , 'huber']
    criterions = ['friedman_mse', 'mse']

    kernels = ['rbf']  # , 'poly']
    epsilons = [0.1, 0.2]


    is_first = True
    dataset_experiment = {
       'boston': datasets.load_boston(),
        'facebook': load_facebook(),
        'load_bike_rental': load_bike_rental(),
        'load_OnlineNewsPopularity': load_OnlineNewsPopularity(),
        'load_student_grades': load_student_grades()
    }

    #run experiment for each of the parameters
    for name_ds, ds in dataset_experiment.iteritems():

        X, y = shuffle(ds.data, ds.target, random_state=13)
        X = X.astype(np.float32)
        offset = int(X.shape[0] * 0.9)
        X_train, y_train = X[:offset], y[:offset]
        X_test, y_test = X[offset:], y[offset:]
        stats = {}
        stats['dataset'] = name_ds
        svr_only_predict = False
        print name_ds
        for est in n_estimators:
            for max_depth in max_depths:
                for min_split in min_samples_split:
                    for learning_rate in learning_rates:
                        for loss in losses:
                            for criterion in criterions:
                                params_gbr = {'loss': loss,
                                              'n_estimators': est,
                                              'max_depth': max_depth,
                                              'min_samples_split': min_split,
                                              'learning_rate': learning_rate,
                                              'criterion': criterion}

                                for kernel in kernels:
                                    for epsilon in epsilons:
                                        params_svr = {'kernel': kernel, 'epsilon': epsilon}

                                        #original GBR classifier
                                        gbr_clf = ensemble.GradientBoostingRegressor(**params_gbr)

                                        # just predict with SVR in leaves based GBR classifier
                                        gbr_svr_clf_just_predict = GradientBoostingRegressorSVRSimpleWrapper(params_gbr, params_svr, 'predict')
                                        gbr_svr_clf_just_predict.fit(X_train, y_train)
                                        mse_gbr_svr_just_predict = mean_squared_error(y_test,
                                                                                      gbr_svr_clf_just_predict.predict(
                                                                                          X_test))

                                        # SVR in leaves based GBR classifier
                                        params_gbr_svr = params_gbr
                                        params_gbr_svr['tree_constructor'] = DecisionTreeSVRRegressor
                                        params_gbr_svr['extra_args'] = params_svr
                                        gbr_svr_clf = ensemble.GradientBoostingRegressor(**params_gbr_svr)
                                        gbr_svr_clf.fit(X_train, y_train)
                                        mse_gbr_svr = mean_squared_error(y_test, gbr_svr_clf.predict(X_test))

                                        gbr_clf.fit(X_train, y_train)
                                        mse_gbr = mean_squared_error(y_test, gbr_clf.predict(X_test))

                                        stats['mse_gbr'] = mse_gbr
                                        stats['mse_gbr_svr'] = mse_gbr_svr
                                        stats['mse_gbr_svr_just_predict'] = mse_gbr_svr_just_predict
                                        stats.update(params_gbr_svr)


                                        print "svr"
                                        print mse_gbr_svr
                                        print "svr predict:"
                                        print mse_gbr_svr_just_predict
                                        print "gbr:"
                                        print mse_gbr

                                        print stats
                                        write_line('results.csv', stats, is_first)
                                        is_first = False
                                        del gbr_svr_clf
                                        del gbr_clf

if __name__ == '__main__':
    main()