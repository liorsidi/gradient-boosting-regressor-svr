# Create by Lior Sidi April 2017
# classification model Wrap GradientBoostingRegressor to run DecisionTreeSVRRegressor as it's estimators
# the DecisionTreeSVRRegressor are not generated in train phase only on test!
# my update for scikit learn gradient boosting package will enable to set a generic model
import os

from sklearn.svm import SVR
from sklearn import ensemble
import cPickle

class GradientBoostingRegressorSVRSimpleWrapper(object):
    # Predict only Gradient boosting regressor SVR in leaves
    # takes a trained original GBR and when predict it uses svr models on the tree in each stage

    def __init__(self, params_gbr, params_svr, dataset):
        self.params_gbr = params_gbr
        self.params_svr = params_svr
        self.svr_stages_leaves = None
        self.svr_on_all = None
        self.dataset = dataset
        self.gbr = ensemble.GradientBoostingRegressor(**params_gbr)

    def is_model_in_disk(self):
        gbr_file, svrs_file = self.models_name()
        return os.path.isfile(gbr_file) and os.path.isfile(svrs_file)

    def save(self):
        gbr_file, svrs_file = self.models_name()
        with open(gbr_file, 'wb') as fid:
            cPickle.dump(self.gbr, fid, cPickle.HIGHEST_PROTOCOL)
        with open(svrs_file, 'wb') as fid:
            cPickle.dump(self.svr_stages_leaves, fid, cPickle.HIGHEST_PROTOCOL)

    def load(self):
        if self.is_model_in_disk():
            gbr_file, svrs_file = self.models_name()
            with open(gbr_file, 'rb') as fid:
                self.gbr = cPickle.load(fid)
            with open(svrs_file, 'rb') as fid:
                self.svr_stages_leaves = cPickle.load(fid)

    def models_name(self):
        gbr = '{}_{}_{}_{}_{}_{}'.format(
            'gbr',
            str(self.params_gbr['n_estimators']),
            str(self.params_gbr['max_depth']),
            str(self.params_gbr['min_samples_split']),
            str(self.params_gbr['learning_rate']),
            str(self.params_gbr['criterion']))

        svrs = '{}_{}_{}'.format(
            'svrs',
            str(self.params_svr['kernel']),
            str(self.params_svr['epsilon']))
        gbr_file = 'models/' + self.dataset + '_' + gbr + '.pkl'
        svrs_file = 'models/' + self.dataset + '_' + svrs + '.pkl'
        return gbr_file, svrs_file

    def fit(self, X, y,force_fit = True, gbr = None):
        if self.is_model_in_disk() and not force_fit:
            self.load()
        else:
            if gbr is None:
                self.gbr.fit(X, y)
            else:
                self.gbr = gbr
            leaves_stages = self.gbr.apply(X)
            predict_stages = self.gbr.staged_predict(X)

            self.svr_stages_leaves = []
            # for each stage, extract instances in leaves, and predict instance error

            x_stage_leaves = []
            residual_stage_leaves = []
            for stage_id in range(0, int(leaves_stages.shape[1])):
                y_stage_pred = next(predict_stages)
                x_leaves = {}
                y_err_leaves = {}
                for x_id in range(0, X.shape[0]):
                    leaf_id = int(leaves_stages[x_id, stage_id])
                    if not x_leaves.has_key(leaf_id):
                        x_leaves[leaf_id] = []
                        y_err_leaves[leaf_id] = []
                    x_leaves[leaf_id].append(X[x_id])
                    residual = self.gbr.loss_.negative_gradient(y[x_id], y_stage_pred[x_id], sample_weight=None)
                    y_err_leaves[leaf_id].append(residual)
                # print y_err_leaves
                x_stage_leaves.append(x_leaves)
                residual_stage_leaves.append(y_err_leaves)

            for stage_id in range(0, int(leaves_stages.shape[1])):
                self.svr_stages_leaves.append({})
                for leaf_id in x_stage_leaves[stage_id].keys():
                    svr = SVR(**self.params_svr)
                    svr.fit(x_stage_leaves[stage_id][leaf_id], residual_stage_leaves[stage_id][leaf_id])
                    self.svr_stages_leaves[stage_id][leaf_id] = svr

            self.save()

    def predict(self, X):
        x_leaves_stages = self.gbr.apply(X)
        x_id = 0
        y = []
        # list of users with their staged predictions

        for x in X:
            y_u = self.gbr._init_decision_function(x)[0]
            for stage_id in range(0, x_leaves_stages.shape[1]):
                leaf_in_stage = x_leaves_stages[x_id][stage_id]
                svr = self.svr_stages_leaves[stage_id][leaf_in_stage]
                pred = svr.predict(x)[0]
                y_u += self.params_gbr['learning_rate'] * pred
            y.append(y_u)
            x_id += 1
        return y

