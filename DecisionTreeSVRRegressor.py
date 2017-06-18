# Create by Lior Sidi April 2017
# classification model that hold svr model in a tree leaves

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

class DecisionTreeSVRRegressor():
    #class that hold svr model in the tree leaves
    #implements all of DecisionTree methods and properties
    def __init__(self,
                 params_svr = {'kernel': 'rbf', 'epsilon': 0.2},
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 presort=False):
        self.base_tree = DecisionTreeRegressor(
            criterion=criterion,
            splitter='best',
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            presort=presort)

        self.params_svr = params_svr
        self.x_leaves = {}
        self.y_leaves = {}
        self.svrs_leaves = {}
        self.tree_ = None

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):

        self.base_tree.fit(X, y, sample_weight, check_input,
            X_idx_sorted)

        self.tree_ = self.base_tree.tree_

        leaves = self.base_tree.apply(X)

        for x_id in range(0, X.shape[0]):
            leaf_id = int(leaves[x_id])
            if not self.x_leaves.has_key(leaf_id):
                self.x_leaves[leaf_id] = []
                self.y_leaves[leaf_id] = []
            self.x_leaves[leaf_id].append(X[x_id])
            self. y_leaves[leaf_id].append(y[x_id])


        for leaf_id in self.x_leaves.keys():
            svr = SVR(**self.params_svr)
            svr.fit(self.x_leaves[leaf_id], self.y_leaves[leaf_id])
            self.svrs_leaves[leaf_id] = svr

        return self

    def predict(self, X, check_input=True):
        y = []
        x_leaves = self.base_tree.apply(X)
        for x in X:
            leaf = x_leaves[x]
            svr = self.svrs_leaves[leaf]
            pred = svr.predict(x)[0]
            y.append(pred)
        return y

    def apply(self, X, check_input=True):
        return self.base_tree.apply(X)

    def decision_path(self, X, check_input=True):
        return self.decision_path.apply(X)

    def feature_importances_(self):
        return self.decision_path.feature_importances_()


    def _validate_X_predict(self, X, check_input=True):
        return self.base_tree._validate_X_predict(X, check_input )

