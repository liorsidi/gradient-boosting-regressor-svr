# GradientBoostingRegressorSVR

## Regressors
we evaluated three type of gradient boost regressors:
1.	**Original Gradient Boosting Regressor** - Scikit-learn, train trees with average in the leaves
2.	**Gradient boosting Regressor SVR in leaves** – we generalized the scikit-learn implementation to get any tree model and not to be a specific decision tree. under BaseGradientBoosting class we added 2 new parameters to the constructor:
a.	Tree constructor: the constructor function of the new tree
b.	Extra_args: relevant args for the tree
We also implemented a new class of decision tree regressor, DecisionTreeSVRRegressor that hold SVR model in the tree leaves. We use this tree in our new gradient boosting regressor implementation
3.	**Gradient boosting SVR wrapper** – GradientBoostingRegressorSVRSimpleWrapper, train an original GBR with the regular decision tree but when predict it uses SVR models in the leaves in each stage. This model apply the SVR only on predict.

## Experiment 
We evaluate “max depth” and the “min split” tree parameters on 5 different datasets, with 5 fold cross validation, for each evaluation we calculated the MSE (mean & std), MAE (mean & std), and the training time of the model.
Regard all other gradient boost and the SVR parameters; we set the following fixed parameters:
•	**Gradient boost** - 100 estimators with 0.01 learning rate with friedman_mse criterion
•	**SVR in the leaves** - rbf  kernel with 0.1 epsilon 
(These parameters showed in general good enough baseline results compare to other parameters)

## Results
GradientBoostingRegressorSVRSimpleWrapper has a significantly lower error rate compare to the other regressors. We assume the wrapper model showed better results due to the extra train after the gradient boost model-training phase. The non-wrapper models showed more or less similar results meaning that the optimization of the residual on SVR results does not show significant improvement compare to the improvement over the mean. 

But the GradientBoostingRegressorSVRSimpleWrapper time to train is higher than the others and the standard deviation of its results from the 5 fold cross validation was also higher meaning that the model might be is less stable than the others

For evaluating the “max depth” and “min spit” parameters we ran experiment with same validation method and with all other configurations presented above.
Unfortunately, we could not find any significant value of “max depth” or “min split” that will decrease the error significantly. We tested their values of 2, 4, 8, 16, 32 and there wasn’t a specific value that outperform the others
  

 
