import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle
import os
from sklearn.model_selection import cross_val_score

def find_params(X, y, X_test, feature_name, grid_size):
    # Parameters to try.
    param_grid = {"penalty": ['l1', 'l2'],
                  "C": [0.001, 0.005, 0.01, 0.05] + [0.1 * x for x in range(1, 50)],
                  "max_iter": [300],
                  "solver": ['liblinear'],
                  "tol": [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
                 }

    lreg = LogisticRegression()

    grid_search = GridSearchCV(lreg, param_grid=param_grid, cv=10, scoring='neg_log_loss', verbose=1, n_jobs=-1)
    grid_search.fit(X, y)

    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    parameters = grid_search.cv_results_['params']

    best_idx = grid_search.best_index_
    best_score = -grid_search.best_score_

    # Print the absolute best error for reference.
    print("")
    print("Absolute best LogRegr Score: " + str(best_score))
    print("Std Dev of best scoring estimator: " + str(stds[best_idx]))
    print("Params of best scoring estimator: " + str(parameters[best_idx]))
    print("")

    lowest_std = stds[best_idx]
    one_standard_idx = -1

    # Use the one-standard rule.
    for counter in range(len(means)):
        if (best_score + stds[counter] > -means[counter]) and stds[counter] < lowest_std:
            lowest_std = stds[counter]
            one_standard_idx = counter

            print("Found one with one-standard rule.")

    # None with lower variance was found, so use the one with lowest error.
    if one_standard_idx == -1:
        one_standard_idx = best_idx

    # Print the error and stddev of the one-standard rule estimator.
    print("One Standard LogRegr Score: " + str(-means[one_standard_idx]))
    print("Std Dev of 1-std estimator: " + str(stds[one_standard_idx]))
    print("Params of 1-std estimator: " + str(parameters[one_standard_idx]))
    print("")

    # Mainly for running on cluster.
    sys.stdout.flush()

    # Retrain using one-standard parameters.
    one_standard = LogisticRegression(**parameters[one_standard_idx])
    one_standard.fit(X, y)

    # Check cross-validation score again, since we had to retrain.
    # It may vary a lot if the algorithm is stochastic.
    scores = cross_val_score(one_standard, X, y, scoring='neg_log_loss', cv=10, n_jobs=-1)
    print("Refitted LogRegr score: " + str(-scores.mean()))
    print("Refitted stddev: " + str(scores.std()))
    print("")

    # Mainly for running on cluster.
    sys.stdout.flush()

    # Pickle one-standard estimator.
    est_save_path = "./src/estimators/"
    if not os.path.exists(est_save_path):
        os.makedirs(est_save_path)

    pickle.dump(one_standard, open(est_save_path + feature_name + '_grid_' + str(grid_size) + '_logreg.pkl', 'wb'))

    # Output predictions as a probability.
    pred_save_path = "./src/predictions/"
    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)

    y_test_prob = one_standard.predict_proba(X_test)

    # Output is the probability of being in class 1.
    if (str(one_standard.classes_[1]) == "1"):
        class_idx = 1
    else:
        class_idx = 0

    y_test = [0] * len(y_test_prob)

    for idx in range(len(y_test_prob)):
        y_test[idx] = y_test_prob[idx][class_idx]

    with open(pred_save_path + feature_name + '_grid_' + str(grid_size) + '_logreg_pred.csv', 'w') as out:
        out.write("ID,Prediction\n")
        for i in range(1, len(y_test) + 1):
            out.write(str(i) + "," + str(y_test[i - 1]) + "\n")

    # Return cross-validation error, stddev for weighting later.
    return (-scores.mean(), scores.std())