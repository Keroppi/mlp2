import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import sys
from sklearn.model_selection import cross_val_score

class Model:
    def __init__(self, function, param_grid, feature_name, grid_size, crop_str):
        self.model_name = function.__name__
        self.feature_name = feature_name
        self.grid_size = grid_size
        self.crop_str = crop_str
        self.function = function
        self.param_grid = param_grid
        self.estimator = None

    def find_hyperparams(self, X, y):
        # Try all combinations of parameters.
        model = self.function()
        grid_search = GridSearchCV(model, param_grid=self.param_grid, cv=10, scoring='neg_log_loss', verbose=1, n_jobs=-1)
        grid_search.fit(X, y)

        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        parameters = grid_search.cv_results_['params']

        best_idx = grid_search.best_index_
        best_score = -grid_search.best_score_

        # Print the absolute best error for reference.
        print("Absolute best " + self.model_name + " score: " + str(best_score))
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
        print("One Standard " + self.model_name + " score: " + str(-means[one_standard_idx]))
        print("Std Dev of 1-std estimator: " + str(stds[one_standard_idx]))
        print("Params of 1-std estimator: " + str(parameters[one_standard_idx]))

        # Mainly for running on cluster.
        sys.stdout.flush()

        # Retrain using one-standard parameters.
        one_standard = self.function(**parameters[one_standard_idx])
        one_standard.fit(X, y)

        # Check cross-validation score again, since we had to retrain.
        # It may vary a lot if the algorithm is stochastic.
        scores = cross_val_score(one_standard, X, y, scoring='neg_log_loss', cv=10, n_jobs=-1)
        print("Refitted " + self.model_name + " score: " + str(-scores.mean()))
        print("Refitted stddev: " + str(scores.std()))
        print("")

        # Mainly for running on cluster.
        sys.stdout.flush()

        # Pickle one-standard estimator.
        est_save_path = "./src/estimators/"
        if not os.path.exists(est_save_path):
            os.makedirs(est_save_path)

        pickle.dump(one_standard, open(est_save_path + self.feature_name + '_crop_' + self.crop_str + '_grid_' + str(self.grid_size) + '_' + self.model_name + '.pkl', 'wb'))

        self.estimator = one_standard

        # Return cross-validation error, stddev for weighting later.
        return (-scores.mean(), scores.std())

    def output_predictions(self, X_test):
        # Output predictions as a probability.
        pred_save_path = "./src/predictions/"
        if not os.path.exists(pred_save_path):
            os.makedirs(pred_save_path)

        y_test_prob = self.estimator.predict_proba(X_test)

        if (str(self.estimator.classes_[1]) == "1"):
            class_idx = 1
        else:
            class_idx = 0

        y_test = [0] * len(y_test_prob)

        for idx in range(len(y_test_prob)):
            y_test[idx] = y_test_prob[idx][class_idx]

        prediction_file_path = pred_save_path + self.feature_name + '_crop_' + self.crop_str + '_grid_' + str(self.grid_size) + '_' + self.model_name + '_pred.csv'

        with open(prediction_file_path, 'w') as out:
            out.write("ID,Prediction\n")
            for i in range(1, len(y_test) + 1):
                out.write(str(i) + "," + str(y_test[i - 1]) + "\n")

        return prediction_file_path



