from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
import pickle
import sys, os
import Queue
from heapq import heappush, heappop
from random import randint, random
from sklearn.model_selection import GridSearchCV

params = {'batch_size':'auto', 'solver':'lbfgs',
          'learning_rate':'adaptive', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':330, 'shuffle':True,
          'random_state':None, 'verbose':False, 'warm_start':False, 'momentum':0.9,
          'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9,
          'beta_2':0.999, 'epsilon':1e-08}

# Returns the mean cross-val error of the estimator using "one-standard" rule.
# Stores the pickled estimator in ./src/estimators/<feature_name>_nn.pkl.
# Stores its predictions in ./src/predictions/<feature_name>_nn_pred.csv.
def find_params(X, y, X_test, feature_name):
    activations = ['identity', 'logistic', 'tanh', 'relu']

    lowest_err = sys.maxint

    num_stored = 50
    #mlps = Queue.Queue(maxsize = num_stored) # Store up to <num_stored> neural nets.
    mlps = []

    num_loops = 50

    for i in range(num_loops):
        if i % 5 == 0:
            print("NN Loop: " + str(i))

            # Mainly for running on cluster.
            sys.stdout.flush()

        # Monte Carlo the # of layers and neurons
        num_hidden_layers = randint(1, 30)
        layer_structure = [0] * num_hidden_layers
        for j in range(num_hidden_layers):
            layer_structure[j] = randint(2, 100)

        # Randomly choose non-linear function.
        activation = activations[randint(0, len(activations) - 1)]

        # Monte Carlo the value for alpha and tol [0, 0.2)
        alpha = random() / 5
        tol = random() / 5

        # Train neural nets
        mlp = MLPClassifier(hidden_layer_sizes = tuple(layer_structure), alpha = alpha, tol = tol, activation = activation, **params)

        # Cross-Validation
        scores = cross_val_score(mlp, X, y, scoring=make_scorer(log_loss, greater_is_better=False), cv=10, n_jobs=-1)

        # Find the lowest error.
        if (-scores.mean() < lowest_err):
            lowest_err = -scores.mean()

        # Store estimator in queue if it has lower error than the last one that entered the queue.

        if len(mlps) > 0:
            some_estimator = heappop(mlps)

            # If the estimator we just found has lower error, put it in ahead of the other estimator.
            if -scores.mean() < -some_estimator[0]:
                heappush(mlps, (scores.mean(), [mlp, scores.std()]))

                if len(mlps) < num_stored:
                    heappush(mlps, some_estimator)

            # Else only put it in if there's space.
            elif len(mlps) < num_stored:
                heappush(mlps, (scores.mean(), [mlp, scores.std()]))
        else:
            heappush(mlps, (scores.mean(), [mlp, scores.std()]))

    # Pickle the best neural nets to save them.
    est_save_path = './src/estimators/'

    if not os.path.exists(est_save_path):
        os.makedirs(est_save_path)

    lowest_std = sys.maxint

    best_stddev = sys.maxint
    best_est = None
    best_params = None

    one_standard_err = sys.maxint
    one_standard_std = sys.maxint
    one_standard_est = None
    one_standard_params = None
    one_standard_found = False

    while len(mlps) > 0:
        next_nn = heappop(mlps)

        # Get the info on the lowest error neural net.
        if len(mlps) == 0:
            best_stddev = next_nn[1][1]
            best_params = next_nn[1][0].get_params()
            best_est = next_nn[1][0]

        # Use the one-standard rule.
        if (lowest_err + next_nn[1][1] > -next_nn[0]) and next_nn[1][1] < lowest_std:
            lowest_std = next_nn[1][1]

            one_standard_err = -next_nn[0]
            one_standard_std = next_nn[1][1]
            one_standard_est = next_nn[1][0]
            one_standard_params = next_nn[1][0].get_params()
            one_standard_found = True

            print("Found one with one-standard rule.")

    # If nothing found with one-standard rule, use one with lowest error.
    if not one_standard_found:
        one_standard_est = best_est
        one_standard_params = best_params
        one_standard_err = lowest_err
        one_standard_std = best_stddev

    pickle.dump(one_standard_est, open(est_save_path + 'nn.pkl', 'wb'))

    # Print the absolute best error for reference.
    print("Absolute best Neural Net score: " + str(lowest_err))
    print("Std Dev of best scoring estimator: " + str(best_stddev))
    print("Params of best scoring estimator: " + str(best_params))
    print("")

    # Print the error and stddev of the one-standard rule estimator.
    print("One Standard Neural Net score: " + str(one_standard_err))
    print("Std Dev of 1-std estimator: " + str(one_standard_std))
    print("Params of 1-std estimator: " + str(one_standard_params))

    # Mainly for running on cluster.
    sys.stdout.flush()

    # Output predictions as a probability.
    pred_save_path = './src/predictions/'
    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)

    one_standard_est.fit(X, y)
    y_test_prob = one_standard_est.predict_proba(X_test)

    if (str(one_standard_est.classes_[1]) == "1"):
        class_idx = 1
    else:
        class_idx = 0

    y_test = [0] * len(y_test_prob)

    for idx in range(len(y_test_prob)):
        y_test[idx] = y_test_prob[idx][class_idx]

    with open(pred_save_path + 'nn_pred.csv', 'w') as out:
        out.write("ID,Prediction\n")
        for i in range(1, len(y_test) + 1):
            out.write(str(i) + "," + str(y_test[i - 1]) + "\n")

    # Return cross-validation error for weighting later.
    return (one_standard_err, one_standard_std)



