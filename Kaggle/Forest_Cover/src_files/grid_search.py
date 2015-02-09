from __future__ import print_function

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import global_vars
import os

import pandas as pd

DATA_FILE = "train.csv"

def read_data():


    data = pd.read_csv( os.path.join( global_vars.data_dir, DATA_FILE),
                        header = 0 )
    features = data.columns.values.tolist()
    label = features.pop()

    return data, features, label

def search(model, parameters, filename):
    filename = filename + "_output.txt"
    outfile = open(os.path.join(global_vars.output_dir, filename), 'w+')
    data, features, label = read_data()

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data[features], data[label], test_size=0.33, random_state=0)

    scores = ['precision']

    for score in scores:
        outfile.write("# Tuning hyper-parameters for %s\n\n" % score)

        clf = GridSearchCV(model, parameters, cv=5, scoring=score)
        clf.fit(X_train, y_train)

        outfile.write("Best parameters set found on training set:\n\n")
        outfile.write(str(clf.best_estimator_))
        outfile.write("\n")
        outfile.write("Grid scores on training set:\n")
        for params, mean_score, scores in clf.grid_scores_:
            outfile.write("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() / 2, params))
            outfile.write("\n")
        outfile.write("\n")

    outfile.close()
    return clf.best_estimator_