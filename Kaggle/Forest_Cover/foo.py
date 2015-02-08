from __future__ import print_function

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

data = pd.read_csv( 'train.csv', header = 0 )
outfile = open("output.txt", 'w+')
features = data.columns.values.tolist()
label = features.pop()

# Split the dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[label], test_size=0.33, random_state=0)

# Set the parameters
tuned_parameters = [{'n_estimators': [10, 100],
					 'max_features': ['sqrt', 'log2', None],
					 'max_depth': [None, 3, 5],
					 'n_jobs': [4]}]

objects = [RandomForestClassifier()]
scores = ['precision', 'recall']

for score in scores:
    outfile.write("# Tuning hyper-parameters for %s\n\n" % score)

    clf = GridSearchCV(objects[0], tuned_parameters, 
    				   cv=5, scoring=score)
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