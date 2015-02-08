from __future__ import print_function

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

data = pd.read_csv( 'train.csv', header = 0 )
features = data.columns.values.tolist()
label = features.pop()

# Split the dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[label], test_size=0.33, random_state=0)

# Set the parameters

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, 
    				   cv=5, scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on training set:\n")
    print(clf.best_estimator_)
    print("\n")
    print("Grid scores on training set:\n")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))