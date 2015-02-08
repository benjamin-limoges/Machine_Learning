from __future__ import print_function

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

data = pd.read_csv( 'train.csv', header = 0 )
features = data.columns.values.tolist()
label = features.pop()

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[label], test_size=0.5, random_state=0)

# Set the parameters by cross-validation


tuned_parameters = [{'weights': ['uniform'], 
					'n_neighbors' : [1, 3, 5, 7, 9]},
					{'weights': ['distance'],
					'n_neighbors' : [1, 3, 5, 7, 9]}]
scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))