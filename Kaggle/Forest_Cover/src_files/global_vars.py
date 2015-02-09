from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

classifiers = [ExtraTreesClassifier(), RandomForestClassifier(),
			   LogisticRegression()]


parameters = [{'n_estimators': [10, 100],
			   'max_features': ['sqrt', 'log2', None],
			   'max_depth': [3, 5, None],
			   'n_jobs': [4]},
			  {'n_estimators': [10, 100],
			   'max_features': ['sqrt', 'log2', None],
			   'max_depth': [3, 5, None],
			   'n_jobs': [4]},
			  {'penalty': ['l1', 'l2'],
			   'C': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]}]

model_names = ["Extra_Trees", "Random_Forests", "Logistic_Regression"]

output_dir = "output/"
data_dir = "data/"
