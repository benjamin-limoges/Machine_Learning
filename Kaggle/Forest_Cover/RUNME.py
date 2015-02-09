import global_vars
from grid_search import search, read_data
from sklearn.cross_validation import train_test_split

import os
import numpy as np

def main():
	names = global_vars.model_names
	parameters = global_vars.parameters
	classifiers = global_vars.classifiers

	outfile = open(os.path.join(global_vars.output_dir, "final_output.txt"),
		           'w+')

	models = []

	for item in range(0, len(classifiers)):
		models.append(search(classifiers[item], parameters[item], 
			          names[item]))

	data, features, label = read_data()
	X_train, X_test, y_train, y_test = train_test_split(
        data[features], data[label], test_size=0.33, random_state=0)
	for model in models:
		model.fit(X_train, y_train)
		outfile.write( str(model) )
		outfile.write( np.asarray(model.score(X_test, y_test)) )

	outfile.close()
main()