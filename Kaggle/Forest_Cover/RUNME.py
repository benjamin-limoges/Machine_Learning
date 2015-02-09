from src_files import global_vars
from src_files.grid_search import search, read_data
from sklearn.cross_validation import train_test_split

import os
import pickle

def main():
	names = global_vars.model_names
	parameters = global_vars.parameters
	classifiers = global_vars.classifiers

	for item in range(0, len(classifiers)):
		search(classifiers[item], parameters[item], names[item])

main()