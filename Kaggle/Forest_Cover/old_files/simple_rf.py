from sklearn.ensemble import RandomForestClassifier

import numpy as np 
import pandas as pd 

import csv

from time import time

def main():

	start = time()

	training_set = open_data( "train.csv" )
	test_set = open_data( "test.csv" )

	features = training_set.columns.values.tolist()
	label = features.pop()

	forest = fit_forest( training_set, features, label )
	classes = test_forest( test_set, features, forest )

	predictions = zip( test_set['Id'].values, classes )

	write_out( predictions )

	print time() - start, " seconds"

def write_out( print_Me ):

	with open('output.csv', 'wb') as csvfile:
		writer = csv.writer( csvfile, delimiter = ',')
		writer.writerow( ["Id","Cover_Type"])
		for row in range(0, len(print_Me)):
			writer.writerow(print_Me[row])

def test_forest( test_set, features, forest ):

	classes = forest.predict( test_set[features] )

	return classes

def fit_forest( training_set, features, label ):

	forest = RandomForestClassifier( n_estimators = 1500, bootstrap = False, max_features = "sqrt", n_jobs = 2 )
	forest.fit( training_set[features], training_set[label] )

	return forest

def open_data( filename ):

	df = pd.read_csv( filename, header = 0 )

	return df

main()