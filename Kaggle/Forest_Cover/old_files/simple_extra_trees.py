from sklearn.ensemble import ExtraTreesClassifier

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

	trees = fit_trees( training_set, features, label )
	classes = test_trees( test_set, features, trees, label )

	predictions = zip( test_set['Id'].values, classes )

	write_out( predictions )

	print time() - start, " seconds"

def write_out( print_Me ):

	with open('output.csv', 'wb') as csvfile:
		writer = csv.writer( csvfile, delimiter = ',')
		writer.writerow( ["Id","Cover_Type"])
		for row in range(0, len(print_Me)):
			writer.writerow(print_Me[row])

def test_trees( test_set, features, forest, label ):

	classes = forest.predict( test_set[features] )

	print forest.score( test_set[features], test_set[label] )

	return classes

def fit_trees( training_set, features, label ):

	trees = ExtraTreesClassifier( n_estimators = 1000, n_jobs = 2, bootstrap = False )
	trees.fit( training_set[features], training_set[label] )

	return trees

def open_data( filename ):

	df = pd.read_csv( filename, header = 0 )

	return df

main()