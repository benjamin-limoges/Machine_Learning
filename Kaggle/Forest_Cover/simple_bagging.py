from sklearn.ensemble import BaggingClassifier

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

	bags = fit_bags( training_set, features, label )
	classes = test_bags( test_set, features, bags )

	predictions = zip( test_set['Id'].values, classes )

	write_out( predictions )

	print time() - start, " seconds"

def write_out( print_Me ):

	with open('output.csv', 'wb') as csvfile:
		writer = csv.writer( csvfile, delimiter = ',')
		writer.writerow( ["Id","Cover_Type"])
		for row in range(0, len(print_Me)):
			writer.writerow(print_Me[row])

def test_bags( test_set, features, bags ):

	classes = bags.predict( test_set[features] )

	return classes

def fit_bags( training_set, features, label ):

	bags = BaggingClassifier( n_estimators = 1000, n_jobs = 2, bootstrap = False )
	bags.fit( training_set[features], training_set[label] )

	return bags

def open_data( filename ):

	df = pd.read_csv( filename, header = 0 )

	return df

main()