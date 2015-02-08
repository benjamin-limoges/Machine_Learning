from sklearn.neighbors import KNeighborsClassifier
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

	features = features[1:11]

	neighbors = fit_neighbors( training_set, features, label )
	classes = test_neighbors( test_set, features, neighbors )

	predictions = zip( test_set['Id'].values, classes )

	write_out( predictions )

	print time() - start, " seconds"

def write_out( print_Me ):

	with open('output_knn.csv', 'wb') as csvfile:
		writer = csv.writer( csvfile, delimiter = ',')
		writer.writerow( ["Id","Cover_Type"])
		for row in range(0, len(print_Me)):
			writer.writerow(print_Me[row])

def test_neighbors( test_set, features, neighbors ):

	classes = neighbors.predict( test_set[features] )

	#print neighbors.score( test_set[features], test_set[label])

	return classes

def fit_neighbors( training_set, features, label ):

	neighbors = KNeighborsClassifier( n_neighbors = 7, weights = "distance" )
	neighbors.fit( training_set[features], training_set[label] )

	return neighbors

def open_data( filename ):

	df = pd.read_csv( filename, header = 0 )

	return df

main()