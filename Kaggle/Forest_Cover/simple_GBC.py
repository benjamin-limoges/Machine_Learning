from sklearn.ensemble import GradientBoostingClassifier
import numpy as np 
import pandas as pd 

import csv

from time import time

def main():

	start = time()

	training_set = open_data( "train.csv" )
	test_set = open_data( "test.csv" )

	#test_idx = np.random.randint( 0, 3, len(train) )

	#training_set = train[ test_idx == 0 ]
	#test_set = train[ test_idx != 0]


	features = training_set.columns.values.tolist()
	label = features.pop()

	bags = fit_bags( training_set, features, label )
	classes = test_bags( test_set, features, bags )

	predictions = zip( test_set['Id'].values, classes )

	write_out( predictions )

	print time() - start, " seconds"

def write_out( print_Me ):

	with open('output_GBC.csv', 'wb') as csvfile:
		writer = csv.writer( csvfile, delimiter = ',')
		writer.writerow( ["Id","Cover_Type"])
		for row in range(0, len(print_Me)):
			writer.writerow(print_Me[row])

def test_bags( test_set, features, bags ):

	classes = bags.predict( test_set[features] )

	#print bags.score( test_set[features], test_set[label] )

	return classes

def fit_bags( training_set, features, label ):

	bags = GradientBoostingClassifier( n_estimators = 100, learning_rate = 0.01 )
	bags.fit( training_set[features], training_set[label] )

	return bags

def open_data( filename ):

	df = pd.read_csv( filename, header = 0 )

	return df

main()