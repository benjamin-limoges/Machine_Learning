from sklearn.ensemble import RandomForestClassifier
from time import time
from collections import defaultdict

import numpy as np 
import pandas as pd 

import csv

PARTS = 10
np.random.seed( seed = 1337 )

def main():

	start = time()

	training_set = open_data( "train.csv" )
	test = open_data( "test.csv" )

	features = training_set.columns.values.tolist()
	label = features.pop()

	test_idx = assign_random( len(training_set) )



	results = build_forests( training_set, test, features, label, test_idx )

	prob_distr = add_results( results, test['Id'].values )

	probs = select_max( prob_distr )

	write_out( probs )

	print time() - start

def select_max( prob_distr ):

	probs = []

	for item in range(0, len(prob_distr)):
		_id = prob_distr[item][0]
		index = np.argmax( prob_distr[item][1] ) + 1

		probs.append([_id, index])

	return probs

def add_results( results, ids ):

	probs = []

	for _id in ids:
		x = np.zeros(7)
		for item in range(0, 10):
			if _id in results[item]:
				x = np.add( x, results[item][_id])
		probs.append([_id, x])

	return probs

def build_forests( training_set, test, features, label, test_idx ):

	results = {}

	for i in range(0, PARTS):
		print "Random Forest ", i
		train = training_set[ test_idx == i ]

		forest = train_data( train, features, label )
		probs = test_data( test, features, forest )

		results[i] = probs

	return results


def write_out( print_Me ):

	with open('fun.csv', 'wb') as csvfile:
		writer = csv.writer( csvfile, delimiter = ',')
		writer.writerow( ["Id","Cover_Type"])
		for row in range(0, len(print_Me)):
			writer.writerow(print_Me[row])

def assign_random( length ):

	return np.random.randint(0, PARTS, length)


def open_data( filename ):

	df = pd.read_csv( filename, header = 0)

	return df

def train_data( df, features, label ):

	forest = RandomForestClassifier( n_estimators = 1000, n_jobs = 2 )

	forest.fit( df[features], df[label] )

	return forest

def test_data( df, features, forest ):

	_ids = df['Id'].values

	classifier = forest.predict_proba( df[features] )

	return dict(zip(_ids, classifier))

main()