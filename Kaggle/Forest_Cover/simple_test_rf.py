from sklearn.ensemble import RandomForestClassifier

import numpy as np 
import pandas as pd 

import csv

def main():

	train = open_data( "train.csv" )
	test = open_data( "test.csv" )

	features = train.columns.values.tolist()
	label = features.pop()

	forest = train_data( train, features, label )
	print_Me = test_data( test, features, forest )

	write_out( print_Me )

	#test_idx = np.random.randint( 0 , 10 , len(df) )

	#train = df[test_idx == 0]
	#test = df[test_idx != 0]

def write_out( print_Me ):

	with open('fun.csv', 'wb') as csvfile:
		writer = csv.writer( csvfile, delimiter = ',')
		writer.writerow( ["Id","Cover_Type"])
		for row in range(0, len(print_Me)):
			writer.writerow(print_Me[row])


def open_data( filename ):

	df = pd.read_csv( filename, header = 0)

	return df

def train_data( df, features, label ):

	forest = RandomForestClassifier( n_estimators = 1000, n_jobs = 2 )

	forest.fit( df[features], df[label] )

	return forest

def test_data( df, features, forest ):

	_ids = df['Id'].values

	classifier = forest.predict( df[features] )

	return zip(_ids, classifier)

main()