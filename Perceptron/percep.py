from sklearn import linear_model

import pandas as pd
import numpy as np

import csv

ETA = 0.001
INITIALIZATION = 500
TRAIN = "sonar_train_nonoise.csv"
TEST = "sonar_test_nonoise.csv"

# Inputs: Takes a string as a file name to be read in.
# Outputs: Returns a tuple containing a pandas dataframe and the names
#          of features and labels.

def load_csv( infile ):

	df = pd.read_csv(infile)

	infile_1 = open(infile)
	header = infile_1.next()
	infile_1.close

	header = header.strip('\n')
	header = header.split(',')

	label = header.pop()
	label = label.rstrip()
	return (df, header, label)


def train_data( training_tup ):

	train, features, label = training_tup

	percep = linear_model.Perceptron(penalty = None, fit_intercept = True, eta0 = ETA, n_iter = INITIALIZATION)
	percep.fit( train[features], train[label])

	return percep

def test_data( test_tup, percep ):

	test, features, label = test_tup

	print percep.score( test[features], test[label] )*100, "%"


def main():

	training_tup = load_csv(TRAIN)

	test_tup = load_csv(TEST)

	percep = train_data( training_tup )

	test_data( test_tup, percep )

main()