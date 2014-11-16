from sklearn import linear_model

import pandas as pd
import numpy as np

import csv

ETA = 0.005
CYCLES = 1000
TRAIN = "sonar_train_noise.csv"
TEST = "sonar_test_noise.csv"

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


def train_data( training_tup, penalty ):

	train, features, label = training_tup

	percep = linear_model.Perceptron(penalty = penalty, fit_intercept = True, eta0 = ETA, n_iter = CYCLES)
	percep.fit( train[features], train[label])

	return percep

def test_data( test_tup, percep, penalty ):

	test, features, label = test_tup

	x = percep.score( test[features], test[label] )

	print "With penalty %s, accuracy is %.3f" %(penalty, x)

def main():

	print "Cycles are %d" %CYCLES
	print "ETA IS %.3f" %ETA

	training_tup = load_csv(TRAIN)
	test_tup = load_csv(TEST)

	penalties = ["None", "l1", "l2", "elasticnet"]

	for penalty in penalties:

		percep = train_data( training_tup, penalty )
		test_data( test_tup, percep, penalty )

main()