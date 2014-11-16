from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd

import csv

TRAIN = "train.csv"
TEST = "test.csv"

ETA = 0.001
CYCLES = 1000
PARTS = 10
SEED = 1879

def main():

	df, features, label = load_csv( TRAIN )
	test_idx = assign_random( len(df) )

	run_perceptron( df, features, label, test_idx )
	run_forest( df, features, label, test_idx )


def run_perceptron( df, features, label, test_idx ):

	penalties = ["None", "l1", "l2", "elasticnet"]

	print "Testing Multi-class Perceptron:"
	for penalty in penalties:
		overall_accuracy = []
		for i in range(0, 10):

			train = (df[test_idx == i], features, label)
			test = (df[test_idx != i], features, label)

			percep = train_data_perceptron( train, penalty )
			overall_accuracy.append(test_data_perceptron( test, percep, penalty ))

		print "With penalty %s, overall accuracy is %.3f" %(penalty, sum(overall_accuracy)/len(overall_accuracy))


def run_forest( df, features, label, test_idx ):

	tree_size = [10, 100, 1000, 10000]

	overall_accuracy = []
	print "\nTesting Random Forest:"

	for size in tree_size:
		for i in range(0,10):
			train = (df[test_idx == i], features, label)
			test = (df[test_idx != i], features, label)

			forest = train_data_forest( train, size )
			acc = test_data_forest( test, forest )

			overall_accuracy.append(acc)
 
		print "Overall Accuracy for %d trees is %.3f" %(size, sum(overall_accuracy)/len(overall_accuracy))

def train_data_forest( tup, size ):

	df, features, label = tup

	forest = RandomForestClassifier(n_estimators = size, n_jobs = 2)
	forest.fit( df[features], df[label] )

	return forest

def test_data_forest( tup, forest ):

	df, features, label = tup

	x = forest.score( df[features], df[label] )

	return x


def train_data_perceptron( tup, penalty):

	df, features, label = tup

	percep = linear_model.Perceptron( penalty = penalty, fit_intercept = True, eta0 = ETA, n_iter = CYCLES, n_jobs = 2 )
	percep.fit( df[features], df[label] )

	return percep

def test_data_perceptron( tup, percep, penalty ):

	df, features, label = tup

	x = percep.score( df[features], df[label] )

	return x


def assign_random( length ):

	np.random.seed( seed = SEED )
	test_idx = np.random.randint( 0, PARTS, length )

	return test_idx

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

main()