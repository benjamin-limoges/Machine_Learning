from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from time import time

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

	start = time()

	df, features, label = load_csv( TRAIN )
	test_idx = assign_random( len(df) )

	run_perceptron( df, features, label, test_idx )
	run_logit( df, features, label, test_idx )
	run_adaboost( df, features, label, test_idx )
	run_forest( df, features, label, test_idx )

	print time() - start, " seconds"

def run_perceptron( df, features, label, test_idx ):

	penalties = ["None", "l1", "l2", "elasticnet"]

	print "Testing Multi-class Perceptron:"
	for penalty in penalties:
		overall_accuracy = []
		for i in range(0, 10):

			train = (df[test_idx == i], features, label)
			test = (df[test_idx != i], features, label)

			percep = train_data_perceptron( train, penalty )
			overall_accuracy.append(test_data_perceptron( test, percep ))

		print "With penalty %s, overall accuracy is %.3f" %(penalty, sum(overall_accuracy)/len(overall_accuracy))


def run_logit( df, features, label, test_idx ):

	penalties = ["l1", "l2"]

	print "\nTesting Logistic Regression:"
	for penalty in penalties:
		overall_accuracy = []
		for i in range(0, 10):

			train = (df[test_idx == i], features, label)
			test = (df[test_idx != i], features, label)

			logit = train_data_logit( train, penalty )
			overall_accuracy.append(test_data_logit( test, logit ))

		print "With penalty %s, overall accuracy is %.3f" %(penalty, sum(overall_accuracy)/len(overall_accuracy))

def run_adaboost( df, features, label, test_idx ):

	tree_size = [10, 100, 1000]

	print "\nTesting AdaBoost:"

	for size in tree_size:

		overall_accuracy = []

		for i in range(0,10):
			train = (df[test_idx == i], features, label)
			test = (df[test_idx != i], features, label)

			ada = train_data_ada( train, size )
			acc = test_data_ada( test, ada )

			overall_accuracy.append(acc)

		print "With %d trees, overall accuracy is %.3f" %(size, sum(overall_accuracy)/len(overall_accuracy))


def run_forest( df, features, label, test_idx ):

	tree_size = [10, 100, 1000]

	print "\nTesting Random Forest:"

	for size in tree_size:

		overall_accuracy = []
		
		for i in range(0,10):
			train = (df[test_idx == i], features, label)
			test = (df[test_idx != i], features, label)

			forest = train_data_forest( train, size )
			acc = test_data_forest( test, forest )

			overall_accuracy.append(acc)
 
		print "With %d trees, overall accuracy is %.3f" %(size, sum(overall_accuracy)/len(overall_accuracy))

def train_data_perceptron( tup, penalty ):

	df, features, label = tup

	percep = Perceptron( penalty = penalty, fit_intercept = True, eta0 = ETA, n_iter = CYCLES, n_jobs = 2 )
	percep.fit( df[features], df[label] )

	return percep

def test_data_perceptron( tup, percep ):

	df, features, label = tup

	x = percep.score( df[features], df[label] )

	return x

def train_data_logit( tup, penalty ):

	df, features, label = tup

	logit = LogisticRegression( penalty = penalty, fit_intercept = True)
	logit.fit( df[features], df[label] )

	return logit

def test_data_logit( tup, logit ):

	df, features, label = tup

	x = logit.score( df[features], df[label] )

	return x

def train_data_ada( tup, size ):

	df, features, label = tup

	ada = AdaBoostClassifier( n_estimators = size )
	ada.fit( df[features], df[label] )

	return ada

def test_data_ada( tup, ada ):

	df, features, label = tup

	x = ada.score( df[features], df[label] )

	return x

def train_data_forest( tup, size ):

	df, features, label = tup

	forest = RandomForestClassifier(n_estimators = size, n_jobs = 2)
	forest.fit( df[features], df[label] )

	return forest

def test_data_forest( tup, forest ):

	df, features, label = tup

	x = forest.score( df[features], df[label] )

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