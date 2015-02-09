from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import pandas as pd 

import csv

PARTS = 10

class RF(object):

	def __init__(self, train, test, trees):
		self.trees = trees
		self.train = train
		self.test = test
		self.features, self.label = self.find_features()
		self.forest = self.train_forest()

	def find_features(self):
		features = self.train.columns.values.tolist()
		label = features.pop()

		return features, label

	def train_forest(self):
		self.forest = RandomForestClassifier( n_estimators = self.trees, bootstrap = False, max_features = 'sqrt')
		self.forest.fit( self.train[self.features], self.train[self.label] )
		return self.forest

	def score_forest(self):
		print self.forest.score( self.test[self.features], self.test[self.label] )

def main():

	data = pd.read_csv( "train.csv", header = 0 )
	test_idx = np.random.randint(0, 10, len(data))

	forests = []

	for i in range(PARTS):
		forests.append( RF( data[test_idx == i], data[test_idx != i], 100) )

	for i in range(PARTS):
		forests[i].score_forest()

if __name__ == '__main__':
	main()