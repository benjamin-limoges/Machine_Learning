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
		label = features.pop(0)

		return features, label

	def train_forest(self):
		self.forest = RandomForestClassifier( n_estimators = self.trees, bootstrap = False, max_features = 'sqrt')
		self.forest.fit( self.train[self.features], self.train[self.label] )
		return self.forest

	def score_forest(self):
		print self.forest.score( self.test[self.features], self.test[self.label] )

	def predict_test(self):
		return self.forest.predict( self.test[self.features] )
def main():

	train = pd.read_csv( "train.csv", header = 0 )
	test = pd.read_csv( "test.csv", header = 0 )

	forest = RF( train, test, 1000 )

	predictions = forest.predict_test()

	with open("output_rf.csv", 'wb') as outfile:
		writer = csv.writer( outfile, lineterminator='\n' )
		writer.writerow(["ImageID", "Label"])
		for val in range(0,len(predictions)):
			writer.writerow([ val + 1, predictions[val]])

main()