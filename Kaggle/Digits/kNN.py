from sklearn.neighbors import KNeighborsClassifier

import numpy as np 
import pandas as pd 

import csv

from time import time

PARTS = 10
SEED = 13861930
class kNN(object):

	def __init__(self, train, test, k, weights):
		self.train = train
		self.test = test
		self.k = k
		self.weights = weights
		self.features, self.label = self.find_features()
		self.neighbors = self.train_neighbors()

	def find_features(self):
		features = self.train.columns.values.tolist()
		label = features.pop(0)
		return features, label

	def train_neighbors(self):
		neighbors = KNeighborsClassifier( n_neighbors = self.k, weights = self.weights ) 
		neighbors.fit( self.train[self.features], self.train[self.label] )
		return neighbors

	def score_neighbors(self):
		print self.neighbors.score( self.test[self.features], self.test[self.label] )

	def predict_test(self):
		return self.neighbors.predict( self.test[self.features] )

start = time()

np.random.seed( seed = SEED )
data = pd.read_csv( "train.csv", header = 0 )
test_idx = np.random.randint( 0, PARTS, len(data) )

train = data[test_idx == 0]
test = data[test_idx != 0]
#test = pd.read_csv("test.csv", header = 0)

weights = ["uniform", "distance"]

for k in range(1, 20, 2):
	for weight in weights:
		model = kNN( train, test, k, weight )
		print "Built model with %d neighbors and weights %s" %(k, weight)
		model.score_neighbors()

"""
predictions = model.predict_test().tolist()

with open("output_knn.csv", 'wb') as outfile:
	writer = csv.writer( outfile, lineterminator='\n' )
	writer.writerow(["ImageID", "Label"])
	for val in range(0,len(predictions)):
		writer.writerow([ val + 1, predictions[val]])

print time() - start, " seconds"
"""