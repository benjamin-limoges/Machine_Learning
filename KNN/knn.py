from sklearn import neighbors
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import csv
import sys

# Returns an array that has the permutation number for each data
# point.
def assignRandom( length, part_len ):

	np.random.seed(seed = 1337)
	test_idx = np.random.randint( 0, part_len, length )

	return test_idx

# Returns a tuple with the dataframe, features list, label name,
# number of partitions (as an integer), and an array that labels
# the partition number for each data point.
def setupkNN( infile, part_len ):

	tup = loadData( infile )

	# Extracts the loaded data
	df = tup[0]
	features = tup[1]
	label = features.pop()

	# Assign fold number
	test_idx = assignRandom(len(df), part_len)

	return (df, features, label, test_idx)

# Tuple returned has a panadas dataframe in the 0th position
# In the first is the header of the csv
def loadData( infile ):

	# Sets the random seed
	np.random.seed(seed = 1337)

	# Reads data into the pandas dataframe
	df = pd.read_csv( infile )

	# Gets the names of the features, label
	infile_1 = open( infile )
	header = infile_1.next()
	infile_1.close()

	header = header.rstrip('\n')
	header = header.split(',')

	return (df, header)

# Implements the kth nearest neighbor algorithm and stores the results
# It calls the plots 
def kNN( tup, neighbor_num, part_len, infile, normalization ):
	df = tup[0]
	features = tup[1]
	label = tup[2]
	test_idx = tup[3]

	label = label.rstrip()

	results = {}
	w = ['uniform', 'distance', lambda x: np.log(x)]
	names = ['uniform', 'distance', 'log(x)']

	for item in range(0, len(names)):
		result = []
		results[names[item]] = result

	# Iterates through the number of neighbors permitted.  Odd values
	# only.  To get the user specified number of neighbors, 1 is added
	# so that number is encompassed in the range.
	for n in range(1, neighbor_num + 1, 2):
		for weight in range(0,len(w)):
			accuracy = []
			for part in range(0, part_len):
				train = df[test_idx != part]
				test = df[test_idx == part]

				knn = neighbors.KNeighborsClassifier(n_neighbors = n, weights = w[weight])
				knn.fit(train[features], train[label])
				preds = knn.predict(test[features])

				acc = (np.where(preds== test[label], 1, 0).sum() * 100)/ float(len(test))
				accuracy.append(acc)
			overall_accuracy = sum(accuracy) / float(len(accuracy))
			results[names[weight]].append([n, overall_accuracy])
			print "Neighbors: %d, Accuracy: %f, Weights: %s" %(n, overall_accuracy, names[weight])

	plots( results, infile, neighbor_num, normalization )

# Plots the results.  Lets matplotlib choose defaults
def plots( results, infile, neighbor_num, normalization ):

	result_uniform = pd.DataFrame(results['uniform'], columns = ['n', 'accuracy'])
	result_distance = pd.DataFrame(results['distance'], columns = ['n', 'accuracy'])
	results_log = pd.DataFrame(results['log(x)'], columns = ['n', 'accuracy'])

	plt.plot(result_uniform.n, result_uniform.accuracy, label = "Uniform")
	plt.plot(result_distance.n, result_distance.accuracy, label = "Distance")
	plt.plot(results_log.n, results_log.accuracy, label = "Log(x)")
	#plt.axis([1,neighbor_num, 60, 100])
	plt.xlabel('Varying Values of K')
	plt.ylabel('Accuracy')
	plt.xticks(result_uniform.n)
	title = "Accuracy of Kth Nearest Neighbor on " + infile[:-4].upper() + ", " + normalization
	plt.title(title)
	plt.legend()
	save_file = "accuracy_" + str(infile[:-4]) + "_" + normalization + ".png"
	plt.savefig(save_file)
	plt.close()

# Changes each of the columns except the label into z-score normalized
# random variables.
def z_score_normalize( tup ):

	df = tup[0]
	features = tup[1]
	label = tup[2]
	test_idx = tup[3]

	for col in features:
		df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)

	return(df, features, label, test_idx)


def main():
	if len(sys.argv) == 4:

		infile = sys.argv[1]
		neighbor_num = int(sys.argv[2])
		part_len = int(sys.argv[3])

		if neighbor_num < 0:
			warning_message = """
			Number of neighbors must be an odd postive integer
			"""
			raise Exception(warning_message)
		
		elif neighbor_num % 2 == 0:
			warning_message = """
			Number of neighbors must be an odd positive integer
			"""
			raise Exception(warning_message)

		elif part_len <= 0:
			warning_message = """
			Number of partitions must a positive integer
			"""
			raise Exception(warning_message)

		# Main control for the file
		tup = setupkNN(infile, part_len)
		kNN(tup, neighbor_num, part_len, infile, "Unnormalized")
		tup = z_score_normalize(tup)
		kNN(tup, neighbor_num, part_len, infile, "Normalized")

	else:
		
		warning_message = """ 
		Incorrect running format.\n 
		Please provide an input file, then an
		integer for number of neighbors, then
		an integer for number of partitions.
		"""
		raise Exception(warning_message)

main()