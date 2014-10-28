import pandas as pd
import numpy as np

import csv
import sys

# Inputs: Command line arguments
# Ouputs: Returns the training step.  If no training step is found
#         defaults to 0.001.  If a negative number is input, an
#		  exception is raised.

def find_eta(args):

	for item in range(0, len(args)):
		if args[item] == "-eta" and item+1 < len(args):
			num = float(args[item+1])
			if num > 0:
				return num
			else:
				warning_message = """
				Enter a positive real number for eta parameter
				"""
				raise Exception(warning_message)

	warning_message = """
	No eta flag found.  Default training step selected as 0.001
	"""
	print warning_message
	return 0.001
 
# Inputs: Accepts the command line argument as an array and a flag
#         to find
# Outputs: Returns the name of the file as a string where data is
#          contained.  Raises exception if flag isn't found.

def find_filename(args, name):

	test = "-" + name

	for item in range(0, len(args)):
		if args[item] == test and item+1 < len(args):
			return str(args[item+1])

	warning_message = """
	Please enter valid %sing file name.
	""" %(name)
	raise Exception(warning_message)

# Inputs: Takes a string as a file name to be read in.
# Outputs: Returns a tuple containing a pandas dataframe and the names
#          of features and labels.

def load_csv( infile ):

	df = pd.read_csv(infile)

	infile_1 = open(infile)
	header = infile_1.next()
	infile_1.close()

	return (df, header)


def train_data( initialization, ):
	sgn = lambda x: (x > 0) - (x < 0)

	weights = []

	return weights

def main(args):

	initialization = 500

	eta = find_eta(args)

	training_file = find_filename(args, "train")
	training_tup = load_csv(training_file)

	test_file = find_filename(args, "test")
	test_tup = load_csv(test_file)

	weights = train_data( initialization )

main(sys.argv)