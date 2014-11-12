import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

ALPHA_ = 1
BETA_ = 1

def main():

	dataA = {'total' : 1000., 'conversions' : 380}
	dataB = {'total' : 1000., 'conversions' : 370}

	sum_statistics( dataA, dataB )
	setup_disc_plots( dataA, dataB )

	samplesA, samplesB = sample_posteriors( dataA, dataB )

	discrete_samples( samplesA, samplesB )

	continuous_prob( samplesA, samplesB )


def sum_statistics( dataA, dataB ):

	print "Group A's Conversions : %d" %(dataA['conversions'])
	print "Group A's Total : %d" %(dataA['total'])
	print "Group B's Conversions : %d" %(dataB['conversions'])
	print "Group B's Total : %d" %(dataB['total'])

	print "\nNaive Estimate"
	print "Group A's Conversion Rate : %.3f" %(dataA['conversions']/dataA['total'])
	print "Group A's Conversion Rate : %.3f" %(dataB['conversions']/dataB['total'])


def setup_disc_plots( dataA, dataB ):

	# Creates the x-axis for plots
	x = np.linspace(0.2, 0.5, 200)


	plot_posteriors(x, dataA, 'A')
	plot_posteriors(x, dataB, 'B')

	plt.savefig('discrete_chance.jpg')
	plt.close()


def plot_posteriors( x, data, label ):

	y = beta.pdf( x, ALPHA_ + data['conversions'], BETA_ + data['total'] - data['conversions'] )

	plt.plot( x, y, label = label )
	plt.legend()


def sample_posteriors( dataA, dataB ):

	samplesA = beta.rvs( ALPHA_ + dataA['conversions'],
						 BETA_ + dataA['total'] - dataA['conversions'],
						 size = 1000)

	samplesB = beta.rvs( ALPHA_ + dataB['conversions'],
						 BETA_ + dataB['total'] - dataB['conversions'],
						 size = 1000)

	return samplesA, samplesB

def discrete_samples( samplesA, samplesB ):

	a_g_b = (samplesA > samplesB).mean() * 100
	b_g_a = (samplesB > samplesA).mean() * 100
 
	print a_g_b, " percent chance group A is greater than B"
	print b_g_a, " percent chance group B is greater than A"

def continuous_prob( samplesA, samplesB ):

	increase = samplesA - samplesB
	plt.hist( increase, bins = 25 )
	plt.savefig('continuous_chance.jpg')

	median = np.percentile(increase, 50)

	print "\nGroup A is larger than Group B by a median of %.3f" %(median)

main()

