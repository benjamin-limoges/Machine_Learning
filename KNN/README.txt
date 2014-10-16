Author: Benjamin Limoges
Date: 10/15/2014

***********
How to Run
***********

python knn.py <filename.csv> neighbor_num partitions

Where:

<filename.csv> is a well specified CSV file with the last 

neighbor_num is the maximum value of k to be run. neighbor_num should be an odd positive integer.

partition is the number of partitions - the algorithm will split the dataset into partitions.

***********
Logic
***********



This implements the kth nearest neighbor algorithm.