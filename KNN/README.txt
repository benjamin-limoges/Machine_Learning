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

Reads in the data.  Raises exception of neighbor_num is not an odd positive integer or if partitions is negative or 0.

CSV file should be in the same folder as knn.py - functionality to be build to allow it to be placed in a different folder.

CSV file must have labels at the end and not the beginning of each line.  If the file has them at the beginning, run "flip_csv.py" first.

Values should be continous and not discrete - my choice of distance metrics does not permit classifications as features.

This implements the kth nearest neighbor algorithm, using 3 different weighting schemes: uniform, inverse distance and log of distance.


1. Reads in command line arguments

2. Sets up KNN.  It does this by reading in the data, extracting the first line as a header, and randomly assigning each line a partition.  Returns a tuple containing the dataframe, the header row (minus the label's name), the label (or classification), and the randomly assigned partition numbers.

3. Builds KNN.  It does this without normalizing the data.