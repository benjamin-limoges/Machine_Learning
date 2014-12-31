from DBSCAN import DBSCAN
from dataset import dataset

def main():
	
	data = dataset( "vertebrate_train_nonoise.csv")
	data.normalize()

	x = DBSCAN( data , 2, 3 )
	x.train_DBSCAN()


main()