from DBSCAN import DBSCAN
from dataset import dataset

def main():
	
	data = dataset( "vertebrate_train_nonoise.csv")
	x = data.print_data()
	data.normalize()
	y = data.print_data()

	x = DBSCAN( data , 2, 3 )
	x.train_DBSCAN()


main()