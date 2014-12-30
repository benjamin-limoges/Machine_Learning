import csv

class dataset:

	def __init__(self, train_file):
		self.data = self.load_data(train_file)

	def load_data(self, train_file):

		data = []

		with open( train_file, 'rb' ) as csvfile:
			reader = csv.reader(csvfile, delimiter = ',')
			reader.next()
			for row in reader:
				label = row.pop()
				x = row
				for item in range(0, len(row)):
					x[item] = float(x[item])
				data.append({
						 'label' : label,
						 'point' : x,
						 'mark'  : 0 })

		return data

	def normalize(self):
		
		features = len(self.data[0]['point'])
		print features

		for feat in range(0, features):
			x = 0
			x2 = 0
			for row in range(0, len(self.data)):
				x += self.data[row]['point'][feat]
				x2 += self.data[row]['point'][feat]**2
			mean = float(x) / float(len(self.data))
			stdev = ((float(x2) / float(len(self.data))) - (mean**2))**0.5
			
			for row in range(0, len(self.data)):
				if stdev == 0:
					self.data[row]['point'][feat] = 0
				else:
					self.data[row]['point'][feat] = (self.data[row]['point'][feat] - mean) / stdev

	def print_data(self):

		return self.data