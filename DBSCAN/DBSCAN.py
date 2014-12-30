class DBSCAN:

	def __init__(self, data, eps, MinPts):
		self.data = data
		self.eps = eps
		self.MinPts = MinPts

	def train_DBSCAN(self):
		distances = self.create_distances()
		print distances

	def create_distances(self):
		distances = []

		for i in range(0, len(self.data.data)):
			distances.append([])
			for j in range(0, len(self.data.data)):
				print self.data.data[i], self.data.data[j]
				print self.find_distances(self.data.data[i], self.data.data[j])

	def find_distances(point1, point2):
		features = len(point1['point'])
		distance = 0

		for feat in range(0, features):
			distance += (point1['point'][feat] - point2['point'][feat])**2
		return distance**0.5