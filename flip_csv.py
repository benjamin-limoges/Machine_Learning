import csv

infile = open('wine.csv')

data = []

for row in infile:
	data.append(row)

infile.close()
for item in range(0,len(data)):
	data[item] = data[item].split(',')
	for n in range(0, len(data[item])):
		data[item][n] = data[item][n].rstrip('\n')
	x = data[item].pop(0)
	data[item].append(x)

outfile = open('wine1.csv', 'w')
writer = csv.writer(outfile)

writer.writerows(data)
outfile.close()