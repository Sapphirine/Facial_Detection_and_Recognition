## Reformat the positives.txt file to include number of occurences and dimensions
## Script will be invoked by haartraining executable
##################################################################################

from PIL import Image
import sys

input_path = './positives_old.txt'
output_path = './positives.txt'

positives = open(output_path, 'a')

with open(input_path) as fileobj:
	for line in fileobj:
		info = ' '
		i = 1
		while i <= 5:
			info += (' ' + str(sys.argv[i]))		
			i += 1
		positives.write(line.strip() + info + '\n')
