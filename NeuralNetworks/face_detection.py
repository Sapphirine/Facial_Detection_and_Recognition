"""
===============================================================================
Face Detection 
===============================================================================

Detect faces in one or more images using a fitted convolutional neural network.
The hyperparameters and weights for the model are loaded from the output of  
"CNN_training.py"


Justine Elizabeth Morgan (jem2268)
Stamatios Paterakis (sp3290)
Lauren Nicole Valdivia (lnv2107)
Team ID: 201512-53

Columbia University E6893 Big Data Analytics Fall 2015 Final Project

"""

import numpy as np
import os, sys, scipy
import math
from PIL import Image, ImageDraw, ImageOps, ImageFont
from scipy import misc
from skimage import exposure
from skimage.transform import pyramid_gaussian
from sklearn.utils import shuffle
from lasagne import layers
from lasagne.nonlinearities import  softmax, rectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

##############################################################################
#  Parameters to be calibrated by user
##############################################################################

# Downscaling factor 
scale_factor = 1.2  
# Step of scanning window (in pixels)
step = 3
# Directory where image(s) are located
SourceDir = './itest'  
# If true, then aggressive merging of overlapping windows is performed
aggressive = False  
# Location where the trained CNN weights are saved at
CNN_Weights= './CNN_weights'

##############################################################################
# Functions used to merge overlapping identified faces
##############################################################################

# Boolean function that checks if two rectangles overlap (up to a certain percentage)
def overlap(rect_a, rect_b, pct):
    rect_a_left, rect_a_top, rect_a_right, rect_a_bottom = rect_a
    rect_b_left, rect_b_top, rect_b_right, rect_b_bottom = rect_b

    separate = rect_a_right < (1+pct)*rect_b_left or \
        	   rect_a_left > (1-pct)*rect_b_right or \
        	   rect_a_bottom < (1+pct)*rect_b_top or \
        	   rect_a_top > (1-pct)*rect_b_bottom

    return not separate   

def avg_rectangle(rectangles):
				sum_x_left = 0; sum_y_left = 0; sum_x_right = 0; sum_y_right = 0

				for r in len(rectangles):
					rect = rectangles[r]
					print rect
					rect_x_left = rect[0]; rect_y_left = rect[1]; rect_x_right = rect[2]; rect_y_right = rect[3]
					sum_x_left += rect_x_left
					sum_x_left += rect_x_left
					sum_x_left += rect_x_left
					sum_x_left += rect_x_left

				return sum_x_left/len(r), sum_x_right/len(r),sum_y_left/len(r), sum_y_right/len(r)

def same_element(l1, l2):
	same = False
	for elem in l1:
		if elem in l2:
			same = True
	return same

def merge(l1, l2):
	return l1 + list(set(l2) - set(l1)) #elements in l2 but not l1    

## Mild Merging of overlapping detections				
def mild_merging(rectangles, draw):	
	rectangles = np.asarray(rectangles)
	l = range(len(rectangles));  # list of indices for the face rectangles
	u = [0]*len(rectangles)
	groups = []

	for i in l:

		if u[i] == 0:

			u[i] = 1
			l_tmp = []
			l_tmp.append(l.index(i))

			for j in range(i,len(l)):

				if overlap(tuple(rectangles[l[i]]), tuple(rectangles[j]), 0) and i != j:
					
					l_tmp.append(l.index(j))
					u[j] = 1

			groups.append(l_tmp)

	for ind in range(len(groups)):

		if len(groups[ind]) > 2:
		
			merged_rect = np.mean(rectangles[groups[ind]], axis=0)	
			merged_rect = [ int(x) for x in merged_rect]
			draw.rectangle(merged_rect, outline = 'Chartreuse')	

## Aggressive Merging of overlapping detections
def aggr_merging(rectangles, draw):	
	rectangles = np.asarray(rectangles)

	##detect overlap
	l = range(len(rectangles))

	groups=[]
	for i in l:
		l_tmp = []
		l_tmp.append(l.index(i))

		for j in range(i,len(l)):

			if overlap(tuple(rectangles[l[i]]), tuple(rectangles[j]), 0) and i != j:
				l_tmp.append(l.index(j))
		groups.append(l_tmp)

	#merge groups containing the same elements	
	merged = []
	for g in groups:
		tmp  = g	
		for q in range(0, len(groups)):   
			if same_element(g,groups[q]):
				tmp = merge(g,groups[q])
				g = tmp
				groups[q] = []
		if len(tmp) > 0:
			merged.append(tmp)

	##convert back to rectangles		
	rect = []
	for g in merged:
		r_tmp = []
		for i in g:
			r_tmp.append(rectangles[l.index(i)])
		rect.append(r_tmp)
	avg_rect = []
	for m in rect:
		x0 = []; y0 = []; x1 = []; y1=[]
		for p in range(0,len(m)):
			n = m[p]
			x0.append(n[0])
			y0.append(n[1])
			x1.append(n[2])
			y1.append(n[3])
		tmp2 = [int(sum(x0)/float(len(m))), int(sum(y0)/float(len(m))), int(sum(x1)/float(len(m))), int(sum(y1)/float(len(m)))]
		avg_rect.append(tmp2)

	avg_rect = np.asarray(avg_rect)
	for y in avg_rect:
		x0 = y[0]
		y0 = y[1]
		x1 = y[2]
		y1 = y[3]
		merged_rect = (x0,y0,x1,y1) 
		draw.rectangle((merged_rect), outline = 'Chartreuse')

##############################################################################
# Loading the neural network and its weights
##############################################################################

w, h = 20, 20  # w x h sized window

net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv', layers.Conv2DLayer),
        ('pool', layers.MaxPool2DLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape = (None, 1, 20, 20),
    conv_num_filters = 32, conv_filter_size = (3, 3), 
    pool_pool_size = (2, 2),
	hidden_num_units = 50,
    output_num_units = 2, output_nonlinearity = softmax,

    update_learning_rate=0.01,
    update_momentum = 0.9,

    regression = False,
    max_epochs = 50,
    verbose = 1,
    )

net.load_params_from(CNN_Weights) 

##############################################################################
# Scan the entire image with a (w x h) window
##############################################################################

for root, dirs, files in os.walk(SourceDir): 
    for name in files:
        
        ext = ['.jpg', '.jpeg', '.gif', '.png']
        if name.endswith(tuple(ext)):

			path = os.path.join(root,name)
			orig_image = Image.open(path).convert('RGBA')
			image = orig_image.convert('L')  # Convert to grayscale
			image = ImageOps.equalize(image)  # Histogram equalization
			max_height, max_width = np.array(image).shape
			face_found = False
			rectangles = []

			# Pyramid scaling
			pyramid = tuple(pyramid_gaussian(image, downscale=scale_factor))			
			correction = 1

			for p in range(2, len(pyramid)-1): 

				img_scaled = Image.fromarray(pyramid[p])
				# Adjust coordinates and size to scale
				max_height /= correction
				max_width /= correction
				correction = scale_factor

				# Image scanning
				x, y = 0, 0  # Initialize scanning coordinates
				faces = []

				while y <= max_height - h:
					while x <= max_width - w:

						rectangle = (int(x), int(y), int(x+w), int(y+h))  
						scan_window = img_scaled.crop(rectangle)
						test_X = []
						test_X.append(np.expand_dims(np.asarray(scan_window, dtype='float64'), axis=0))
						
						if net.predict(test_X) == 1:
							face_found = True
							box = (int(x*correction**p), int(y*correction**p), \
								   int((x+w)*correction**p), int((y+h)*correction**p))	
							rectangles.append(box)  # store all the possible face rectangles in a list
						x += step  # move to the left
					x = 0  # reset x axis 
					y += step  # and move down	

			draw = ImageDraw.Draw(orig_image)

				
			if face_found:
				if aggressive:
					aggr_merging(rectangles, draw)
				else:
					mild_merging(rectangles, draw)	
			del draw
			scipy.misc.imsave(os.path.join('/Users/stamatiospaterakis/Desktop', 'testoutmild', \
											'scl'+str(p)+'_'+str(x)+'_'+str(y)+'_'+name), \
											np.asarray(orig_image, dtype='float64'))
			#orig_image.show()	