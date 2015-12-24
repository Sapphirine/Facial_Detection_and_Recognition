"""
===============================================================================
Generating multiple positive and negative samples for the model training
===============================================================================

Making use of the text files specifying the facial keypoints coordinates for 
the FERET database, as well as random background images contained in the 
CalTech dataset, this code generates multiple face and non-face examples.

The output is then used to fit the facial detection models and algorithms.


Justine Elizabeth Morgan (jem2268)
Stamatios Paterakis (sp3290)
Lauren Nicole Valdivia (lnv2107)
Team ID: 201512-53

Columbia University E6893 Big Data Analytics Fall 2015 Final Project

"""

import numpy as np
import os, sys, scipy
import math
from PIL import Image, ImageDraw, ImageOps
from skimage import exposure
from skimage.transform import pyramid_gaussian
import random

##############################################################################
# Parameters to be used as input by user
##############################################################################

# Location of the FERET set containing the images and text files w/ keypoints
PositiveDir = './NN_faces/positive'
# Folder with non-face images (e.g. the CalTech dataset)
NegativeDir = './NN_faces/negative'
# Path where the cropped images are to be saved (i.e. the training sample)
TrainSet = './CNN_training'
# Scan window size
w, h = 20, 20 

# Fuction loading the facial keypoint coordinates from the FERET database txt files
def features_coordinates(txtDir):

	with open (txtDir) as file:
	
		for line in file:
	
			if 'left_eye_coordinates=' in line:
	
				s = line.split('=')
				left_eye_coords = map(int, s[1].split())
	
			if 'right_eye_coordinates' in line:
	
				s = line.split('=')
				right_eye_coords = map(int, s[1].split())
	
			if 'nose_coordinates=' in line:
	
				s = line.split('=')
				nose_coords = map(int, s[1].split())
	
			if 'mouth_coordinates=' in line:
	
				s = line.split('=')
				mouth_coords = map(int, s[1].split())
	
	return left_eye_coords, right_eye_coords, nose_coords, mouth_coords	


print 'Generating the faces training sample...'
# Generate training set for faces
for root, dirs, files in os.walk(PositiveDir): 

    for name in files:

        ext = ['.jpg', '.jpeg', '.gif', '.png']
        
        if name.endswith(tuple(ext)):
		
			positive = False
			path = os.path.join(root,name)
			foo = path.split('.')
			txtDir = foo[0] + '.txt'
			features = features_coordinates(txtDir)
			left_eye_x, left_eye_y = features[0]
			right_eye_x, right_eye_y = features[1]
			nose_x, nose_y = features[2]
			mouth_x, mouth_y = features[3]
			
			image = Image.open(path).convert('L')  # Load image & convert to grayscale
			image = ImageOps.equalize(image)  # Histogram equalization
			max_height, max_width = np.array(image).shape
			x_metric = left_eye_x-right_eye_x  # between eyes distance
			y_metric = mouth_y-(left_eye_y+right_eye_y)/2  # distance between mouth and eyes
			
			w_star, h_star = 2*x_metric, 2*y_metric	
			x_star, y_star = (right_eye_x-0.5*x_metric), (right_eye_y-0.5*y_metric)
			

			# Pyramid scaling
			scale_factor = 1.1  # downscaling factor 
			pyramid = tuple(pyramid_gaussian(image, downscale=scale_factor))			
			correction = 1
			
			for p in xrange(len(pyramid)):

				img_scaled = Image.fromarray(pyramid[p])
				
				# Adjust coordinates and size to scale
				max_height /= correction
				max_width /= correction
				x_star /= correction; y_star /= correction; w_star /= correction; h_star /= correction; nose_x /= correction; nose_y /= correction
				
				# Image scanning
				x, y = 0, 0  # Scanning coordinates
				step = 3  # Pixel step
				dist = 10^6  # proximity to center (initialize to +oo)
				
				while y <= max_height - h:
				
					while x <= max_width - w:
				
						rectangle = (int(x), int(y), int(x+w), int(y+h))  
						scan_window = img_scaled.crop(rectangle)
						center = (x+w)/2, (y+h)/2
						
						if (x <= x_star and x + w >= x_star + w_star and \
						    y <= y_star and y + h >= y_star + h_star and \
						    w >= w_star and h >= h_star):
							
							if math.sqrt((nose_x-center[0])**2 + (nose_y-center[1])**2) < dist:
							
								face = scan_window
								dist = math.sqrt((nose_x-center[0])**2 + (nose_y-center[1])**2)
								positive = True
						
						x += step  # move to the left
					
					x = 0  # reset x axis 
					y += step  # and move down	
				
				if positive:
				
					scipy.misc.imsave(os.path.join(TrainSet, 'face', \
                    		'scl'+str(p)+'_'+str(x)+'_'+str(y)+'_'+name), \
							np.asarray(face, dtype='float64'))		
					break	

				# correction factor to bring face coordinates to scale
				correction = scale_factor	

print 'Done with faces, now generating non-face sample...'
# Create a training set for non-faces
for root, dirs, files in os.walk(NegativeDir): 

    for name in files:
    
        ext = ['.jpg', '.jpeg', '.gif', '.png']
    
        if name.endswith(tuple(ext)):
	
			path = os.path.join(root,name)
			image = Image.open(path).convert('L')  # Load image & convert to grayscale
			max_height, max_width = np.array(image).shape
			
			# Pyramid scaling
			scale_factor = 2  # downscaling factor 
			pyramid = tuple(pyramid_gaussian(image, downscale=scale_factor))			
			correction = 1
			
			for p in range(5):
				
				img_scaled = Image.fromarray(pyramid[p])
				# Adjust coordinates and size to scale
				max_height /= correction
				max_width /= correction
				
				if max_width >= w and max_height >= h:
				
					# Store random snapshots from the nonface image
					for i in range(15):
				
							x, y = random.uniform(0, max_width - w), random.uniform(0, max_height - h) 
							rectangle = (int(x), int(y), int(x+w), int(y+h))  
							scan_window = img_scaled.crop(rectangle)
							scipy.misc.imsave(os.path.join(TrainSet, 'nonface', \
											'scl'+str(p)+'_'+str(x)+'_'+str(y)+'_'+name), \
											np.asarray(scan_window, dtype='float64'))	
				correction = scale_factor					