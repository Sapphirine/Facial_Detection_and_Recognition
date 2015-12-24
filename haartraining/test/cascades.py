####################################################################################
#DETECT FACES IN STILL IMAGES
####################################################################################
import numpy as np
import cv2
import sys, os

#read in command line arguments specifying classifier, scale, sensitivity
cascade = str(sys.argv[1])
scale, sens = float(sys.argv[2]), int(sys.argv[3])

#default scale/sensitivity
if len(sys.argv) < 4:
    scale, sens = 1.3, 3

#directories
inputDir = './input_images/'
outputDir = './output_images/'
cascadeDir = './trained_classifiers/'

#iterate through all the image files in the input directory
for dirs, root, files in os.walk(inputDir):

	for name in files:
		name = str(name)
		if name.endswith('.jpg' or '.jpeg' or '.gif' or '.png'):
			print name

			#create the detector
			face_cascade = cv2.CascadeClassifier(cascadeDir + cascade + '.xml')

			#import image and convert to grayscale
			img = cv2.imread(inputDir + name)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			#detect faces
			faces = face_cascade.detectMultiScale(gray, scale, sens)
			face_cnt = 0

			#draw detection rectangle around faces
			for (x,y,w,h) in faces:
			    face_cnt+=1
			    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			    roi_gray = gray[y:y+h, x:x+w]
			    roi_color = img[y:y+h, x:x+w]

			#output information about detected faces to a text file
			output_path = 'results.txt'
			results = open(output_path, 'a')
			info = str(scale) + ' ' + str(sens) + ' ' + str(face_cnt)
			results.write(name.strip() + info + '\n')
			results.close()

			#save the images (uncomment the other lines of code to display images as program runs)

			#cv2.imshow('Faces',img)
			cv2.imwrite(outputDir + name + '_' + str(scale) + '_' + str(sens) + '.jpg', img)
			#cv2.waitKey(0)
			#cv2.destroyAllWindows()