
Train & Test Your Own OpenCV Face Detector :

These files are setup to allow you to easily train and test your own face detector using OpenCV. 

INSTRUCTIONS:

#Training:

1. Move all positive images (containing faces) into the positive_images folder within the train directory. Make sure that all the images have the same dimensions and that the faces are located in the same part of the image.

2. Move all negative images into the negative_images folder within the train directory. Images should be as varied as possible and similar in dimensions.

3. Open the haartraining executable file in the train directory. If necessary, modify the arguments on line 25. The arguments provide information about the location of the faces in the positive images. The arguments follow the format below:

(number of faces in image, initial x coordinate for face, initial y coordinate for face, width of face in image, height of face in image)

You will also need to modify parameters for the opencv_traincascade function to specify how many positive and negative images you’re using for the training (see link for description of parameters: http://docs.opencv.org/2.4/doc/user_guide/ug_traincascade.html)

4. Close the haartraining file and run it from command line: ‘./haartraining’

5. Wait for the training to complete. Once it has completed, you will find cascade.xml has been created in the classifier directory within the train directory. This is your classifier - move this file into the trained_classifier folder within the test directory.

#Testing:

#On Images:

1. Move all the images you want to test into the input_images directory within the test directory.

2. From the command line run the cascades.py python script with the following command line arguments (in the given order): classifier name (without xml extension), scale factor (optional), sensitivity factor (optional)

3. Find the output images within the output_images directory

#On Webcam:

1. From command line, run the python script with 1 command line argument: classifier name (without xml extension)

(Troubleshooting tip: If the video is not detecting faces, try connecting your computer to a power source and running the script again)

Although throughout, we have referred to this as being face detection, if you were to substitute the face images in the positive_images folder with images of another object, you could build an object detector for a wide range of objects!
