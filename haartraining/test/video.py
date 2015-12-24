# REAL TIME FACE DETECTION
####################################################################################

import cv2, sys, os

#read in classifier name
cascade = sys.argv[1]

cascPath = os.getcwd() + 'trained_classifiers/' + cascade + '.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video = cv2.VideoCapture(0)

while True:

    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #convert to grayscale

    #detect the faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display rectangle around detected faces
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
video.release()
cv2.destroyAllWindows()