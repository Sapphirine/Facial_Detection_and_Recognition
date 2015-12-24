Facial_Detection_and_Recognition

Recognition.py
This contains a script that performs facial recognition on the Labeled Faces in the Wild dataset.  It was adapted from the scikit-learn example located at http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html.  It performs principal components analysis for feature selection and dimension reduction followed by linear discriminant analysis for better class separation.  The script then uses an SVM with RBF kernel to classify the test set.
To run: python Recognition.py min_faces n_pca n_lda
min_faces is the minimum number of faces required for each class to be included in the algorithm.
n_pca is the number of eigenvectors desired.
n_lda is the numbe of LDA features desired.
