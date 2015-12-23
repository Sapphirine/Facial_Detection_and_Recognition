##to run: python Recognition.py min_faces n_pca n_lda
#i.e. python Recognition.py 80 50 9

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import sys

##Source :
##  http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html contains an example of
##  facial recognition that uses PCA with SVM.  That following was adapted from that example to perform LDA after PCA
##  for better class separability.

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)




if __name__=="__main__":


	min_faces=int(sys.argv[1])  #the number of images a class must have to be included
	n_components=int(sys.argv[2])  #the number of components to use for PCA
	n_lda=int(sys.argv[3])   ## the number of features to use in LDA

	people = fetch_lfw_people(min_faces_per_person=min_faces, resize=0.4)  #use the Labeled Faces in the Wild dataset

	n_pics, h, w = people.images.shape

	X = people.data


	y = people.target
	target_names = people.target_names
	n_classes = target_names.shape[0]

	print("Total dataset size:")
	print("n_samples: %d" % n_pics)
	print("n_classes: %d" % n_classes)


	# Split into a training set and a test set
	X_train, X_test, y_train, y_test = train_test_split(
	    X, y, test_size=0.25)
	
	# Perform PCA (eigenfaces) on the face dataset
	print("Extracting the top %d eigenfaces from %d faces"
	      % (n_components, X_train.shape[0]))
	t0 = time()
	pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
	print("done in %0.3fs" % (time() - t0))

	eigenfaces = pca.components_.reshape((n_components, h, w))

	print("Projecting the input data on the eigenfaces orthonormal basis")
	t0 = time()
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	print("done in %0.3fs" % (time() - t0))

	# Perform LDA (fisherfaces) on the PCA-projected data for better separability
	lda=LDA(n_components=n_lda).fit(X_train_pca,y_train)
	X_train_lda=lda.transform(X_train_pca)
	X_test_lda=lda.transform(X_test_pca)

	


	# Train an SVM with RBF Kernel
	print("Fitting the classifier to the training set")
	t0 = time()
	param_grid = {'C': [1,1e1,1e2,1e3, 5e3, 1e4, 5e4, 1e5],
	              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

	clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
	clf = clf.fit(X_train_lda, y_train)
	print("done in %0.3fs" % (time() - t0))
	print("Best estimator found by grid search:")
	print(clf.best_estimator_)


	# Test performance
	print("Predicting people's names on the test set")
	t0 = time()
	y_pred = clf.predict(X_test_lda)
	print("done in %0.3fs" % (time() - t0))
	print(classification_report(y_test, y_pred, target_names=target_names))
	print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


	
	# Visualize the eigenfaces and a subset of the test results.
	prediction_titles = [title(y_pred, y_test, target_names, i)
	                     for i in range(y_pred.shape[0])]

	fig1=plt.figure()
	plot_gallery(X_test, prediction_titles, h, w)
	plt.savefig('Prediction.png')


	eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
	fig2=plt.figure()
	plot_gallery(eigenfaces, eigenface_titles, h, w)

	plt.savefig('Eigenfaces.png')











