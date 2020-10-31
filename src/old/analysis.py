# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% [markdown]
"""Import modules"""

#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras as ks
import cv2
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications import ResNet50
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from matplotlib import pyplot as plt
import os

%matplotlib inline

#%% [markdown]
"""
some constants
"""
#%%
TRAIN = 'train2'
classes = ['300.3', '300.9', '600.3', '600.9']
imgDir = 'train2'
TRAINING_SIZE = 360

#%% [markdown]
"""
Create sliding window of the 4 original dataset
"""
#%% [markdown]
"""
two helper functions
"""
#%%
"""
read all images from a folder
"""

def load_images_from_folder(folder):
	images = []
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder,filename))
		if img is not None:
			images.append(img)
	return images


"""
write all possible sliding windows to disk
"""

def sliding_window(image, stepSize, windowSize, classPath):
	# slide a window across the image
	counterImages = 0
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			currWindow = image[y: y + windowSize[1], x: x + windowSize[0]]
			if currWindow.shape != (448, 448, 3):
				continue
			currImagePath = os.path.join(classPath, str(counterImages)) + '.png'
			cv2.imwrite(currImagePath, currWindow)
			counterImages += 1

#%% [markdown]
"""
start the sliding window generation process
"""
#%%

images = load_images_from_folder('orig')

counter = 0
for image in images:
	classPath = os.path.join(TRAIN, classes[counter])
	if not os.path.exists(classPath):
		os.mkdir(classPath)
	sliding_window(image, 3, (448, 448), classPath)
	counter+=1


#%% [markdown]
"""
Read generated sliding window and do data augmentation 
by randomly seleft and random flips
"""
#%%


data = None
labels = None
aug = ImageDataGenerator(
		horizontal_flip=True,
		vertical_flip=True,
		fill_mode="nearest")


augmentation = aug.flow_from_directory(imgDir, batch_size = TRAINING_SIZE, target_size=(224,224), save_to_dir="augmented", seed=10, class_mode="sparse")
print(augmentation.class_indices)
for x, y in augmentation:
	data = x
	labels = y
	break
data = data.astype(np.float32) / 255.0

#%% [markdown]
"""
Draw figure of original pics
"""
#%%
fig = plt.figure(figsize=(16, 8))
counter = 1
for subdir, dirs, files in os.walk('orig'):
	for file in files:
		origImg = cv2.imread(os.path.join(subdir, file))
		fig.add_subplot(4, 1, counter)
		plt.axis('off')
		plt.imshow(origImg)
		counter += 1
plt.savefig('original.png', transparent=True)
plt.show()

#%% [markdown]
"""
Figure X: The four original dataset. 
"""
#%% [markdown]
"""
Draw figure for augmented pics
"""
#%%
fig = plt.figure(figsize=(16, 16))

counter = 1
for i in range(4):
		idx = np.random.choice(np.where(labels == i)[0], 5)
		for j in idx:
			   fig.add_subplot(4, 5, counter)
			   plt.axis('off')
			   plt.imshow(data[j])
			   counter += 1
plt.savefig('sample.png', transparent=True)
plt.show()

#%% [markdown]
"""
Figure X: Some sample of augmented dataset
"""
#%% [markdown]
"""
extract features
"""
#%%

network = ResNet50(weights="imagenet", include_top=False)
features = network.predict(data)
features = features.reshape(TRAINING_SIZE, features.shape[1] * features.shape[2] * features.shape[3])

#%% [markdown]
"""
split train and test then fit SVM model with all dimensions (extremely slow)
"""
#%%

X_train, X_test, y_train, y_test = train_test_split(features, labels, shuffle=False, test_size = 0.2)

model = svm.LinearSVC(max_iter=9999999)
model.fit(X_train, y_train)
model.score(X_test, y_test)

#%% [markdown]
"""
Helper function to evaluate different dimensions of PCA with different type of SVM
"""
#%%
def evaluateSVM_PCA(dimensionRange, data, labels, SVM_type):
	accuracies = []
	for n in dimensionRange:
		#print('Accuracy of SVM with PCA with ' + str(n) + ' principal components:')
		pca = PCA(n_components=n)
		pcaFeatures = pca.fit_transform(data)
		X_train, X_test, y_train, y_test = train_test_split(pcaFeatures, labels, shuffle=False, test_size = 0.2)
		model = None
		if SVM_type == 'linear':
			model = svm.LinearSVC(max_iter=99999)
		elif SVM_type == 'poly':
			model = svm.SVC(kernel='poly', degree=3, gamma='auto')
		elif SVM_type == 'rbf':
			model = svm.SVC(kernel='rbf', gamma='auto')
		elif SVM_type == 'gaussian':
			pass
		
		model.fit(X_train, y_train)
		score = model.score(X_test, y_test)
		accuracies.append(score)
	return accuracies
		#print(score)

#%%
dimensions = [5, 10, 20, 50, 80, 100, 150, 200, 250, 300]
results = evaluateSVM_PCA(dimensions, features, labels, 'linear')

#%% [markdown]
"""
Evaluate and plot accuracies of different models
"""

#%%

accuracies = dict()
accuracies['LinearSVM'] = []
accuracies['3Poly'] = []
accuracies['RBF'] = []

#%%

accuracies['LinearSVM'] = evaluateSVM_PCA( range(2, 20), features, labels, 'linear')
accuracies['3Poly'] = evaluateSVM_PCA( range(2, 20), features, labels, 'poly')
accuracies['RBF'] = evaluateSVM_PCA(range(2, 20), features, labels, 'rbf')
#%%
pca = PCA(n_components=4)
pcaFeatures = pca.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(pcaFeatures, labels, shuffle=False, test_size=0.2)


#%% [markdown]
"""
draw the confusion matrix
"""
#%% [markdown]
"""
helper function to draw the confusion matrix
"""
#%%
def plot_confusion_matrix(y_true, y_pred, classes,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

#%% [markdown]
"""
Train the classifer with dimension=4 and evaluate the result
"""
#%%
model = svm.SVC(kernel='poly', degree=3, gamma='auto')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plot_confusion_matrix(y_test, y_pred, classes, 'Confusion matrix, without normalization')
plt.show()
#%% [markdown]
"""
run iterative k means to determine the number of clusters
"""
scores = []
kmeansAccuracies = []
for k in range(2, 20):
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(X_train)
	scores.append(kmeans.score(X_test))
	kmeans.predict(X_test)

#%% [markdown]
"""
visualize feature using t-SNE in 3D (possibily useless)
"""
#%%

pca = PCA(n_components=4)
pcaFeatures = pca.fit_transform(features)
X_embedded = TSNE(n_components=3).fit_transform(pcaFeatures)

#%%
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

counter = 1
for i in range(4):
		idx = np.where(labels == i)
		ax.scatter(X_embedded[idx, 0], X_embedded[idx, 1], X_embedded[idx, 2])

plt.legend(classes)
plt.show()
#%% [markdown]
"""
Figure X: Confusion matrix of the optimal model. 
"""

#%% [markdown]
"""
Visualize feature using t-SNE in 2D
"""
#%%
pca = PCA(n_components=4)
pcaFeatures = pca.fit_transform(features)
X_embedded = TSNE(n_components=2).fit_transform(pcaFeatures)

#%%
fig = plt.figure()
fig.add_subplot()

counter = 1
for i in range(4):
		idx = np.where(labels == i)
		plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], s=4)

plt.legend(classes)
plt.show()
#%% [markdown]
"""
Figure X: Using t-SNE to visualize the 4 dimensional features in 2D. 
"""
#%% [markdown]
"""
visualize feature using first two components of PCA
"""
#%%
fig = plt.figure()
fig.add_subplot()

counter = 1
for i in range(4):
		idx = np.where(labels == i)
		plt.scatter(pcaFeatures[idx, 0], pcaFeatures[idx, 1], s=4)

plt.legend(classes)
plt.show()

#%% [markdown]
"""
Figure X: Plot of first two princial components of the dataset 
"""
#%% [markdown]
"""
plot accuracies with different SVM and PCA components
"""

fig = plt.figure()
fig.add_subplot()
x = range(2,20)
plt.plot(x, accuracies['LinearSVM'])
plt.plot(x, accuracies['3Poly'])
plt.plot(x, accuracies['RBF'])
plt.legend(['LinearSVM', '3Degree', 'RBF'])
plt.show()

#%% [markdown]
"""
Figure X: Accuracy vs number of principal comnponents. 
Tested on three SVM models. 
A normal SVM model
with no kernel. A SVM model with 3 degree polynomial kernel 
and a SVM with radial basis kernel
"""

#%%
