import numpy as np
from tensorflow import keras as ks
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications import VGG16, VGG19, ResNet50
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras import backend as K
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.utils.multiclass import unique_labels
from skimage.io import imread, imsave
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import figure
from matplotlib.ticker import MaxNLocator
from styleTransfer import generateImageFromGramMatrix, generateImageFromStyle
import os
import asyncio

TRAIN = 'train_euler'
VGG_19_WEIGHTS_PATH = "vggnormalized.h5"
ORIGINAL = "orig_euler_smooth"
classes = ['300.9', '600.3', '600.9']
classes_all = ['300.3', '300.9', '600.3', '600.9']
CLASS_LABELS = ['M1', 'M2', 'M3']
TRAINING_SIZE_SMALL = 160
imgDir = 'train_euler'
CLASSES_AL = ['al']
AL_DIR = 'al'
TRAINING_SIZE = 1800
BATCH_SIZE = 20
IMAGE_SIZE = (448, 448)
N_PC = 5
COLORS = [(0.2, 0.2, 1), (0.2, 1, 0.2), (1, 0.2, 0.2), (0, 0, 0.5), (0, 0.5, 0), (0.5, 0, 0)]
CONTENT_WEIGHT = 0
TOTAL_VARIATION_WEIGHT = 7.5
STYLE_WEIGHT = 1.0
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
			if currWindow.shape != IMAGE_SIZE + (3,):
				continue
			currImagePath = os.path.join(classPath, str(counterImages)) + '.png'
			cv2.imwrite(currImagePath, currWindow)
			counterImages += 1


def deprocess_image(x):
    # if K.image_data_format() == "channels_first":
    #     x = x.reshape((3, img_width, img_height))
    #     x = x.transpose((1, 2, 0))
    # else:
    #     x = x.reshape((img_width, img_height, 3))

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # BGR -> RGB
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def gram_matrix_batch(x):
    features = K.reshape(x, shape=(K.shape(x)[0], x.shape[1] * x.shape[2], x.shape[3]))
    features_T = K.permute_dimensions(features, (0, 2, 1))
    gram = K.batch_dot(features_T, features)
    gram = K.batch_flatten(gram)
    return gram


def get_gram_batch(data, functor):
    n_split = TRAINING_SIZE / BATCH_SIZE
    spliltted = np.array_split(data, n_split)
    results = []
    for batch in spliltted:
        batch_result = functor(batch)
        results.append(np.concatenate(batch_result, axis=1))
    features = np.concatenate(results, axis=0)
    return features
    
"""
Helper function to evaluate different dimensions of PCA with different type of SVM
"""

async def evaluateSVM_PCA(dimension, data, labels, SVM_type):
	pca = PCA(n_components=dimension)
	pcaFeatures = pca.fit_transform(data)
	X_train, X_test, y_train, y_test = train_test_split(pcaFeatures, labels, shuffle=False, test_size = 0.2)
	if SVM_type == 'linear':
		model = svm.LinearSVC(max_iter=10000)
	elif SVM_type == 'poly':
		model = svm.SVC(kernel='poly', degree=3, gamma='auto')
	elif SVM_type == 'rbf':
		model = svm.SVC(kernel='rbf', gamma='auto')
		
	model.fit(X_train, y_train)
	score = model.score(X_test, y_test)
	
	return score


def drawThingsOnReducedDomain(pcaFeatures, labels, things, legends, fileName, colors, sizes, markers, alphas):
    fig, ax = plt.subplots()
    counter = 1
    for i in range(3):
        idx = np.where(labels == i)
        ax.scatter(pcaFeatures[idx[0], 0], pcaFeatures[idx[0], 1], s=4, c=[COLORS[i]])
    
    for i in range(len(things)):
        ax.scatter(things[i][:, 0], things[i][:, 1], s=sizes[i], marker=markers[i], c=colors[i], alpha=alphas[i])
        counter += 1
    
    lgd = ax.legend([*CLASS_LABELS, *legends], bbox_to_anchor=(1.25, 0.5), loc='center right', borderaxespad=0, frameon=False)
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')

    plt.savefig(fileName, dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def drawThingsOnReducedDomain3D(pcaFeatures, labels, things, legends, fileName, colors, sizes, markers, alphas):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=60)

    counter = 1
    for i in range(3):
        idx = np.where(labels == i)
        ax.scatter(pcaFeatures[idx[0], 0], pcaFeatures[idx[0], 1], pcaFeatures[idx[0], 2], s=4, c=[COLORS[i]], alpha=0.3)
    
    for i in range(len(things)):
        ax.scatter(things[i][:, 0], things[i][:, 1], things[i][:, 2], s=sizes[i], marker=markers[i], c=colors[i], alpha=alphas[i])
        counter += 1
    
    lgd = ax.legend([*CLASS_LABELS, *legends], bbox_to_anchor=(1.25, 0.5), loc='center right', borderaxespad=0, frameon=False)
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')

    plt.savefig(fileName, dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def generateFromReducedDomain(features, pca, network=None):
    STYLE_WEIGHT = 1.0
    VAR_WEIGHT = 1.0
    gram_mat = pca.inverse_transform(features)
    generated = generateImageFromGramMatrix(gram_mat, H=IMAGE_SIZE[0], W=IMAGE_SIZE[1], C=3, S_weight=STYLE_WEIGHT, V_weight=VAR_WEIGHT, iters=20, vgg19=network)
    return deprocess_image(generated)

def projectImage(newImg, pca, pcaFeatures):

    newFeatures = get_gram_batch(newImg[np.newaxis, :, :])
    newPCAFeatures = pca.transform(newFeatures)
    return newPCAFeatures
    # fig = plt.figure()
    # fig.add_subplot()

    # counter = 1
    # for i in range(3):
    #     idx = np.where(labels == i)
    #     plt.scatter(pcaFeatures[idx, 0], pcaFeatures[idx, 1], s=4)

    # plt.scatter(newPCAFeatures[0, 0], newPCAFeatures[0, 1], s=400, marker='x')
    # plt.legend([*classes, 'New Data'], loc='lower right')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.show()


def correcoeHist(imgA, imgB):
    histA_R = cv2.calcHist(imgA, [0],None, [256], [0,256])
    histA_G = cv2.calcHist(imgA, [1],None, [256], [0,256])
    histA_B = cv2.calcHist(imgA, [2],None, [256], [0,256])

    histB_R = cv2.calcHist(imgB, [0],None, [256], [0,256])
    histB_G = cv2.calcHist(imgB, [1],None, [256], [0,256])
    histB_B = cv2.calcHist(imgB, [2],None, [256], [0,256])
    
    histA = np.concatenate([histA_R, histA_G, histA_B], axis=0)
    histB = np.concatenate([histB_R, histB_G, histB_B], axis=0)
    # return cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)
    
    # histA = cv2.calcHist([imgA], [0, 1, 2], None, [256] * 3, [0,256] * 3)
    # histB = cv2.calcHist([imgB], [0, 1, 2], None, [256] * 3, [0,256] * 3)
    
    return [cv2.compareHist(histA_R, histB_R, cv2.HISTCMP_CORREL), cv2.compareHist(histA_G, histB_G, cv2.HISTCMP_CORREL), cv2.compareHist(histA_B, histB_B, cv2.HISTCMP_CORREL)]


def generateInBetween(imgA, imgB, pca):
    STYLE_WEIGHT = 1.0
    VAR_WEIGHT = 5.0
    featuresAB_full = get_gram_batch(np.stack([imgA, imgB], axis=0))
    featuresAB = pca.transform(featuresAB_full)
    midpoint = np.sum(featuresAB, axis=0) / 2.0
    gram_mat_mid = pca.inverse_transform(midpoint)
    generated = generateImageFromGramMatrix(gram_mat_mid, H=IMAGE_SIZE[0], W=IMAGE_SIZE[1], C=3, S_weight=STYLE_WEIGHT, V_weight=VAR_WEIGHT, iters=20)
    return generated, midpoint



def generateImageForEachPCs(startPoint, nSteps, stepSize, pca, network=None):
    generatedImages = []
    points = []
    
    for i in range(4):
        generatePoint = np.copy(startPoint)
        PCvalues = np.linspace(startPoint[i] - nSteps * stepSize[i], startPoint[i] + nSteps * stepSize[i], num = nSteps * 2 + 1)
        print(PCvalues)
        for val in np.nditer(PCvalues):
            generatePoint[i] = val
            #generatedImages.append(generateFromReducedDomain(generatePoint, pca, network))
            points.append(np.copy(generatePoint[np.newaxis, :]))
    return generatedImages, points