#%%
import numpy as np
import cv2
import pickle
import keras
from keras.models import Sequential, Model
from sklearn.decomposition import PCA
from keras.applications import ResNet50
import matplotlib.pyplot as plt


# features = pickle.load(open("storedFeatures", "rb"))
# pca = PCA(n_components=50)
# pca.fit(features)

# img = cv2.imread("18.png")
# img = cv2.resize(img, (224, 224)) / 255.0
# network = ResNet50(weights="imagenet", include_top=False)
# feature = network.predict(img[np.newaxis, :]).reshape((1, -1))
generator = keras.models.load_model("generatorv2.hdf5")
noise = np.random.normal(0, 1, (10, 100))
#generated = generator.predict(pca.transform(feature))
#generator = ks.models.load_model("discriminator.hdf5")
generated = generator.predict(noise)
generated = 0.5 * generated + 0.5
generated *= 255.0
num = 0
for img in generated:
    cv2.imwrite("generated/gen" + str(num) + ".png", img)
    num+=1