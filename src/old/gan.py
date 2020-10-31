#%%

import numpy as np
import tensorflow as tf
import keras as ks
import cv2
import pickle

from sklearn.decomposition import PCA
#from keras.datasets import mnist
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys
import numpy as np

#%%

class DCGAN():

    def __init__(self, load, epoch):

        # Input shape

        self.img_rows = 224
        self.img_cols = 224
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 50
        optimizer = Adam(0.0002, 0.5)
        self.epoch = epoch

        # Build and compile the discriminator
        if not load:

            self.discriminator = self.build_discriminator_ResNet()
            self.discriminator.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
            
            # Build the generator
            self.generator = self.build_generator()
            # The generator takes noise as input and generates imgs
            z = layers.Input(shape=(self.latent_dim,))
            img = self.generator(z)
            
            # For the combined model we will only train the generator
            # The discriminator takes generated images as input and determines validity
            self.discriminator.trainable = False
            valid = self.discriminator(img)
            # The combined model  (stacked generator and discriminator)
            # Trains the generator to fool the discriminator
            self.combined = Model(z, valid)
            self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
            self.combined.summary()
        else:
            self.discriminator = ks.models.load_model("discriminator.hdf5")
            self.discriminator.layers[0].trainable = False
            self.discriminator.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
            self.generator = ks.models.load_model("generator.hdf5")
            z = layers.Input(shape=(self.latent_dim,))
            img = self.generator(z)
            self.discriminator.summary()
            self.generator.summary()
            self.discriminator.trainable = False
            valid = self.discriminator(img)
            self.combined = Model(z, valid)
            #self.combined = ks.models.load_model("combined.hdf5")
            self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

            self.discriminator.summary()
            self.generator.summary()
            self.combined.summary()


    def build_generator(self):

        model = Sequential()
        model.add(layers.Dense(128 * 56 * 56, activation="relu", input_dim=self.latent_dim))
        model.add(layers.Reshape((56, 56, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(layers.Activation("tanh"))
        model.summary()
        #noise = layers.Input(shape=(self.latent_dim,))
        #img = model(noise)
        #return Model(noise, img)
        return model

    def build_discriminator_ResNet(self):
        model = Sequential()
        # 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
        # NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
        model.add(ResNet50(input_shape = (self.img_rows, self.img_cols, self.channels), include_top = False, weights='imagenet'))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        # Say not to train first layer (ResNet) model as it is already trained
        model.layers[0].trainable = False
        model.summary()
        return model

    def build_discriminator(self):

        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(layers.ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        model.summary()
        img = layers.Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def save_imgs(self, epoch):

            r, c = 5, 5
            noise = np.random.normal(0, 1, (r * c, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(gen_imgs[cnt])
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig("images/ebsd_%d.png" % epoch)
            plt.close()

    def train(self,X_train, Z_Train, epochs, batch_size=128, save_interval=50):

        valid = np.ones((batch_size))
        fake = np.zeros((batch_size))
            #plt.subplots()
            #plt.imshow(gen_imgs[0])
            #plt.imshow(imgs[0])
            #plt.show()
            #print(loss)

        while self.epoch < epochs:
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            z = Z_Train[idx]
            
            # Generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            if self.epoch % 2 == 0:
                gen_imgs = self.generator.predict(noise)
            else:
                gen_imgs = self.generator.predict(z)
            # Train the discriminator
            d_loss = self.discriminator.train_on_batch(np.concatenate((imgs, gen_imgs)), np.concatenate((valid, fake)))
            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator (to have the discriminator label samples as valid)
            if self.epoch % 2 == 0:
                g_loss = self.combined.train_on_batch(noise, valid)
            else:
                g_loss = self.combined.train_on_batch(z, valid)
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (self.epoch, d_loss[0], 100*d_loss[1], g_loss))
            # If at save interval => save generated image samples
            if self.epoch % save_interval == 0:
                self.save_imgs(self.epoch)

            if self.epoch % (save_interval * 5) == 0:
                self.generator.save('generator.hdf5')
                self.discriminator.save('discriminator.hdf5')
                self.combined.save('combined.hdf5')
            self.epoch += 1



#%%
# imgDir = 'train2'
# TRAINING_SIZE = 3600

# data = None
# labels = None
# aug = ImageDataGenerator(
#         horizontal_flip=True,
#         vertical_flip=True,
#         fill_mode="nearest")

# augmentation = aug.flow_from_directory(imgDir, batch_size = TRAINING_SIZE, target_size=(224,224), save_to_dir="augmented", seed=10, class_mode="sparse")
# print(augmentation.class_indices)
# for x, y in augmentation:
#     data = x
#     labels = y
#     break
# data = data.astype(np.float32) / 255.0
# network = ResNet50(weights="imagenet", include_top=False)
# features = network.predict(data)
# features = features.reshape(TRAINING_SIZE, features.shape[1] * features.shape[2] * features.shape[3])
# pca = PCA(n_components=50)
# pcaFeatures = pca.fit_transform(features)



#%%
# pickle.dump(data, open("storedAugedImgs", "wb"))
# pickle.dump(features, open( "storedFeatures", "wb" ))


#%%
data = pickle.load( open( "storedAugedImgs", "rb" ) )
features = pickle.load(open("storedFeatures", "rb"))
pca = PCA(n_components=50)
pcaFeatures = pca.fit_transform(features)
#%%
ks.backend.clear_session()
gan = DCGAN(False, 126001)

#%%
tf.logging.set_verbosity(tf.logging.ERROR)
gan.train(data, pcaFeatures, 500000, batch_size=15, save_interval=200)