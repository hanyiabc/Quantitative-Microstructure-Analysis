#%%

import numpy as np
import tensorflow as tf
import keras as ks
import cv2
import pickle

from sklearn.decomposition import PCA
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import LeakyReLU, UpSampling2D, Conv2D, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys
import numpy as np

imgDir = 'train2'
NDIMS = 100

#%%

class DCGAN():

    def __init__(self, load=False):

        # Input shape

        self.img_rows = 224
        self.img_cols = 224
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = NDIMS
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        if not load:

            self.discriminator = self.build_discriminator()
            self.discriminator.compile(
                loss='binary_crossentropy',
                optimizer='SGD',
                metrics=['accuracy'])
            
            # Build the generator
            self.generator = self.build_generator()
            # The generator takes noise as input and generates imgs
            self.discriminator.trainable = False
            # connect them
            self.combined = Sequential()
            # add generator
            self.combined.add(self.generator)
            # add the discriminator
            self.combined.add(self.discriminator)
            # compile model
            self.combined.summary()
            self.combined.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            # z = layers.Input(shape=(self.latent_dim,))
            # img = self.generator(z)
            
            # # For the combined model we will only train the generator
            # # The discriminator takes generated images as input and determines validity
            # self.discriminator.trainable = False
            # valid = self.discriminator(img)
            # # The combined model  (stacked generator and discriminator)
            # # Trains the generator to fool the discriminator
            # self.combined = Model(z, valid)
            # self.combined.compile(loss='binary_crossentropy',
            # optimizer=optimizer,
            # metrics=['accuracy'])
        else:
            self.discriminator = ks.models.load_model("discriminatorv2.hdf5")
            self.discriminator.layers[0].trainable = False
            self.discriminator.compile(
                loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
            self.generator = ks.models.load_model("generatorv2.hdf5")
            z = layers.Input(shape=(self.latent_dim,))
            img = self.generator(z)
            self.discriminator.summary()
            self.generator.summary()
            self.discriminator.trainable = False
            valid = self.discriminator(img)
            self.combined = Model(z, valid)
            #self.combined = ks.models.load_model("combined.hdf5")
            self.combined.compile(
                loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

            self.discriminator.summary()
            self.generator.summary()
            self.combined.summary()


    def build_generator(self):

        model = Sequential()
        model.add(layers.Dense(128 * 56 * 56, activation="relu", input_dim=self.latent_dim))
        model.add(layers.Reshape((56, 56, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(200, kernel_size=3, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(UpSampling2D())
        model.add(Conv2D(100, kernel_size=3, padding="same"))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(layers.Activation("tanh"))
        model.summary()

        noise = layers.Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

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
        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    def save_imgs(self, epoch, generatedImgs):
            # Rescale images 0 - 1
            r, c = 3, 3
            generatedImgs = 0.5 * generatedImgs + 0.5
            fig, axs = plt.subplots(r, c)
            fig.set_size_inches(25, 25)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(generatedImgs[cnt])
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig("images/ebsd_%d.png" % epoch)
            plt.close()

    def train(self, dir, pcaModel, epochs, batch_size=128, save_interval=50, currEpoch=0):
        self.epoch = currEpoch
        valid = np.ones((batch_size))
        fake = np.zeros((batch_size))
        aug = ImageDataGenerator(
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode="nearest")

        augmentation = aug.flow_from_directory(dir, batch_size = batch_size, class_mode="sparse", target_size=(224,224), seed=111)
        print(augmentation.class_indices)
        for x, y in augmentation:
            x /= 255.0
            x = (x - 0.5) * 2.0
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
           
            gen_imgs = self.generator.predict(noise)
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(x, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # if self.epoch % 2 == 0:
            #     self.epoch += 1
            #     continue

            g_loss = self.combined.train_on_batch(noise, valid)
            
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, acc.: %.2f%%]" % (self.epoch, d_loss[0], 100*d_loss[1], g_loss[0], 100 * g_loss[1]))
            # If at save interval => save generated image samples
            if self.epoch % save_interval == 0:
                self.save_imgs(self.epoch, gen_imgs)

            if self.epoch % (save_interval * 5) == 0:
                self.generator.save('generatorv2.hdf5')
                self.discriminator.save('discriminatorv2.hdf5')
            self.epoch += 1
            if self.epoch > epochs:
                self.generator.save('generatorv2.hdf5')
                self.discriminator.save('discriminatorv2.hdf5')
                break



#%%
imgDir = 'train2'
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
# pca = PCA(n_components=NDIMS)
# pcaFeatures = pca.fit_transform(features)



#%%
# pickle.dump(data, open("storedAugedImgs", "wb"))
# pickle.dump(features, open( "storedFeatures", "wb" ))

#%%
features = pickle.load(open("storedFeatures", "rb"))
pca = PCA(n_components=NDIMS)
pca.fit(features)
#%%

#ks.backend.clear_session()
# ks.backend.set_floatx('float16')
# ks.backend.set_epsilon(1e-4)
gan = DCGAN()
#%%
#tf.logging.set_verbosity(tf.logging.ERROR)
gan.train(imgDir, pca, 500000, batch_size=15, save_interval=200)