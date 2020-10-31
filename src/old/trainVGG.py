#%%
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import multi_gpu_model
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import gc

tf.compat.v1.disable_eager_execution()
#%%

TRAIN = 'train_euler'
classes = ['300.3', '300.9', '600.3', '600.9']
MODEL_PATH = '01-1.30.h5'
BATCH_SIZE_SINGLE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 50
INIT_EPOCH = 0
NUMBER_OF_GPUS = 1
BATCH_SIZE = NUMBER_OF_GPUS * BATCH_SIZE_SINGLE
LOADING = False
NUM_CLASSES = len(classes)

#%%

# gen =  keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)

# sample_gen = gen.flow_from_directory(TRAIN, classes=classes, batch_size=10000, target_size=IMAGE_SIZE, class_mode='categorical', seed=4, subset='training', interpolation='nearest', shuffle=True)
# sample, y_sample = sample_gen.__next__()


# hist = np.sum(y_sample, axis=0)
# weight = y_sample.shape[0] / (NUM_CLASSES * hist)
# zipped = zip(np.arange(NUM_CLASSES, dtype=np.int32).tolist(), weight.tolist()) 
# classWeights = dict(zipped)
# print(hist, classWeights)
# classWeights = None
#%%
# gen =  keras.preprocessing.image.ImageDataGenerator(validation_split=0.2, 
#                                                     featurewise_center=False, 
#                                                     featurewise_std_normalization=False, 
#                                                     rescale=1/255.0
#                                                     )

# gen.fit(sample)
#%%
gen = keras.preprocessing.image.ImageDataGenerator(validation_split=0.2, rescale=1/255.0)
train_gen = gen.flow_from_directory(TRAIN, 
                                    classes=classes, 
                                    batch_size=BATCH_SIZE, 
                                    target_size=IMAGE_SIZE,
                                    class_mode='categorical', 
                                    seed=4, subset='training', 
                                    interpolation='nearest', 
                                    shuffle=True
                                    )
                                    
val_gen = gen.flow_from_directory(TRAIN, 
                                classes=classes, 
                                batch_size=BATCH_SIZE, 
                                target_size=IMAGE_SIZE,
                                class_mode='categorical', 
                                seed=4, subset='validation', 
                                interpolation='nearest', 
                                shuffle=True
                                )

weight_save_callback = keras.callbacks.ModelCheckpoint('./{epoch:02d}-{loss:.2f}.h5', 
                                                        monitor='loss', 
                                                        verbose=0, 
                                                        save_best_only=False, 
                                                        mode='auto'
                                                        )
#%%
keras.backend.clear_session()
gc.collect()

# with tf.device('/cpu:0'):
# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
if LOADING:
    model = keras.models.load_model(MODEL_PATH)
else:
    model = keras.models.Sequential()
    model.add(keras.applications.VGG16(include_top=True, weights=None, input_shape=(*IMAGE_SIZE, 3)), classes=NUM_CLASSES)
    # model.add(Flatten(name='flatten'))
    # model.add(Dense(4096, activation='relu', name='fc1'))
    # model.add(Dense(4096, activation='relu', name='fc2'))
    # model.add(Dense(NUM_CLASSES, activation='softmax', name='predictions'))
if NUMBER_OF_GPUS > 1:
    model = multi_gpu_model(model, cpu_merge=True, cpu_relocation=True, gpus=NUMBER_OF_GPUS)
model.compile(optimizer='Adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
            )
model.summary()

#%%
model.fit(train_gen, 
        epochs=EPOCHS, 
        validation_data=val_gen, 
        validation_steps=500/BATCH_SIZE, 
        callbacks = [weight_save_callback], 
        initial_epoch=INIT_EPOCH, 
        max_queue_size=100
        )
