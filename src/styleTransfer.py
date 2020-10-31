import time
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow import keras as ks
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG19
from scipy.optimize import fmin_l_bfgs_b
from skimage.io import imread, imsave
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def content_loss(content, combination):
    return K.sum(K.square(combination - content))

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination, size):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def style_loss_constant(gram_style, combination, size):
    S = gram_style
    C = gram_matrix(combination)
    channels = 3
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x, H, W):
    a = K.square(x[:, :H-1, :W-1, :] - x[:, 1:, :W-1, :])
    b = K.square(x[:, :H-1, :W-1, :] - x[:, :H-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

class StyleTransfer:
    
    def eval_loss_and_grads(self, x):
        x = x.reshape((1, self.H, self.W, 3))
        outs = self.f_outputs([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        return loss_value, grad_values

    def __init__(self, H=512, W=512, C=3, C_weight=1e-3, S_weight=1.0, V_weight = 1.0, content_path=None, style_path=None, style_gram_mat=None, style_image=None, vgg19=None):

        self.H = H
        self.W = W

        if style_gram_mat is not None:
            

            if vgg19 is None:
                input_tensor = ks.Input(shape=(H, W, 3))
                self.model = VGG19(input_tensor=input_tensor, weights='imagenet',
                include_top=False, pooling='avg')
            else:
                input_tensor = vgg19.input
                self.model = vgg19
                # self.model.summary()
            combination_image = input_tensor
            self.layers = dict([(layer.name, layer.output) for layer in self.model.layers])
            # layer_features = self.layers['block2_conv2']
            # combination_features = layer_features[0, :, :, :]
            feature_layers = ['block1_pool', 'block2_pool',
                  'block3_pool', 'block4_pool',
                  'block1_conv1']
            idx = 0
            self.loss = K.constant(0)
            for layer_name in feature_layers:
                layer_features = self.layers[layer_name]
                combination_features = layer_features[0, :, :, :]
                gram_size = combination_features.shape[2]
                style_gram_mat_tensor = K.reshape(K.variable(style_gram_mat[idx:idx + gram_size ** 2]), ((gram_size, gram_size))) 
                idx += gram_size ** 2
                sl = style_loss_constant(style_gram_mat_tensor, combination_features, H * W)
                self.loss += (S_weight / len(feature_layers)) * sl
            self.loss = self.loss + total_variation_loss(combination_image, H, W)
            grads = K.gradients(self.loss, combination_image)
            outputs = [self.loss]
            outputs += grads
            self.f_outputs = K.function([combination_image], outputs)
        else:
            if style_path:
                style_image = imread(style_path).astype(np.float32)[np.newaxis, :]
                # style_image = cv2.resize(style_image, (H, W))[np.newaxis, :]

            style_image = K.variable(style_image)
            combination_image = K.placeholder((1, H, W, 3))

            input_tensor = K.concatenate([style_image, combination_image], axis=0)
            if vgg19 is None:
                self.model = VGG19(input_tensor=input_tensor, weights='imagenet',
                include_top=False, pooling='avg')
            else:
                self.model = vgg19(input_tensor)

            self.layers = dict([(layer.name, layer.output) for layer in self.model.layers])
            feature_layers = ['block1_pool', 'block2_pool',
                  'block3_pool', 'block4_pool',
                  'block1_conv1']
            self.loss = K.constant(0)
            for layer_name in feature_layers:
                layer_features = self.layers[layer_name]
                style_features = layer_features[0, :, :, :]
                combination_features = layer_features[1, :, :, :]
                sl = style_loss(style_features, combination_features, H * W)
                self.loss += (S_weight / len(feature_layers)) * sl
            self.loss = self.loss + total_variation_loss(combination_image, H, W)
            grads = K.gradients(self.loss, combination_image)
            outputs = [self.loss]
            outputs += grads
            self.f_outputs = K.function([combination_image], outputs)
    


class Evaluator(object):

    def __init__(self, styleTransfer):
        self.loss_value = None
        self.grads_values = None
        self.st = styleTransfer

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.st.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


def train(evaluator, H, W, iterations=10):
    
    # x = np.random.uniform(0, 1.0, (1, H, W, 3))
    x = np.ones((1, H, W, 3)) / 2.0
    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                        fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))

    return x

def generateImageFromStyle(H=512, W=512, C=3, S_weight=1.0, V_weight=1.0, style_image=None, iters=20, style_path=None):

    st = StyleTransfer(H=H, W=W, C=3, S_weight=S_weight, V_weight=V_weight, content_path='noise.png', style_image=style_image, style_path=style_path)
    evaluator = Evaluator(st)
    generated = train(evaluator, H, W ,iters)
    generated = generated.reshape((H, W, 3))
    # generated = (np.clip(generated, 0.0, 255.0)).astype('uint8')
    return generated

def generateImageFromGramMatrix(gram_mat, H=512, W=512, C=3, S_weight=1.0, V_weight=1.0, iters=20, vgg19=None):
    st = StyleTransfer(H=H, W=W, C=3, S_weight=S_weight, V_weight=V_weight, content_path='noise.png', style_gram_mat=gram_mat, vgg19=vgg19)
    evaluator = Evaluator(st)
    generated = train(evaluator,H, W, iters)
    generated = generated.reshape((H, W, 3))
    # generated =(np.clip(generated, 0.0, 255.0)).astype('uint8')
    return generated


if __name__ == '__main__':
    H = 778
    W = 937
    N_ITERS = 20

    CONTENT_WEIGHT = 0
    TOTAL_VARIATION_WEIGHT = 1.0
    STYLE_WEIGHT = 1.0

    STYLE_PATH = 'cropped_same/600.3.png'
    generated = generateImageFromStyle(H=H, W=W, C=3, S_weight=STYLE_WEIGHT, V_weight=TOTAL_VARIATION_WEIGHT, style_path=STYLE_PATH, iters=N_ITERS)
    # st = StyleTransfer(H=H, W=W, C=3, C_weight=CONTENT_WEIGHT, S_weight=STYLE_WEIGHT, V_weight=TOTAL_VARIATION_WEIGHT, content_path=CONTENT_PATH, style_path=STYLE_PATH)
    # evaluator = Evaluator(st)
    # generated = train(evaluator, N_ITERS)

    # generated = generated.reshape((H, W, 3))
    # generated = np.clip(generated, 0, 255).astype('uint8')
    imsave('generated_600.3.png', generated)