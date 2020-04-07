# -*- coding: utf-8 -*-

""" Variational Auto-Encoder Example.
Using a variational auto-encoder to generate digits images from noise.
MNIST handwritten digits are used as training examples.
References:
    - Auto-Encoding Variational Bayes The International Conference on Learning
    Representations (ICLR), Banff, 2014. D.P. Kingma, M. Welling
    - Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    - [VAE Paper] https://arxiv.org/abs/1312.6114
    - [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

    This article is great to understand all that's going on here.
    https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy import array, newaxis, expand_dims
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import hsv_to_rgb
from scipy.stats import norm  # A normal continuous random variable.
# The location (loc) keyword specifies the mean. The scale (scale) keyword specifies the standard deviation.

import tensorflow as tf

import tflearn
import h5py
import PokeAE.pokedataset32_vae_functions as utilities
from PIL import Image
import colorsys

# We don't need the Ys.
X_full_HSV, Y_full_HSV = utilities.prepare_dataset_for_input_layer('pokedataset32_full_HSV.h5')

# We don't need the Ys.
X_full_RGB, Y_full_RGB = utilities.prepare_dataset_for_input_layer('pokedataset32_full_RGB.h5')

# Load the hdf5 dataset for the RGB data, to show it in the output.
X_12_3_RGB, Y_12_3_RGB = utilities.prepare_dataset_for_input_layer('pokedataset32_12_3_RGB.h5')

X, Y = utilities.prepare_dataset_for_input_layer('pokedataset32_12_3_HSV.h5')

test_X, test_Y = utilities.prepare_dataset_for_input_layer('pokedataset32_12_3_HSV.h5',
                                                                             in_dataset_x_label='pokedataset32_X_test',
                                                                             in_dataset_y_label='pokedataset32_Y_test')

# 3072 in total size.
# 891 total images.

# Params
image_dimension = 32
image_color_dimension = 3
original_dim = 3072  # 32 x 32 RGB images.
# hidden_dim = 768
# latent_dim = 2  # Why is latent dim = 2?
pokemon_types_dim = 18 * 2  # 18 *2, since we need space for the two possible types.
Y = np.reshape(np.asarray(Y), newshape=[Y.shape[0], pokemon_types_dim])
test_Y = np.reshape(np.asarray(test_Y), newshape=[test_Y.shape[0], pokemon_types_dim])

# Now we add the extra info from the Ys.
expanded_X = np.append(X, Y, axis=1)
# expanded_augmented_X = utilities.image_augmentation(expanded_X)  # I need to make this not on runtime?
expanded_Y = np.append(X, Y, axis=1)  # Not really used right now

# Now, we do the same for the training data
expanded_test_X = np.append(test_X, test_Y, axis=1)
expanded_test_Y = np.append(test_X, test_Y, axis=1)  # Not used right now.

# Right now it's the only expanded full that we need.
expanded_full_X_HSV = np.append(X_full_HSV, Y_full_HSV, axis=1)

print("expanded Xs and Ys ready")


image_aug = tflearn.ImageAugmentation()
image_aug.add_random_blur(sigma_max=2.0)

NUM_FILTERS_FIRST = 64
NUM_FILTERS_SECOND = 64
FILTER_SIZE_FIRST = 5
FILTER_SIZE_SECOND = 3
FILTER_STRIDES_FIRST = 1
FILTER_STRIDES_SECOND = 1

FULLY_CONNECTED_1_UNITS = 128
FULLY_CONNECTED_2_UNITS = 64
# FULLY_CONNECTED_3_UNITS = 64

DECODER_WIDTH = 8
EMBEDDED_VECTOR_SIZE = DECODER_WIDTH * DECODER_WIDTH
EMBEDDED_VECTOR_TOTAL = EMBEDDED_VECTOR_SIZE * image_color_dimension

# Building the encoder
# The size of the input should be 3108 = 3072 + 18*2
networkInput = tflearn.input_data(shape=[None, original_dim + pokemon_types_dim], data_augmentation=image_aug)

# Once the data is in, we need to split the pixel data and the types data.
map_flat = tf.slice(networkInput, [0, 0], [-1, original_dim])
pokemonTypesFlat = tf.slice(networkInput, [0, original_dim], [-1, -1])

# We reshape the flat versions to something more like the original.
mapShape = tf.reshape(map_flat, [-1, image_dimension, image_dimension, image_color_dimension])
print("mapShape dimensions, before Conv_2D #1 are: " + str(mapShape))
pokemonTypes = tf.reshape(pokemonTypesFlat, [-1, pokemon_types_dim])

encoderStructure = tflearn.conv_2d(mapShape, NUM_FILTERS_FIRST, FILTER_SIZE_FIRST,
                                   strides=FILTER_STRIDES_FIRST, activation='relu')
print("encoderStructure before dropout is: " + str(encoderStructure))
encoderStructure = tflearn.dropout(encoderStructure, 0.5)
print("encoderStructure before max_pool_2D #1 is: " + str(encoderStructure))
encoderStructure = tflearn.max_pool_2d(encoderStructure, 2, strides=2)
print("encoderStructure before conv_2D #2 is: " + str(encoderStructure))
encoderStructure = tflearn.conv_2d(encoderStructure, NUM_FILTERS_SECOND, FILTER_SIZE_SECOND,
                                   strides=FILTER_STRIDES_SECOND, activation='relu')
print("encoderStructure before max_pool_2D #2 is: " + str(encoderStructure))
encoderStructure = tflearn.max_pool_2d(encoderStructure, 2, strides=2)
print("encoderStructure before flatten is: " + str(encoderStructure))

flatStructure = tflearn.flatten(encoderStructure)
print("flatStructure is = " + str(flatStructure))
flatStructureSize = flatStructure.shape[1]  # Why is it size 2048 with 8 filters and 1024 with 4?
print('flatStructureSize = ' + str(flatStructureSize))

encoder = tf.concat([flatStructure, pokemonTypes], 1)

encoder = tflearn.fully_connected(encoder, FULLY_CONNECTED_1_UNITS, activation='relu')

encoder = tflearn.fully_connected(encoder, FULLY_CONNECTED_2_UNITS, activation='relu')  # embedded representation? Yes.

# decoder = tflearn.fully_connected(encoder, FULLY_CONNECTED_1_UNITS, activation='relu')

decoder = tflearn.fully_connected(encoder, int(EMBEDDED_VECTOR_TOTAL + pokemon_types_dim), activation='relu')

decoderStructure = tf.slice(decoder, [0, 0], [-1, EMBEDDED_VECTOR_TOTAL])
decoderTypes = tf.slice(decoder, [0, EMBEDDED_VECTOR_TOTAL], [-1, -1])
print("decoder types size is: " + str(decoderTypes))

decoderStructure = tf.reshape(decoderStructure, [-1, DECODER_WIDTH, DECODER_WIDTH,
                                                 image_color_dimension])

decoderStructure = tflearn.conv_2d(decoderStructure, NUM_FILTERS_SECOND, FILTER_SIZE_SECOND,
                                   strides=FILTER_STRIDES_SECOND, activation='relu')
decoderStructure = tflearn.upsample_2d(decoderStructure, 2)
decoderStructure = tflearn.conv_2d(decoderStructure, NUM_FILTERS_FIRST, FILTER_SIZE_FIRST,
                                   strides=FILTER_STRIDES_FIRST, activation='relu')
decoderStructure = tflearn.upsample_2d(decoderStructure, 2)

decoderStructure = tflearn.flatten(decoderStructure)  # With 4 filters, it has 65536*3018 connections...

network = tf.concat([decoderStructure, decoderTypes], 1)

print("network before the final fully_connected is: " + str(network))
network = tflearn.fully_connected(network, original_dim + pokemon_types_dim, activation='relu')

network = tflearn.regression(network, optimizer='adadelta',
                             metric='R2',
                             loss='mean_square',
                             learning_rate=0.07)  # adagrad?

print("regression successful, network is now: " + str(network))

model = tflearn.DNN(network)

print("Preparing model to fit.")

#"""
model.fit(expanded_X, Y_targets=expanded_X,
          n_epoch=100,
          shuffle=True,
          show_metric=True,
          snapshot_epoch=True,
          batch_size=32,
          validation_set=0.2,  # It also accepts a float < 1 to performs a data split over training data.
          # validation_set=(expanded_test_X, expanded_test_X),
          run_id='encoder_decoder')

model.save("pokedatamodel32_April_7_1.tflearn")
#"""

"""
# This hasn't been commited yet, due to network restrictions (AKA slow upload connection).
model.load("saved models from pokemon/pokedatamodel32_April_1_3.tflearn")

# Add the fake types.
new_types_array = utilities.generate_all_one_type(len(X_full_HSV),
                                                                    in_type="Fire", in_second_type="None")
new_types_array = np.reshape(np.asarray(new_types_array), newshape=[new_types_array.shape[0], pokemon_types_dim])
expanded_fake_X = np.append(X_full_HSV, new_types_array, axis=1)
"""

print("getting samples to show on screen.")
encode_decode_sample = model.predict(expanded_full_X_HSV)
# encode_decode_sample = model.predict(expanded_fake_X)

reconstructed_pixels = []
reconstructed_types = []
reshaped_sample = []

for i in range(0, len(encode_decode_sample)):
    sample = encode_decode_sample[i][0:3072]
    reshaped_sample = np.reshape(sample, [1024, 3])
    # https://matplotlib.org/api/_as_gen/matplotlib.colors.hsv_to_rgb.html#matplotlib.colors.hsv_to_rgb
    reshaped_sample = matplotlib.colors.hsv_to_rgb(reshaped_sample)
    pixel_list = reshaped_sample.flatten()
    reconstructed_pixels.append(pixel_list)
    reshaped_types = np.reshape(encode_decode_sample[i][3072:3108], [2, 18])
    reconstructed_types.append(reshaped_types)


print("Exporting reconstructed pokemon as an image.")
utilities.export_as_atlas(expanded_X, reconstructed_pixels)
correct_indices = utilities.export_types_csv(Y_full_RGB, reconstructed_types)
# correct_indices = utilities.export_types_csv(new_types_array, reconstructed_types)

correct_X_RGB = [X_full_RGB[i] for i in correct_indices]
correct_reconstructed_pixels = [reconstructed_pixels[i] for i in correct_indices]
utilities.export_as_atlas(correct_X_RGB, correct_reconstructed_pixels, name_annotations='correct')

"""
# I used this before to show the results, but now I have the whole image being saved.
print("PREPARING TO SHOW IMAGE")
# Compare original images with their reconstructions.
f, a = plt.subplots(2, 20, figsize=(20, 2), squeeze=False)  # figsize=(50, 2),
for i in range(20):
    # reshaped_pokemon = np.multiply(reshaped_pokemon, 255.0)
    reshaped_pokemon = np.reshape(np.asarray(X[i]), [1024, 3])
    RGBOriginal = matplotlib.colors.hsv_to_rgb(reshaped_pokemon)
    RGBOriginal = np.asarray(RGBOriginal).flatten()
    temp = [[ii] for ii in list(RGBOriginal)]  # WTH? Python, you're drunk haha.
    print("ORIGINAL Types for Pokemon " + str(i) + " are: ")
    pokedataset32_vae_functions.print_pokemon_types(Y[i])
    a[0][i].imshow(np.reshape(temp, (32, 32, 3)))
    temp = [[ii] for ii in list(reconstructed_pixels[i])]
    a[1][i].imshow(np.reshape(temp, (32, 32, 3)))
    print("Types for Pokemon " + str(i) + " are: ")
    pokedataset32_vae_functions.print_pokemon_types(reconstructed_types[i])
f.show()
plt.draw()
plt.waitforbuttonpress()
"""
