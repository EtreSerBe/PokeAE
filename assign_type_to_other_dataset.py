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
import pokedataset32_vae_functions as utilities
from PIL import Image
import colorsys

# current_dataset = 'pokedataset'
# current_dataset = 'anime_faces_'

# We don't need all of those.
# X_full_HSV, Y_full_HSV, X_full_RGB, Y_full_RGB, X, Y, test_X, test_Y = utilities.ready_all_data_sets(current_dataset)

X_full_RGB, Y_full_RGB = utilities.prepare_dataset_for_input_layer(
    'pokedataset32_full_RGB_Two_Hot_Encoded.h5')

X_full_RGB_faces, Y_full_RGB_faces = utilities.prepare_dataset_for_input_layer(
    'anime_faces_32_full_RGB_Two_Hot_Encoded.h5', in_dataset_x_label='anime_faces_32_X',
    in_dataset_y_label='anime_faces_32_Y')


# IMPORTANT: we need both datasets loaded in matrix shape. not flattened.
reshaped_image = tf.reshape(X_full_RGB, shape=[len(X_full_RGB), -1, 3])
reshaped_grayscale = tf.image.rgb_to_grayscale(reshaped_image)

reshaped_image_faces = tf.reshape(X_full_RGB_faces, shape=[len(X_full_RGB_faces), -1, 3])
reshaped_grayscale_faces = tf.image.rgb_to_grayscale(reshaped_image_faces)

# Then, split them by type, so we have all fire types in one array, all water in another, and so on.
pokemon_by_types = [None] * utilities.pokemon_types_dim
for (current_image, current_type) in zip(X_full_RGB, Y_full_RGB):
    # Need to convert the types into 1 scalar, according to their alphabetical order. e.g. bug = 0, etc.
    pokemon_by_types[current_type].append(current_image)



# Now, we can compare all images from the non-labeled data set against the types
ssim_results = []
anime_faces_by_type = [utilities.pokemon_types_dim]
for unlabeled_image in X_full_HSV_faces:
    current_best_ssim = -1  # This is the actual [0,1] value returned by the ssim comparison.
    current_best_ssim_index = -1  # This is the index at which it was obtained the best ssim.
    # compare unlabeled_image to all


# For each non-labeled image

# We run the SSIM comparison, and we save that value. We can either average it or some other way to measure.
# for all the images.







