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
import pokedataset32_vae_functions as utilities
from PIL import Image
import colorsys

# current_dataset = 'pokedataset'
current_dataset = 'anime_faces_'

X_full_HSV, Y_full_HSV, X_full_RGB, Y_full_RGB, X, Y, test_X, test_Y = utilities.ready_all_data_sets(current_dataset)

# NOTE: Use these lines to output a visualization of the data sets, if you think
# there is any problem with them. But I've checked and they seem correct.
# X = utilities.convert_to_format(X[:], 'HSV_TO_RGB')
# utilities.export_as_atlas(X, X)
Y = Y * 0.5
test_Y = test_Y * 0.5
Y_full_HSV = Y_full_HSV * 0.5  # np.clip(Y_full_HSV, 0.0, 1.0)
Y_full_RGB = Y_full_RGB * 0.5

small_X = np.concatenate((X[0:200], test_X[0:200]), axis=0)
small_Y = np.concatenate((Y[0:200], test_Y[0:200]), axis=0)

# utilities.create_hashmap(X_full_HSV)

# Now we add the extra info from the Ys.
expanded_X = np.append(X, Y, axis=1)  # It already contains the Flip-left-right augmentation.

# Now, we do the same for the training data
expanded_test_X = np.append(test_X, test_Y, axis=1)

# Right now it's the only expanded full that we need.
expanded_full_X_HSV = np.append(X_full_HSV, Y_full_HSV, axis=1)
expanded_small_X = np.append(small_X, small_Y, axis=1)
print("expanded Xs and Ys ready")

# utilities.initialize_session()
# current_session = utilities.get_session()

# I put the network's definition in the pokedataset32_vae_functions.py file, to unify it with the load model.
network_instance = utilities.get_network()

predict_full_dataset = False
optimizer_name = 'adam'
loss_name = 'vae_loss'
final_model_name = utilities.get_model_descriptive_name(optimizer_name, loss_name, in_version='_2by2')

network_instance = tflearn.regression(network_instance,
                                      # optimizer='rmsprop',
                                      optimizer=optimizer_name,
                                      metric='R2',
                                      # loss='mean_square',
                                      loss=utilities.vae_loss,
                                      # loss=utilities.vae_loss_abs_error,
                                      learning_rate=0.001)  # adagrad? #adadelta #nesterov did good,

# proximaladagrad did meh, almost same as others.
# With adadelta I can't get it to do anything with a small learning rate. with 0.07 i can get near nesterov.
# Adagrad gets stuck around- 0.2400 R2.
# rmsprop not usable?
# momentum got mostly to the same as Nesterov.
# sgd gets stuck around -2400 R2 too.

model = tflearn.DNN(network_instance)  #, session=current_session)  # , tensorboard_verbose=2)

print("Preparing model to fit.")

model.fit(expanded_X, Y_targets=expanded_X,
          n_epoch=5,
          shuffle=True,
          show_metric=True,
          snapshot_epoch=True,
          batch_size=64,
          # validation_set=0.15,  # It also accepts a float < 1 to performs a data split over training data.
          validation_set=(expanded_test_X, expanded_test_X),  # We use it for validation for now. But also test.
          run_id='encoder_decoder')

print("getting samples to show on screen.")
encode_decode_sample = []
if predict_full_dataset:
    predicted_X = X
    predicted_Y = Y_full_RGB
    encode_decode_sample = utilities.predict_batches(expanded_full_X_HSV, model, in_samples_per_batch=64)
else:
    predicted_X = small_X
    predicted_Y = small_Y
    encode_decode_sample = utilities.predict_batches(expanded_small_X, model, in_samples_per_batch=64)

# encode_decode_sample = model.predict(expanded_X)  # Just to test training with RGB. It seemed worse.

print("The number of elements in the predicted samples is: " + str(len(encode_decode_sample)))

reconstructed_pixels = []
reconstructed_types = []
# reshaped_sample = []

# Made a function to avoid repeating that fragment of code in other python files.
reconstructed_pixels, reconstructed_types = utilities.reconstruct_pixels_and_types(encode_decode_sample)

print("Exporting reconstructed pokemon as an image.")
# utilities.export_as_atlas(X_full_RGB, reconstructed_pixels)  # I have checked that it works perfectly.
if predict_full_dataset:
    correct_indices = utilities.export_types_csv(Y_full_RGB, reconstructed_types)
else:
    correct_indices = utilities.export_types_csv(small_Y, reconstructed_types)

# This is used to export an image only containing the ones whose types were correctly predicted by the NN.
# correct_X_RGB = [X_full_RGB[i] for i in correct_indices]
# correct_reconstructed_pixels = [reconstructed_pixels[i] for i in correct_indices]
# utilities.export_as_atlas(correct_X_RGB, correct_reconstructed_pixels, name_annotations='correct')

# I used this before to show the results, but now I have the whole image being saved.
print("PREPARING TO SHOW IMAGE")
# Compare original images with their reconstructions.
f, a = plt.subplots(2, 20, figsize=(20, 2), squeeze=False)  # figsize=(50, 2),
for i in range(20):
    # reshaped_pokemon = np.multiply(reshaped_pokemon, 255.0)
    reshaped_pokemon = np.reshape(np.asarray(predicted_X[i]), [1024, 3])
    RGBOriginal = matplotlib.colors.hsv_to_rgb(reshaped_pokemon)
    RGBOriginal = np.asarray(RGBOriginal).flatten()
    temp = [[ii] for ii in list(RGBOriginal)]  # WTH? Python, you're drunk haha.
    print("ORIGINAL Types for Pokemon " + str(i) + " are: ")
    utilities.print_pokemon_types(predicted_Y[i])
    a[0][i].imshow(np.reshape(temp, (32, 32, 3)))
    temp = [[ii] for ii in list(reconstructed_pixels[i])]
    a[1][i].imshow(np.reshape(temp, (32, 32, 3)))
    print("Types for Pokemon " + str(i) + " are: ")
    utilities.print_pokemon_types(reconstructed_types[i])
f.show()
plt.draw()
plt.waitforbuttonpress()

print('Now saving the model')
model.save(final_model_name)
print('Save successful, closing application now.')
