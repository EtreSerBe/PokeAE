from __future__ import division, print_function, absolute_import

import numpy as np

import tensorflow as tf

import tflearn

import matplotlib.colors

import pokedataset32_vae_functions as utilities

current_dataset = 'pokedataset'
# current_dataset = 'anime_faces_'

# X and Y are not used in this file.
X_full_HSV, Y_full_HSV, X_full_RGB, Y_full_RGB, X, Y, test_X, test_Y = utilities.ready_all_data_sets(current_dataset)

Y_full_HSV = Y_full_HSV * 0.50
small_X = np.concatenate((X[0:200], test_X[0:200]), axis=0)
small_Y = np.concatenate((Y[0:200], test_Y[0:200]), axis=0)
len_X_div_2 = int(len(X) / 2)
small_X_RGB = np.concatenate((X_full_RGB[0:200], X_full_RGB[len_X_div_2:len_X_div_2 + 200]), axis=0)
expanded_small_X = np.append(small_X, small_Y, axis=1)

test_X = np.asarray(test_X)
test_Y = np.asarray(test_Y)
X_full_RGB = np.asarray(X_full_RGB)

Y = Y * 0.5
test_X = test_X[0:294]  # We only want half of it.
test_Y = test_Y[0:294]
test_Y = test_Y * 0.5
expanded_test_X = np.append(test_X, test_Y, axis=1)
expanded_X = np.append(X, Y, axis=1)

expanded_full_X_HSV = np.append(X_full_HSV, Y_full_HSV, axis=1)
print("getting network to load model*******************")
network_instance = utilities.get_network()

network_instance = tflearn.regression(network_instance,
                                      optimizer='adam',
                                      metric='R2',
                                      loss=utilities.vae_loss,
                                      learning_rate=0.00001)  # adagrad? #adadelta #nesterov did good,

model = tflearn.DNN(network_instance)

print("LOADING MODEL.")

# This hasn't been commited yet, due to network restrictions (AKA slow upload connection).
# Double check to have a folder with the correct path here.
model.load("saved_models/model_Jul_13_optim_adam_loss_vae_loss_"
           "last_activ_relu_latent_128_num_filters_512_1024_decoder_width_8_transfer_V4_poke3_noise4.tflearn")

predict_full_dataset = True
if predict_full_dataset:
    expanded_full_predicted_X = expanded_full_X_HSV

    predicted_X_pixels = X_full_HSV
    predicted_Y = Y_full_HSV
    exporting_RGB = X_full_RGB
else:
    predicted_X = expanded_test_X
    predicted_X_pixels = test_X
    predicted_Y = test_Y

print("getting samples to show on screen.")
encode_decode_sample_original = utilities.predict_batches(expanded_full_X_HSV, model, in_samples_per_batch=64)
reconstructed_pixels_original, reconstructed_types_original = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_original)
print("MSE value for the ORIGINAL model, over the WHOLE dataset is: ")
utilities.mean_square_error(reconstructed_pixels_original, X_full_HSV)

# Now, training only:
encode_decode_sample_original_train = utilities.predict_batches(expanded_X, model, in_samples_per_batch=64)
reconstructed_pixels_original_train, reconstructed_types_original_train = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_original_train)
print("MSE value for the ORIGINAL model, over the TRAINING dataset is: ")
utilities.mean_square_error(reconstructed_pixels_original_train, X)

# Now, testing data only:
encode_decode_sample_original_test = utilities.predict_batches(expanded_test_X, model, in_samples_per_batch=64)
reconstructed_pixels_original_test, reconstructed_types_original_test = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_original_test)
print("MSE value for the ORIGINAL model, over the TESTING dataset is: ")
utilities.mean_square_error(reconstructed_pixels_original_test, test_X)

###############################

# Now, we can load the transfer learning model.
model.load("saved_models/model_Jul_13_optim_adam_loss_vae_loss_"
           "last_activ_relu_latent_128_num_filters_512_1024_decoder_width_8_transfer_V4_poke3.tflearn")

# Both training and testing data together.
encode_decode_sample_transfer = utilities.predict_batches(expanded_full_X_HSV, model, in_samples_per_batch=64)
reconstructed_pixels_transfer, reconstructed_types_transfer = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_transfer)
print("MSE value for the TRANSFER Poke3 learning model, over the WHOLE dataset is: ")
utilities.mean_square_error(reconstructed_pixels_transfer, X_full_HSV)

# Training data only.
encode_decode_sample_transfer_train = utilities.predict_batches(expanded_X, model, in_samples_per_batch=64)
reconstructed_pixels_transfer_train, reconstructed_types_transfer_train = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_transfer_train)
print("MSE value for the TRANSFER Poke3 learning model, over the TRAINING dataset is: ")
utilities.mean_square_error(reconstructed_pixels_transfer_train, X)

# Testing data only.
encode_decode_sample_transfer_test = utilities.predict_batches(expanded_test_X, model, in_samples_per_batch=64)
reconstructed_pixels_transfer_test, reconstructed_types_transfer_test = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_transfer_test)
print("MSE value for the TRANSFER Poke3 learning model, over the TESTING dataset is: ")
utilities.mean_square_error(reconstructed_pixels_transfer_test, test_X)

# ATLAS EXPORTATION ####################################################

print("Exporting both TRAINING and TESTING reconstructed pokemon as an image.")
utilities.export_multi_type_atlas(X_full_RGB, reconstructed_pixels_original, reconstructed_pixels_transfer,
                                  name_prefix='_POKE3NOISE4_VS_POKE3_')

# Now, to export the testing samples only.
exporting_RGB = np.concatenate((X_full_RGB[827:974], X_full_RGB[1801:]), axis=0)
print("Exporting TESTING-ONLY reconstructed pokemon as an image.")
utilities.export_multi_type_atlas(exporting_RGB, reconstructed_pixels_original_test, reconstructed_pixels_transfer_test,
                                  name_prefix='_TESTONLY_POKE3NOISE4_VS_POKE3_')
# print('The number of ORIGINAL types correct types were: ')
# correct_indices = utilities.export_types_csv(predicted_Y, reconstructed_types_original)
# print('The number of FAKE types correct types were: ')
# correct_indices = utilities.export_types_csv(new_types_array, reconstructed_types_fake)
