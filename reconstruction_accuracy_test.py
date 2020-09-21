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

X_full_HSV_regional, Y_full_HSV_regional = \
    utilities.prepare_dataset_for_input_layer("pokedataset32_full_HSV_Two_Hot_Encoded_Regional.h5")
Y_full_HSV_regional = Y_full_HSV_regional * 0.5
expanded_full_X_HSV_regional = np.append(X_full_HSV_regional, Y_full_HSV_regional, axis=1)
X_full_RGB_regional, Y_full_RGB_regional = \
    utilities.prepare_dataset_for_input_layer("pokedataset32_full_RGB_Two_Hot_Encoded_Regional.h5")
Y_full_RGB_regional = Y_full_HSV_regional
expanded_full_X_RGB_regional = np.append(X_full_HSV_regional, Y_full_HSV_regional, axis=1)

Y_full_HSV = Y_full_HSV * 0.50
small_X = np.concatenate((X[0:200], test_X[0:200]), axis=0)
small_Y = np.concatenate((Y[0:200], test_Y[0:200]), axis=0)
len_X_div_2 = int(len(X) / 2)
small_X_RGB = np.concatenate((X_full_RGB[0:200], X_full_RGB[len_X_div_2:len_X_div_2 + 200]), axis=0)
expanded_small_X = np.append(small_X, small_Y, axis=1)

test_X = np.asarray(test_X)
test_Y = np.asarray(test_Y)
X_full_RGB = np.asarray(X_full_RGB)
# X_train_RGB = X_full_RGB[0:int(len(X)/2)]
# X_test_RGB = X_full_RGB[int(len(X)/2):int(len(X_full_RGB)/2)]
X_train_RGB = np.concatenate((X_full_RGB[0:int(len(X) / 4)],
                              X_full_RGB[int(len(X_full_RGB) / 2): int(len(X_full_RGB) / 2) + int(len(X) / 4)]), axis=0)
X_test_RGB = np.concatenate((X_full_RGB[int(len(X) / 4):int(len(X_full_RGB) / 2)],
                             X_full_RGB[-int(len(test_X) / 4):]), axis=0)

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

print_ssim_scores = True

print("LOADING MODEL.")
first_model_name = "_regular_V3_no_noise2"  # This is the one for the encoder-only part. From Jul_28
# first_model_name = "_regular_V2_more_noise2"  # this was from Jul_23
# This hasn't been commited yet, due to network restrictions (AKA slow upload connection).
# Double check to have a folder with the correct path here.
model.load("saved_models/model_Jul_29_optim_adam_loss_vae_loss_"
           "last_activ_relu_latent_128_num_filters_512_1024_decoder_width_8" + first_model_name + ".tflearn")

print("getting samples to show on screen.")
encode_decode_sample_original = utilities.predict_batches(expanded_full_X_HSV, model, in_samples_per_batch=64)
reconstructed_pixels_original, reconstructed_types_original = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_original)
print("MSE value for the " + first_model_name + " model, over the WHOLE dataset is: ")
utilities.mean_square_error(reconstructed_pixels_original, X_full_RGB)
if print_ssim_scores:
    print("SSIM is: ")
    utilities.ssim_comparison(reconstructed_pixels_original, X_full_RGB)

# Now, training only:
encode_decode_sample_original_train = utilities.predict_batches(expanded_X, model, in_samples_per_batch=64)
reconstructed_pixels_original_train, reconstructed_types_original_train = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_original_train)
print("MSE value for the " + first_model_name + " model, over the TRAINING dataset is: ")
utilities.mean_square_error(reconstructed_pixels_original_train[0:len(X_train_RGB)], X_train_RGB)
if print_ssim_scores:
    print("SSIM is: ")
    utilities.ssim_comparison(reconstructed_pixels_original_train[0:len(X_train_RGB)], X_train_RGB)

# Now, testing data only:
encode_decode_sample_original_test = utilities.predict_batches(expanded_test_X, model, in_samples_per_batch=64)
reconstructed_pixels_original_test, reconstructed_types_original_test = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_original_test)
print("MSE value for the " + first_model_name + " model, over the TESTING dataset is: ")
utilities.mean_square_error(reconstructed_pixels_original_test[0:len(X_test_RGB)], X_test_RGB)
if print_ssim_scores:
    print("SSIM is: ")
    utilities.ssim_comparison(reconstructed_pixels_original_test[0:len(X_test_RGB)], X_test_RGB)

# Now, Regional test data only:
encode_decode_sample_original_test_regional = utilities.predict_batches(expanded_full_X_HSV_regional,
                                                                        model, in_samples_per_batch=64)
reconstructed_pixels_original_test_regional, reconstructed_types_original_test_regional = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_original_test_regional)
print("MSE value for the " + first_model_name + " model, over the REGIONAL dataset is: ")
utilities.mean_square_error(reconstructed_pixels_original_test_regional, X_full_RGB_regional)
if print_ssim_scores:
    print("SSIM is: ")
    utilities.ssim_comparison(reconstructed_pixels_original_test_regional, X_full_RGB_regional)

###############################

# Now, we can load the transfer learning model.
# second_model_name = "_V3_noise2"  # 21 of july  # "_V3_noise4" THIS WAS A GOOD MODEL.
second_model_name = "_anime_V2_poke5"
model.load("saved_models/model_Jul_29_optim_adam_loss_vae_loss_"
           "last_activ_relu_latent_128_num_filters_512_1024_decoder_width_8" + second_model_name + ".tflearn")

# Both training and testing data together.
encode_decode_sample_transfer = utilities.predict_batches(expanded_full_X_HSV, model, in_samples_per_batch=64)
reconstructed_pixels_transfer, reconstructed_types_transfer = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_transfer)
print("MSE value for the " + second_model_name + " learning model, over the WHOLE dataset is: ")
utilities.mean_square_error(reconstructed_pixels_transfer, X_full_RGB)
if print_ssim_scores:
    print("SSIM is: ")
    utilities.ssim_comparison(reconstructed_pixels_transfer, X_full_RGB)

# Training data only.
encode_decode_sample_transfer_train = utilities.predict_batches(expanded_X, model, in_samples_per_batch=64)
reconstructed_pixels_transfer_train, reconstructed_types_transfer_train = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_transfer_train)
print("MSE value for the " + second_model_name + " learning model, over the TRAINING dataset is: ")
utilities.mean_square_error(reconstructed_pixels_transfer_train[0:len(X_train_RGB)], X_train_RGB)
if print_ssim_scores:
    print("SSIM is: ")
    utilities.ssim_comparison(reconstructed_pixels_transfer_train[0:len(X_train_RGB)], X_train_RGB)

# Testing data only.
encode_decode_sample_transfer_test = utilities.predict_batches(expanded_test_X, model, in_samples_per_batch=64)
reconstructed_pixels_transfer_test, reconstructed_types_transfer_test = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_transfer_test)
print("MSE value for the " + second_model_name + " learning model, over the TESTING dataset is: ")
utilities.mean_square_error(reconstructed_pixels_transfer_test[0:len(X_test_RGB)], X_test_RGB)
if print_ssim_scores:
    print("SSIM is: ")
    utilities.ssim_comparison(reconstructed_pixels_transfer_test[0:len(X_test_RGB)], X_test_RGB)

# Now, Regional test data only:
encode_decode_sample_transfer_test_regional = utilities.predict_batches(expanded_full_X_HSV_regional,
                                                                        model, in_samples_per_batch=64)
reconstructed_pixels_transfer_test_regional, reconstructed_types_transfer_test_regional = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_transfer_test_regional)
print("MSE value for the " + second_model_name + " model, over the REGIONAL dataset is: ")
utilities.mean_square_error(reconstructed_pixels_transfer_test_regional, X_full_RGB_regional)
if print_ssim_scores:
    print("SSIM is: ")
    utilities.ssim_comparison(reconstructed_pixels_transfer_test_regional, X_full_RGB_regional)

# ATLAS EXPORTATION ####################################################

print("Exporting both TRAINING and TESTING reconstructed pokemon as an image.")
utilities.export_multi_type_atlas(X_full_RGB, reconstructed_pixels_original, reconstructed_pixels_transfer,
                                  name_prefix='_' + first_model_name + '_VS_' + second_model_name + '_')

utilities.export_multi_type_atlas(X_full_RGB_regional, reconstructed_pixels_original_test_regional,
                                  reconstructed_pixels_transfer_test_regional,
                                  name_prefix='Regional_' + first_model_name + '_VS_' + second_model_name + '_')

# Now, to export the testing samples only.
exporting_RGB = np.concatenate((X_full_RGB[827:974], X_full_RGB[1801:]), axis=0)
print("Exporting TESTING-ONLY reconstructed pokemon as an image.")
utilities.export_multi_type_atlas(exporting_RGB, reconstructed_pixels_original_test, reconstructed_pixels_transfer_test,
                                  name_prefix='_TESTONLY_' + first_model_name + '_VS_' + second_model_name + '_')
# print('The number of ORIGINAL types correct types were: ')
# correct_indices = utilities.export_types_csv(predicted_Y, reconstructed_types_original)
# print('The number of FAKE types correct types were: ')
# correct_indices = utilities.export_types_csv(new_types_array, reconstructed_types_fake)
