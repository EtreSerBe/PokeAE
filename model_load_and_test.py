from __future__ import division, print_function, absolute_import

import numpy as np

import tensorflow as tf

import tflearn

import matplotlib.colors

import pokedataset32_vae_functions as utilities

# current_dataset = 'pokedataset'
current_dataset = 'anime_faces_'

# X and Y are not used in this file.
X_full_HSV, Y_full_HSV, X_full_RGB, Y_full_RGB, X, Y, test_X, test_Y = utilities.ready_all_data_sets(current_dataset)


Y_full_HSV = Y_full_HSV * 0.50
small_X = np.concatenate((X[0:200], test_X[0:200]), axis=0)
small_Y = np.concatenate((Y[0:200], test_Y[0:200]), axis=0)
len_X_div_2 = int(len(X)/2)
small_X_RGB = np.concatenate((X_full_RGB[0:200], X_full_RGB[len_X_div_2:len_X_div_2+200]), axis=0)
expanded_small_X = np.append(small_X, small_Y, axis=1)

expanded_full_X_HSV = np.append(X_full_HSV, Y_full_HSV, axis=1)
print("getting network to load model*******************")
network_instance = utilities.get_network()

network_instance = tflearn.regression(network_instance,
                                      optimizer='adam',
                                      metric='R2',
                                      loss=utilities.vae_loss,
                                      learning_rate=0.001)  # adagrad? #adadelta #nesterov did good,

model = tflearn.DNN(network_instance)

print("LOADING MODEL.")

# This hasn't been commited yet, due to network restrictions (AKA slow upload connection).
# Double check to have a folder with the correct path here.
model.load("Saved models/model_Jun_11_optim_adam_loss_vae_loss_"
           "last_activ_relu_latent_512_num_filters_512_512_decoder_width_8_2by2.tflearn")

predict_full_dataset = False
if predict_full_dataset:
    predicted_X = expanded_full_X_HSV
    predicted_Y = Y_full_RGB
else:
    predicted_X = expanded_small_X
    predicted_Y = small_Y

# Add the fake types.
poke_type_1 = 'Fire'
poke_type_2 = 'None'
new_types_array = utilities.generate_all_one_type(len(predicted_X),
                                                  in_type=poke_type_1, in_second_type=poke_type_2)
new_types_array = np.reshape(np.asarray(new_types_array), newshape=[new_types_array.shape[0],
                                                                    utilities.pokemon_types_dim])
new_types_array = new_types_array * 0.50
# new_types_array = new_types_array + Y_full_HSV
if predict_full_dataset:
    expanded_fake_X = np.append(X_full_HSV, new_types_array, axis=1)
    exporting_RGB = X_full_RGB
else:
    expanded_fake_X = np.append(small_X, new_types_array, axis=1)
    exporting_RGB = small_X_RGB

print("getting samples to show on screen.")
encode_decode_sample_original = utilities.predict_batches(predicted_X, model, in_samples_per_batch=64)
encode_decode_sample_fake = utilities.predict_batches(expanded_fake_X, model, in_samples_per_batch=64)

reconstructed_pixels_original, reconstructed_types_original = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_original)

reconstructed_pixels_fake, reconstructed_types_fake = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_fake)

print("Exporting reconstructed pokemon as an image.")
utilities.export_multi_type_atlas(exporting_RGB, reconstructed_pixels_original, reconstructed_pixels_fake,
                                  name_prefix='_FORCED_MULTI_TYPES_' + poke_type_1 + '_' + poke_type_2)
print('The number of ORIGINAL types correct types were: ')
correct_indices = utilities.export_types_csv(predicted_Y, reconstructed_types_original)
print('The number of FAKE types correct types were: ')
correct_indices = utilities.export_types_csv(new_types_array, reconstructed_types_fake)

# This is used to export an image only containing the ones whose types were correctly predicted by the NN.
"""correct_X_RGB = [X_full_RGB[i] for i in correct_indices]
correct_reconstructed_pixels = [reconstructed_pixels[i] for i in correct_indices]
utilities.export_as_atlas(correct_X_RGB, correct_reconstructed_pixels,
                          name_prefix='FORCED_TYPES_' + poke_type_1 + '_' + poke_type_2, name_annotations='correct')
"""
