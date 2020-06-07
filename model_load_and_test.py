from __future__ import division, print_function, absolute_import

import numpy as np

import tensorflow as tf

import tflearn

import matplotlib.colors

import pokedataset32_vae_functions as utilities

X_full_HSV, Y_full_HSV = utilities.prepare_dataset_for_input_layer('pokedataset32_full_HSV_Two_Hot_Encoded.h5')

X_full_RGB, Y_full_RGB = utilities.prepare_dataset_for_input_layer('pokedataset32_full_RGB_Two_Hot_Encoded.h5')

# X, Y = utilities.prepare_dataset_for_input_layer('pokedataset32_train_HSV_Two_Hot_Encoded_Augmented.h5')

"""test_X, test_Y = utilities.prepare_dataset_for_input_layer('pokedataset32_train_HSV_Two_Hot_Encoded_Augmented.h5',
                                                           in_dataset_x_label='pokedataset32_X_test',
                                                           in_dataset_y_label='pokedataset32_Y_test')"""

Y_full_HSV = Y_full_HSV * 0.50
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
model.load("Saved models/model_Jun_05_optim_adam_loss_vae_loss_"
           "last_activ_sigmoid_latent_128_num_filters_256_256_decoder_width_8_2by2_BEST.tflearn")

# Add the fake types.
poke_type_1 = 'Fire'
poke_type_2 = 'None'
new_types_array = utilities.generate_all_one_type(len(X_full_HSV),
                                                  in_type=poke_type_1, in_second_type=poke_type_2)
new_types_array = np.reshape(np.asarray(new_types_array), newshape=[new_types_array.shape[0],
                                                                    utilities.pokemon_types_dim])
new_types_array = new_types_array * 5.50
# new_types_array = new_types_array + Y_full_HSV
expanded_fake_X = np.append(X_full_HSV, new_types_array, axis=1)

print("getting samples to show on screen.")
encode_decode_sample_original = utilities.predict_batches(expanded_full_X_HSV, model, in_number_of_chunks=10)
encode_decode_sample_fake = utilities.predict_batches(expanded_fake_X, model, in_number_of_chunks=10)

reconstructed_pixels_original, reconstructed_types_original = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_original)

reconstructed_pixels_fake, reconstructed_types_fake = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_fake)

print("Exporting reconstructed pokemon as an image.")
utilities.export_multi_type_atlas(X_full_RGB, reconstructed_pixels_original, reconstructed_pixels_fake,
                                  name_prefix='_FORCED_MULTI_TYPES_' + poke_type_1 + '_' + poke_type_2)
print('The number of ORIGINAL types correct types were: ')
correct_indices = utilities.export_types_csv(Y_full_RGB, reconstructed_types_original)
print('The number of FAKE types correct types were: ')
correct_indices = utilities.export_types_csv(new_types_array, reconstructed_types_fake)

# This is used to export an image only containing the ones whose types were correctly predicted by the NN.
"""correct_X_RGB = [X_full_RGB[i] for i in correct_indices]
correct_reconstructed_pixels = [reconstructed_pixels[i] for i in correct_indices]
utilities.export_as_atlas(correct_X_RGB, correct_reconstructed_pixels,
                          name_prefix='FORCED_TYPES_' + poke_type_1 + '_' + poke_type_2, name_annotations='correct')
"""
