from __future__ import division, print_function, absolute_import

import numpy as np

import tensorflow as tf

import tflearn

import matplotlib.colors

import pokedataset32_vae_functions as utilities

current_dataset = 'pokedataset'
# current_dataset = 'anime_faces_'

# X_full_HSV, Y_full_HSV, X_full_RGB, Y_full_RGB, X, Y, test_X, test_Y = utilities.ready_all_data_sets(current_dataset)
use_anime_with_types = False
if not use_anime_with_types or current_dataset == 'pokedataset':
    X_full_HSV, Y_full_HSV, X_full_RGB, Y_full_RGB, X, Y, test_X, test_Y = utilities.ready_all_data_sets(
        current_dataset)

    X_full_HSV_regional, Y_full_HSV_regional = \
        utilities.prepare_dataset_for_input_layer("pokedataset32_full_HSV_Two_Hot_Encoded_Regional.h5")
    X_full_HSV_non_regional, Y_full_HSV_non_regional = \
        utilities.prepare_dataset_for_input_layer("pokedataset32_full_HSV_Two_Hot_Encoded_Regional.h5",
                                                  in_dataset_x_label="pokedataset32_X_original",
                                                  in_dataset_y_label="pokedataset32_Y_original")
    Y_full_HSV_regional = Y_full_HSV_regional * 0.5
    Y_full_HSV_non_regional = Y_full_HSV_non_regional * 0.5
    expanded_full_X_HSV_regional = np.append(X_full_HSV_regional, Y_full_HSV_regional, axis=1)
    expanded_full_X_HSV_non_regional = np.append(X_full_HSV_non_regional, Y_full_HSV_non_regional, axis=1)
    X_full_RGB_regional, Y_full_RGB_regional = \
        utilities.prepare_dataset_for_input_layer("pokedataset32_full_RGB_Two_Hot_Encoded_Regional.h5")
    X_full_RGB_non_regional, Y_full_RGB_non_regional = \
        utilities.prepare_dataset_for_input_layer("pokedataset32_full_RGB_Two_Hot_Encoded_Regional.h5",
                                                  in_dataset_x_label="pokedataset32_X_original",
                                                  in_dataset_y_label="pokedataset32_Y_original")
else:
    X, Y = utilities.prepare_dataset_for_input_layer(
        'anime_faces_32_train_HSV_Two_Hot_Encoded_Augmented_With_Types.h5', in_dataset_x_label='anime_faces_32_X',
        in_dataset_y_label='anime_faces_32_Y')
    test_X, test_Y = utilities.prepare_dataset_for_input_layer(
        'anime_faces_32_train_HSV_Two_Hot_Encoded_Augmented_With_Types.h5', in_dataset_x_label='anime_faces_32_X_test',
        in_dataset_y_label='anime_faces_32_Y_test')
    X_full_RGB, Y_full_RGB = utilities.prepare_dataset_for_input_layer(
        'anime_faces_32_full_RGB_Two_Hot_Encoded.h5', in_dataset_x_label='anime_faces_32_X',
        in_dataset_y_label='anime_faces_32_Y')

    X_first_half = X[0:int(len(X) / 2)]
    Y_first_half = Y[0:int(len(Y) / 2)]
    test_X_first_half = test_X[0:int(len(test_X) / 2)]
    test_Y_first_half = test_Y[0:int(len(test_Y) / 2)]
    """X_second_half = X[int(len(X) / 2):]
    Y_second_half = Y[int(len(Y) / 2):]
    test_X_second_half = test_X[int(len(test_X) / 2):]
    test_Y_second_half = test_Y[int(len(test_Y) / 2):]"""
    X_full_HSV = np.concatenate((X_first_half, test_X_first_half), axis=0)
    Y_full_HSV = np.concatenate((Y_first_half, test_Y_first_half), axis=0)
    Y_full_RGB = Y_full_HSV  # Replace it, since RGB was not saved with types.


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
                                      learning_rate=0.0001)  # adagrad? #adadelta #nesterov did good,

model = tflearn.DNN(network_instance)

print("LOADING MODEL.")

# This hasn't been commited yet, due to network restrictions (AKA slow upload connection).
# Double check to have a folder with the correct path here.}
model_name = "_regular_V3_no_noise2"
model_name = "_anime_V2_poke5"
model.load("saved_models/model_Jul_29_optim_adam_loss_vae_loss_"
           "last_activ_relu_latent_128_num_filters_512_1024_decoder_width_8" + model_name + ".tflearn")

# Ratata, Raichu, Ninetales, Slowbro, Marowak, (Galar Meowth is 10)
indices_to_predict = [0, 2, 6, 18, 23]

predict_full_dataset = True
use_regional_types = True
if predict_full_dataset:
    if not use_regional_types:
        predicted_X = expanded_full_X_HSV
        predicted_Y = Y_full_RGB
    else:
        predicted_X = expanded_full_X_HSV_non_regional  # Non-regional to regional
        predicted_Y = Y_full_HSV_non_regional
        # predicted_X = expanded_full_X_HSV_regional  # Regional to non-regional
        # predicted_Y = Y_full_HSV_regional
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
if use_regional_types:
    new_types_array = Y_full_HSV_regional  # Non_regional to regional
    # new_types_array = Y_full_HSV_non_regional  # Regional to non-regional
new_types_array = new_types_array * 20.0
# new_types_array = new_types_array - Y_full_HSV
# new_types_array = new_types_array * 1.000


if predict_full_dataset:
    if not use_regional_types:
        expanded_fake_X = np.append(X_full_HSV, new_types_array, axis=1)
        # expanded_fake_X = expanded_fake_X * 2.0
        exporting_RGB = X_full_RGB
    else:
        # Non-regional to regional
        expanded_fake_X = np.append(X_full_HSV_non_regional, new_types_array, axis=1)
        exporting_RGB = X_full_RGB_non_regional
        # Regional to Non-regional
        # expanded_fake_X = np.append(X_full_HSV_regional, new_types_array, axis=1)
        # exporting_RGB = X_full_RGB_regional
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

wanted_indices_RGB = [exporting_RGB[i] for i in indices_to_predict]
wanted_indices_pixels_original = [reconstructed_pixels_original[i] for i in indices_to_predict]
wanted_indices_pixels_fake = [reconstructed_pixels_fake[i] for i in indices_to_predict]

print("MSE value for the " + model_name + " model, over the Original to Regional indices is: ")
utilities.mean_square_error(reconstructed_pixels_fake, X_full_RGB_regional)
print("SSIM is: ")
utilities.ssim_comparison(reconstructed_pixels_fake, X_full_RGB_regional)

print("Exporting reconstructed pokemon as an image.")
utilities.export_multi_type_atlas(wanted_indices_RGB,
                                  wanted_indices_pixels_original,
                                  wanted_indices_pixels_fake,
                                  name_prefix='Original_to_regional_' + poke_type_1 + '_' + poke_type_2)
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
