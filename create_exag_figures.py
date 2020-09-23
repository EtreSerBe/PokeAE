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
Y_full_RGB_regional = Y_full_HSV_regional * 0.5
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

# Fire swap for EXAG 2020
indices_to_predict = [161, 169, 837]
# Water swap for EXAG 2020
indices_to_predict = [6, 84, 837]
# Grass swap for EXAG 2020, + 974 to get the white-background samples.
indices_to_predict = [6 + 974, 250 + 974, 926 + 974]

# Reconstructs for image 3 in the EXAG 2020 paper.
indices_to_predict = [6, 640, 30, 837, 865]

# Ratata, Raichu, Ninetales, Slowbro, Marowak, (Galar Meowth is 10), for the regional variants figure.
indices_to_predict = [24, 31, 43, 86, 113]
use_regional_types = True

originals = [X_full_HSV[i] for i in indices_to_predict]
originals_types = [Y_full_HSV[i] for i in indices_to_predict]
expanded_originals = np.append(originals, originals_types, axis=1)
originals_RGB = [X_full_RGB[i] for i in indices_to_predict]
regional_indices = [0, 2, 6, 18, 23]  # Rattata, Raichu, Ninetales, Slowbro and Marowak in the regional version.
originals_RGB_only_for_regional = [X_full_RGB[i] for i in indices_to_predict]
originals_RGB_regional = [X_full_RGB_regional[i] for i in regional_indices]
originals_RGB_regional_types = [Y_full_RGB_regional[i] for i in regional_indices]
reconstructed_transfer = []
swapped_transfer = []
reconstructed_non_transfer = []
swapped_non_transfer = []

# Add the fake types.
poke_type_1 = 'Grass'
poke_type_2 = 'None'
new_types_array = utilities.generate_all_one_type(len(originals),
                                                  in_type=poke_type_1, in_second_type=poke_type_2)
new_types_array = np.reshape(np.asarray(new_types_array), newshape=[new_types_array.shape[0],
                                                                    utilities.pokemon_types_dim])
new_types_array = new_types_array * 10.0
originals_fake_type = np.append(originals, new_types_array, axis=1)

network_instance = tflearn.regression(network_instance,
                                      optimizer='adam',
                                      metric='R2',
                                      loss=utilities.vae_loss,
                                      learning_rate=0.00001)  # adagrad? #adadelta #nesterov did good,

model = tflearn.DNN(network_instance)
print("LOADING MODEL.")

#####################
# TRANSFER RECONSTRUCTED
# Now, we can load the transfer learning model.
# second_model_name = "_V3_noise2"  # 21 of july  # "_V3_noise4" THIS WAS A GOOD MODEL.
first_model_name = "_anime_V2_poke5"
model.load("saved_models/model_Jul_29_optim_adam_loss_vae_loss_"
           "last_activ_relu_latent_128_num_filters_512_1024_decoder_width_8" + first_model_name + ".tflearn")

encode_decode_sample_transfer = utilities.predict_batches(expanded_originals, model, in_samples_per_batch=64)
reconstructed_transfer, reconstructed_types_transfer = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_transfer)

###################
# SWAPPED TRANSFER
encode_decode_sample_transfer = utilities.predict_batches(originals_fake_type, model, in_samples_per_batch=64)
swapped_transfer, reconstructed_types_transfer = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_transfer)

###################
# RECONSTRUCTED NON-TRANSFER
second_model_name = "_regular_V3_no_noise2"  # This is the one for the encoder-only part. From Jul_28
# This hasn't been commited yet, due to network restrictions (AKA slow upload connection).
# Double check to have a folder with the correct path here.
model.load("saved_models/model_Jul_29_optim_adam_loss_vae_loss_"
           "last_activ_relu_latent_128_num_filters_512_1024_decoder_width_8" + second_model_name + ".tflearn")

encode_decode_sample_original = utilities.predict_batches(expanded_originals, model, in_samples_per_batch=64)
reconstructed_non_transfer, reconstructed_types_original = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_original)

###############################
# SWAPPED NON-TRANSFER
encode_decode_sample_original = utilities.predict_batches(originals_fake_type, model, in_samples_per_batch=64)
swapped_non_transfer, reconstructed_types_original = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_original)

##############################
# Original to regional figure
if use_regional_types:
    new_types_array = np.asarray(originals_RGB_regional_types)  # Non_regional to regional
    new_types_array = new_types_array * 30.0
originals_fake_type = np.append(originals, new_types_array, axis=1)
encode_decode_sample_original = utilities.predict_batches(originals_fake_type, model, in_samples_per_batch=64)
swapped_non_transfer_regional, reconstructed_types_original_regional = \
    utilities.reconstruct_pixels_and_types(encode_decode_sample_original)

# ATLAS EXPORTATION ####################################################

exporting_images = [originals_RGB, reconstructed_transfer, swapped_transfer, reconstructed_non_transfer,
                    swapped_non_transfer]

# For figure 3 in the EXAG 2020 paper.
exporting_images = [originals_RGB, reconstructed_transfer, reconstructed_non_transfer]

# For the original to regional figure in the EXAG 2020 paper.
exporting_images = [originals_RGB_only_for_regional, swapped_non_transfer_regional, originals_RGB_regional]

print("Exporting both TRAINING and TESTING reconstructed pokemon as an image.")
# Format used for type-swap figures.
# utilities.export_specific_atlas(exporting_images, in_name_annotations="GRASS_")

# This parameter is used to flip the way they are placed.
utilities.export_specific_atlas(exporting_images, in_list_per_column=False, in_name_annotations="Regionals")
