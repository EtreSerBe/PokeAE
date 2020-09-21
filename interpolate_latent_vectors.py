from __future__ import division, print_function, absolute_import

import numpy as np

import tensorflow as tf

import tflearn

import matplotlib.colors
from scipy.stats import norm
import matplotlib.pyplot as plt

import pokedataset32_vae_functions as utilities

# 6 = Charizard
chosen_pokemon_2 = 9  # Squirtle
# chosen_pokemon_2 = 13  # Caterpie
chosen_pokemon = 30  # Pikachu
chosen_pokemon = 146  # Jolteon
# chosen_pokemon = 161  # Dragonite
# chosen_pokemon_2 = 252  # Porygon 2
# chosen_pokemon_2 = 330  # Aaron
# chosen_pokemon = 427  # Jirachi
# chosen_pokemon_2 = 622  # Darmanitan
# chosen_pokemon = 972  # shield doggo.
interpolation_intervals = 7

X_full_HSV, Y_full_HSV = \
    utilities.prepare_dataset_for_input_layer("pokedataset32_full_HSV_Two_Hot_Encoded.h5")

Y_full_HSV = Y_full_HSV * 0.5
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

model_name = "_anime_V2_poke5"
model.load("saved_models/model_Jul_29_optim_adam_loss_vae_loss_"
           "last_activ_relu_latent_128_num_filters_512_1024_decoder_width_8" + model_name + ".tflearn")
print("MODEL SUCCESSFULLY LOADED.")
encoder_model = utilities.get_encoder_network(model)
generator_model = utilities.get_generative_network(model)
print("LOADED GENERATOR MODEL.")

encoded_pokemon = utilities.predict_batches(expanded_full_X_HSV, encoder_model,
                                            in_samples_per_batch=64,
                                            in_input_name='input_images')

# Pick one pokemon and repeat it a number of times equal to the total noise samples
target_pokemon = encoded_pokemon[chosen_pokemon]
second_target_pokemon = encoded_pokemon[chosen_pokemon_2]
target_pokemon_difference = target_pokemon - second_target_pokemon
target_pokemon_difference = target_pokemon_difference / float(interpolation_intervals)  # Divided by the number of samples
interpolated_pokemon = []
for i in range(0, interpolation_intervals):
    current_interpolation = second_target_pokemon + (target_pokemon_difference * i)
    interpolated_pokemon.append(current_interpolation)
interpolated_transfer = utilities.predict_batches(interpolated_pokemon, generator_model,
                                                  in_samples_per_batch=64,
                                                  in_input_name='input_noise')

reconstructed_pixels_generated_transfer, reconstructed_types_generated_transfer = \
    utilities.reconstruct_pixels_and_types(interpolated_transfer)
##########################################################################
second_model_name = "_regular_V3_no_noise2"
model.load("saved_models/model_Jul_29_optim_adam_loss_vae_loss_"
           "last_activ_relu_latent_128_num_filters_512_1024_decoder_width_8" + second_model_name + ".tflearn",
           weights_only=True)
encoder_model_transfer = utilities.get_encoder_network(model)
generator_model_transfer = utilities.get_generative_network(model)
print("LOADED GENERATOR MODEL.")
encoded_pokemon = utilities.predict_batches(expanded_full_X_HSV, encoder_model_transfer,
                                            in_samples_per_batch=64,
                                            in_input_name='input_images_1')

target_pokemon = encoded_pokemon[chosen_pokemon]
second_target_pokemon = encoded_pokemon[chosen_pokemon_2]
target_pokemon_difference = target_pokemon - second_target_pokemon
target_pokemon_difference = target_pokemon_difference / float(interpolation_intervals)  # Divided by the number of samples
interpolated_pokemon = []
for i in range(0, interpolation_intervals):
    current_interpolation = second_target_pokemon + (target_pokemon_difference * i)
    interpolated_pokemon.append(current_interpolation)
interpolated_transfer = utilities.predict_batches(interpolated_pokemon, generator_model_transfer,
                                                  in_samples_per_batch=64,
                                                  in_input_name='input_noise_1')

reconstructed_pixels_generated, reconstructed_types_generated = \
    utilities.reconstruct_pixels_and_types(interpolated_transfer)

utilities.export_latent_exploration_atlas(reconstructed_pixels_generated_transfer, reconstructed_pixels_generated,
                                          name_prefix='Interpolation_T_VS_Orig_',
                                          name_annotations=str(chosen_pokemon) + '_to_' + str(chosen_pokemon_2))

# print('The number of ORIGINAL types correct types were: ')
# correct_indices = utilities.export_types_csv(Y_full_RGB[0:100], reconstructed_types_generated)

"""f, a = plt.subplots(1, 10, squeeze=False)  # figsize=(50, 2),
for i in range(10):
    temp = [[ii] for ii in list(reconstructed_pixels_generated[i])]
    a[0][i].imshow(np.reshape(temp, (32, 32, 3)))
    print("Types for Pokemon " + str(i) + " are: ")
    utilities.print_pokemon_types(reconstructed_types_generated[i])

f.show()
plt.draw()
plt.waitforbuttonpress()"""
