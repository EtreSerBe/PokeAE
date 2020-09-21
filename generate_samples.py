from __future__ import division, print_function, absolute_import

import numpy as np

import tensorflow as tf

import tflearn

import matplotlib.colors
from scipy.stats import norm
import matplotlib.pyplot as plt

import pokedataset32_vae_functions as utilities

# 6 = Charizard
chosen_pokemon = 9  # Squirtle
chosen_pokemon = 13  # Caterpie
chosen_pokemon = 30  # Pikachu
chosen_pokemon = 146  # Jolteon
# chosen_pokemon = 161  # Dragonite
# chosen_pokemon = 252  # Porygon 2
chosen_pokemon_2 = 330  # Aaron
# chosen_pokemon = 427  # Jirachi
# chosen_pokemon = 622  # Darmanitan
# chosen_pokemon = 972  # shield doggo.

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

# This hasn't been commited yet, due to network restrictions (AKA slow upload connection).
# Double check to have a folder with the correct path here.
model_name = "_anime_V2_poke5"
model.load("saved_models/model_Jul_29_optim_adam_loss_vae_loss_"
           "last_activ_relu_latent_128_num_filters_512_1024_decoder_width_8" + model_name + ".tflearn")

print("MODEL SUCCESSFULLY LOADED.")

encoder_model = utilities.get_encoder_network(model)
generator_model = utilities.get_generative_network(model)
print("LOADED GENERATOR MODEL.")

black_images = np.zeros(shape=[1408, 3072])

num_samples = utilities.latent_dimension
mean = 0.00
std_dev = 1.00
range_variation = 20
"""x_axis = norm.ppf(np.linspace(0., 1., 100))
y_axis = norm.ppf(np.linspace(0., 1., 10))
one_per_latent = np.diag(np.ones(utilities.latent_dimension))*10.0"""
input_noise = np.random.normal(mean, std_dev, [utilities.latent_dimension])
input_noise_list = []
latent_dim_changes = []
for i in range(0, utilities.latent_dimension):
    # For each sample, make 10 of them
    for j in range(0, 11):
        temp_noise = np.copy(input_noise)
        temp_latent_change = np.zeros_like(temp_noise)
        temp_latent_change[i] = ((-range_variation / 2.0) + (j * range_variation / 10.0))
        latent_dim_changes.append(temp_latent_change)
        temp_noise[i] = temp_noise[i] + temp_latent_change[i]
        input_noise_list.append(temp_noise)

# input_noise = one_per_latent
"""# Add the fake types.
poke_type_1 = 'Fire'
poke_type_2 = 'None'
new_types_array = utilities.generate_all_one_type(num_samples,
                                                  in_type=poke_type_1, in_second_type=poke_type_2)
new_types_array = np.reshape(np.asarray(new_types_array), newshape=[new_types_array.shape[0],
                                                                    utilities.pokemon_types_dim])
new_types_array = new_types_array * 0.50
# new_types_array = new_types_array + Y_full_HSV
expanded_input_noise = np.append(input_noise, new_types_array, axis=1)"""

encoded_pokemon = utilities.predict_batches(expanded_full_X_HSV, encoder_model,
                                            in_samples_per_batch=64,
                                            in_input_name='input_images')

# Pick one pokemon and repeat it a number of times equal to the total noise samples
target_pokemon = encoded_pokemon[chosen_pokemon]
second_target_pokemon = encoded_pokemon[chosen_pokemon_2]
target_pokemon_difference = target_pokemon - second_target_pokemon
target_pokemon_difference = target_pokemon_difference/30.0  # Divided by the number of samples
interpolated_pokemon = []
for i in range(0, 30):
    current_interpolation = second_target_pokemon + (target_pokemon_difference * i)
    interpolated_pokemon.append(current_interpolation)
interpolated_transfer = utilities.predict_batches(interpolated_pokemon, generator_model,
                                                                     in_samples_per_batch=64,
                                                                     in_input_name='input_noise')

modified_encoded_pokemon = np.tile(target_pokemon, (len(latent_dim_changes), 1))
modified_encoded_pokemon = modified_encoded_pokemon + np.asarray(latent_dim_changes)

# Check how to do it in batches later.  input_noise_list
encode_decode_generated_samples_transfer = utilities.predict_batches(modified_encoded_pokemon, generator_model,
                                                                     in_samples_per_batch=64,
                                                                     in_input_name='input_noise')
# encode_decode_generated_samples = generator_model.predict({'input_noise': input_noise_list})

reconstructed_pixels_generated_transfer, reconstructed_types_generated_transfer = \
    utilities.reconstruct_pixels_and_types(encode_decode_generated_samples_transfer)

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

# Pick one pokemon and repeat it a number of times equal to the total noise samples
target_pokemon = encoded_pokemon[chosen_pokemon]
modified_encoded_pokemon = np.tile(target_pokemon, (len(latent_dim_changes), 1))
modified_encoded_pokemon = modified_encoded_pokemon + np.asarray(latent_dim_changes)

encode_decode_generated_samples = utilities.predict_batches(modified_encoded_pokemon, generator_model_transfer,
                                                            in_samples_per_batch=64, in_input_name='input_noise_1')

reconstructed_pixels_generated, reconstructed_types_generated = \
    utilities.reconstruct_pixels_and_types(encode_decode_generated_samples)

utilities.export_latent_exploration_atlas(reconstructed_pixels_generated_transfer, reconstructed_pixels_generated,
                                          name_prefix='LatentExploration_TransferVSOriginal_',
                                          name_annotations='_range_var_' + str(range_variation))

# print('The number of ORIGINAL types correct types were: ')
# correct_indices = utilities.export_types_csv(Y_full_RGB[0:100], reconstructed_types_generated)

f, a = plt.subplots(1, 10, squeeze=False)  # figsize=(50, 2),
for i in range(10):
    temp = [[ii] for ii in list(reconstructed_pixels_generated[i])]
    a[0][i].imshow(np.reshape(temp, (32, 32, 3)))
    print("Types for Pokemon " + str(i) + " are: ")
    utilities.print_pokemon_types(reconstructed_types_generated[i])

f.show()
plt.draw()
plt.waitforbuttonpress()

# This is used to export an image only containing the ones whose types were correctly predicted by the NN.
"""correct_X_RGB = [X_full_RGB[i] for i in correct_indices]
correct_reconstructed_pixels = [reconstructed_pixels[i] for i in correct_indices]
utilities.export_as_atlas(correct_X_RGB, correct_reconstructed_pixels,
                          name_prefix='FORCED_TYPES_' + poke_type_1 + '_' + poke_type_2, name_annotations='correct')
"""
