from __future__ import division, print_function, absolute_import

import numpy as np

import tensorflow as tf

import tflearn

import matplotlib.colors
from scipy.stats import norm
import matplotlib.pyplot as plt

import pokedataset32_vae_functions as utilities

print("getting network to load model*******************")
network_instance = utilities.get_network()

network_instance = tflearn.regression(network_instance,
                                      optimizer='adam',
                                      # optimizer='rmsprop',
                                      metric='R2',
                                      # loss='mean_square',
                                      loss=utilities.vae_loss,
                                      learning_rate=0.001)  # adagrad? #adadelta #nesterov did good,

model = tflearn.DNN(network_instance)

print("LOADING MODEL.")

# This hasn't been commited yet, due to network restrictions (AKA slow upload connection).
# Double check to have a folder with the correct path here.
model.load("saved_models/model_Jun_16_optim_adam_loss_vae_loss_"
           "last_activ_relu_latent_64_num_filters_512_128_decoder_width_8_V3.tflearn")

print("MODEL SUCCESSFULLY LOADED.")

generator_model = utilities.get_generative_network(model)
print("LOADED GENERATOR MODEL.")

num_samples = utilities.latent_dimension
mean = 0.50
std_dev = 1.0
x_axis = norm.ppf(np.linspace(0., 1., 100))
y_axis = norm.ppf(np.linspace(0., 1., 10))
one_per_latent = np.diag(np.ones(utilities.latent_dimension))
input_noise = np.random.normal(mean, std_dev, [num_samples, utilities.latent_dimension])
# input_noise = input_noise + one_per_latent
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

# Check how to do it in batches later.
encode_decode_generated_samples = generator_model.predict({'input_noise': input_noise})

reconstructed_pixels_generated, reconstructed_types_generated = \
    utilities.reconstruct_pixels_and_types(encode_decode_generated_samples)

utilities.export_as_atlas(reconstructed_pixels_generated, reconstructed_pixels_generated,
                          name_prefix='GENERATED_SAMPLES_',
                          name_annotations='_Mean_' + str(mean) + '_stddev_' + str(std_dev))

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
# input('Press E to exit')
plt.waitforbuttonpress()

# This is used to export an image only containing the ones whose types were correctly predicted by the NN.
"""correct_X_RGB = [X_full_RGB[i] for i in correct_indices]
correct_reconstructed_pixels = [reconstructed_pixels[i] for i in correct_indices]
utilities.export_as_atlas(correct_X_RGB, correct_reconstructed_pixels,
                          name_prefix='FORCED_TYPES_' + poke_type_1 + '_' + poke_type_2, name_annotations='correct')
"""
