from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import tflearn
import matplotlib.pyplot as plt
import matplotlib.colors
import pokedataset32_vae_functions as utilities

X_full_HSV_Type_Swapped, Y_full_HSV_Type_Swapped = \
    utilities.prepare_dataset_for_input_layer('pokedataset32_full_HSV_Two_Hot_Encoded_Type_Swapped.h5')

X_full_RGB_Type_Swapped, Y_full_RGB_Type_Swapped = \
    utilities.prepare_dataset_for_input_layer('pokedataset32_full_RGB_Two_Hot_Encoded_Type_Swapped.h5')

Y_full_HSV_Type_Swapped = np.reshape(np.asarray(Y_full_HSV_Type_Swapped),
                                     newshape=[Y_full_HSV_Type_Swapped.shape[0], utilities.pokemon_types_dim])

expanded_full_X_HSV_Type_Swapped = np.append(X_full_HSV_Type_Swapped, Y_full_HSV_Type_Swapped, axis=1)

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
model.load("Saved models/pokedatamodel32_May_7_1_adam_vae_loss_sigmoid_latent48_FC_228_128_V2.tflearn")

print("getting samples to show on screen.")
encode_decode_sample = model.predict(expanded_full_X_HSV_Type_Swapped)

reconstructed_pixels = []
reconstructed_types = []

reconstructed_pixels, reconstructed_types = utilities.reconstruct_pixels_and_types(encode_decode_sample)


print("Exporting reconstructed pokemon as an image.")
utilities.export_as_atlas(X_full_RGB_Type_Swapped, reconstructed_pixels, name_prefix='Type_Swapped_')
correct_indices = utilities.export_types_csv(Y_full_HSV_Type_Swapped, reconstructed_types)

# This is used to export an image only containing the ones whose types were correctly predicted by the NN.
correct_X_RGB = [X_full_RGB_Type_Swapped[i] for i in correct_indices]
correct_reconstructed_pixels = [reconstructed_pixels[i] for i in correct_indices]
utilities.export_as_atlas(correct_X_RGB, correct_reconstructed_pixels, name_annotations='correct',
                          name_prefix='Type_Swapped_')

# Compare original images with their reconstructions.
f, a = plt.subplots(2, 20, figsize=(20, 2), squeeze=False)  # figsize=(50, 2),
for i in range(20):
    # reshaped_pokemon = np.multiply(reshaped_pokemon, 255.0)
    reshaped_pokemon = np.reshape(np.asarray(X_full_RGB_Type_Swapped[-(i+1)]), [1024, 3])
    reshaped_pokemon = np.asarray(reshaped_pokemon).flatten()
    temp = [[ii] for ii in list(reshaped_pokemon)]  # WTH? Python, you're drunk haha.
    print("ORIGINAL Types for Pokemon " + str(i) + " are: ")
    utilities.print_pokemon_types(Y_full_RGB_Type_Swapped[-(i+1)])
    a[0][i].imshow(np.reshape(temp, (32, 32, 3)))
    temp = [[ii] for ii in list(reconstructed_pixels[-(i+1)])]
    a[1][i].imshow(np.reshape(temp, (32, 32, 3)))
    print("Types for Pokemon " + str(i) + " are: ")
    utilities.print_pokemon_types(reconstructed_types[-(i+1)])
f.show()
plt.draw()
# input('Press E to exit')
plt.waitforbuttonpress()
