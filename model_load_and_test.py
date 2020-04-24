from __future__ import division, print_function, absolute_import

import numpy as np

import tensorflow as tf

import tflearn

import matplotlib.colors

import PokeAE.pokedataset32_vae_functions as utilities

# We don't need the Ys.
X_full_HSV, Y_full_HSV = utilities.prepare_dataset_for_input_layer('pokedataset32_full_HSV.h5')

# We don't need the Ys.
X_full_RGB, Y_full_RGB = utilities.prepare_dataset_for_input_layer('pokedataset32_full_RGB.h5')

X, Y = utilities.prepare_dataset_for_input_layer('pokedataset32_12_3_HSV_Augmented.h5')

test_X, test_Y = utilities.prepare_dataset_for_input_layer('pokedataset32_12_3_HSV_Augmented.h5',
                                                           in_dataset_x_label='pokedataset32_X_test',
                                                           in_dataset_y_label='pokedataset32_Y_test')

print("getting network to load model*******************")
network_instance = utilities.get_network()

model = tflearn.DNN(network_instance)

print("LOADING MODEL.")

# This hasn't been commited yet, due to network restrictions (AKA slow upload connection).
# Double check to have a folder with the correct path here.
model.load("Saved models/pokedatamodel32_April_20_1.tflearn")

# Add the fake types.
new_types_array = utilities.generate_all_one_type(len(X_full_HSV),
                                                  in_type="Fire", in_second_type="None")
new_types_array = np.reshape(np.asarray(new_types_array), newshape=[new_types_array.shape[0],
                                                                    utilities.pokemon_types_dim])
expanded_fake_X = np.append(X_full_HSV, new_types_array, axis=1)

print("getting samples to show on screen.")
encode_decode_sample = model.predict(expanded_fake_X)

reconstructed_pixels = []
reconstructed_types = []
reshaped_sample = []

for i in range(0, len(encode_decode_sample)):
    sample = encode_decode_sample[i][0:3072]
    reshaped_sample = np.reshape(sample, [1024, 3])
    # https://matplotlib.org/api/_as_gen/matplotlib.colors.hsv_to_rgb.html#matplotlib.colors.hsv_to_rgb
    reshaped_sample = matplotlib.colors.hsv_to_rgb(reshaped_sample)
    pixel_list = reshaped_sample.flatten()
    reconstructed_pixels.append(pixel_list)
    reshaped_types = np.reshape(encode_decode_sample[i][3072:3108], [2, 18])
    reconstructed_types.append(reshaped_types)

print("Exporting reconstructed pokemon as an image.")
utilities.export_as_atlas(X_full_RGB, reconstructed_pixels)
# correct_indices = utilities.export_types_csv(Y_full_RGB, reconstructed_types)
correct_indices = utilities.export_types_csv(new_types_array, reconstructed_types)

# This is used to export an image only containing the ones whose types were correctly predicted by the NN.
correct_X_RGB = [X_full_RGB[i] for i in correct_indices]
correct_reconstructed_pixels = [reconstructed_pixels[i] for i in correct_indices]
utilities.export_as_atlas(correct_X_RGB, correct_reconstructed_pixels, name_annotations='correct')

"""
# I used this before to show the results, but now I have the whole image being saved.
print("PREPARING TO SHOW IMAGE")
# Compare original images with their reconstructions.
f, a = plt.subplots(2, 20, figsize=(20, 2), squeeze=False)  # figsize=(50, 2),
for i in range(20):
    # reshaped_pokemon = np.multiply(reshaped_pokemon, 255.0)
    reshaped_pokemon = np.reshape(np.asarray(X[i]), [1024, 3])
    RGBOriginal = matplotlib.colors.hsv_to_rgb(reshaped_pokemon)
    RGBOriginal = np.asarray(RGBOriginal).flatten()
    temp = [[ii] for ii in list(RGBOriginal)]  # WTH? Python, you're drunk haha.
    print("ORIGINAL Types for Pokemon " + str(i) + " are: ")
    pokedataset32_vae_functions.print_pokemon_types(Y[i])
    a[0][i].imshow(np.reshape(temp, (32, 32, 3)))
    temp = [[ii] for ii in list(reconstructed_pixels[i])]
    a[1][i].imshow(np.reshape(temp, (32, 32, 3)))
    print("Types for Pokemon " + str(i) + " are: ")
    pokedataset32_vae_functions.print_pokemon_types(reconstructed_types[i])
f.show()
plt.draw()
plt.waitforbuttonpress()
"""
