from __future__ import division, print_function, absolute_import

import numpy as np
from numpy import array, newaxis, expand_dims
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import hsv_to_rgb
from scipy.stats import norm  # A normal continuous random variable.
# The location (loc) keyword specifies the mean. The scale (scale) keyword specifies the standard deviation.

import tensorflow as tf

import tflearn
import h5py
import pokedataset32_vae_functions as utilities
from PIL import Image
import colorsys
import math

# current_dataset = 'pokedataset'
# current_dataset = 'anime_faces_'

# We don't need all of those.
# X_full_HSV, Y_full_HSV, X_full_RGB, Y_full_RGB, X, Y, test_X, test_Y = utilities.ready_all_data_sets(current_dataset)

# X_full_HSV, Y_full_HSV = utilities.prepare_dataset_for_input_layer('pokedataset32_full_HSV_Two_Hot_Encoded.h5')

X_full_HSV, Y_full_HSV = utilities.prepare_dataset_for_input_layer(
    'pokedataset32_full_HSV_Two_Hot_Encoded.h5', in_dataset_x_label='pokedataset32_X',
    in_dataset_y_label='pokedataset32_Y')

# """
X_full_HSV_faces, Y_full_HSV_faces = utilities.prepare_dataset_for_input_layer(
    'anime_faces_32_train_HSV_Two_Hot_Encoded_Augmented.h5', in_dataset_x_label='anime_faces_32_X',
    in_dataset_y_label='anime_faces_32_Y')

X_test_HSV_faces, Y_test_HSV_faces = utilities.prepare_dataset_for_input_layer(
    'anime_faces_32_train_HSV_Two_Hot_Encoded_Augmented.h5', in_dataset_x_label='anime_faces_32_X_test',
    in_dataset_y_label='anime_faces_32_Y_test')
# """

"""
# FOR DEBUGGING PURPOSES:
X_full_HSV_faces, Y_full_HSV_faces = utilities.prepare_dataset_for_input_layer(
    'pokedataset32_train_HSV_Two_Hot_Encoded_Augmented.h5', in_dataset_x_label='pokedataset32_X',
    in_dataset_y_label='pokedataset32_Y')

X_test_HSV_faces, Y_test_HSV_faces = utilities.prepare_dataset_for_input_layer(
    'pokedataset32_train_HSV_Two_Hot_Encoded_Augmented.h5', in_dataset_x_label='pokedataset32_X_test',
    in_dataset_y_label='pokedataset32_Y_test')
"""

number_of_train_elements = len(X_full_HSV_faces)
X_full_HSV_faces = np.vstack((X_full_HSV_faces, X_test_HSV_faces))
Y_full_HSV_faces = np.vstack((Y_full_HSV_faces, Y_test_HSV_faces))

# FOR ANALYSIS PURPOSES ONLY:
Y_full_HSV = np.asarray(Y_full_HSV)
Y_full_HSV_sum = np.sum(Y_full_HSV, axis=0)  # 1948
"""

"""
total_types = (113, 72, 79, 96, 79, 96, 112, 118, 70, 155, 92, 59, 192, 88, 147, 81, 74, 225)
total_types = np.true_divide(total_types, 1948.0)
max_anime_faces_per_type = np.ceil(total_types * float(len(X_full_HSV_faces)))

"""
92, Mean HSV Bug is: [0.35312263 0.30633614 0.60429242] 
60, Mean HSV Dark is: [0.41979918 0.27389556 0.51082769] 
67, Mean HSV Dragon is: [0.43227957 0.28042493 0.56519744] 
64, Mean HSV Electric is: [0.32714509 0.30821285 0.62397882] 
60, Mean HSV Fairy is: [0.47378639 0.23620771 0.68770429] 
69, Mean HSV Fighting is: [0.32268149 0.28110769 0.59593414] 
79, Mean HSV Fire is: [0.19712324 0.37451623 0.62159068] 
114, Mean HSV Flying is: [0.35152004 0.28757596 0.58125078] 
57, Mean HSV Ghost is: [0.41071182 0.25533073 0.56997127] 
111, Mean HSV Grass is: [0.30854765 0.34115354 0.61638486] 
75, Mean HSV Ground is: [0.33227689 0.27230714 0.57046165] 
45, Mean HSV Ice is: [0.44489253 0.23348907 0.69471803] 
122, Mean HSV Normal is: [0.29801758 0.25264994 0.60827901] 
72, Mean HSV Poison is: [0.51062791 0.29442378 0.56234327] 
104, Mean HSV Psychic is: [0.43539857 0.26321797 0.63086559] 
68, Mean HSV Rock is: [0.34100713 0.23317025 0.5594408 ] 
64, Mean HSV Steel is: [0.39534165 0.22457826 0.55041926] 
154, Mean HSV Water is: [0.42870489 0.32105908 0.63795253]
"""
average_hsv_values = [[0.35312263, 0.30633614, 0.60429242],
                      [0.41979918, 0.27389556, 0.51082769],
                      [0.43227957, 0.28042493, 0.56519744],
                      [0.32714509, 0.30821285, 0.62397882],
                      [0.47378639, 0.23620771, 0.68770429],
                      [0.32268149, 0.28110769, 0.59593414],
                      [0.19712324, 0.37451623, 0.62159068],
                      [0.35152004, 0.28757596, 0.58125078],
                      [0.41071182, 0.25533073, 0.56997127],
                      [0.30854765, 0.34115354, 0.61638486],
                      [0.33227689, 0.27230714, 0.57046165],
                      [0.44489253, 0.23348907, 0.69471803],
                      [0.29801758, 0.25264994, 0.60827901],
                      [0.51062791, 0.29442378, 0.56234327],
                      [0.43539857, 0.26321797, 0.63086559],
                      [0.34100713, 0.23317025, 0.5594408],
                      [0.39534165, 0.22457826, 0.55041926],
                      [0.42870489, 0.32105908, 0.63795253]]


# IMPORTANT: we need both datasets loaded in [HSV] shape. not flattened.
reshaped_image_faces = np.reshape(X_full_HSV_faces, newshape=[len(X_full_HSV_faces), -1, 3])

type_mean_hsv_value_list = [(0, 0, 0)] * utilities.pokemon_types_dim


def stable_matching(in_images, in_types):
    square_distances_list = []
    for current_anime_face in in_images:
        current_mean = current_anime_face.mean(axis=0)
        square_distances = np.square(average_hsv_values - current_mean).mean(axis=1)
        square_distances_list.append(square_distances)
    # 0 means not proposed yet, 1 is provisionally engaged and -1 means turned down.
    proposal_checklist = np.zeros(shape=[len(in_images), utilities.pokemon_types_dim])
    faces_per_types_list = []
    for i_temp in range(0, utilities.pokemon_types_dim):
        faces_per_types_list.append(list())

    # These are M and W initialized to Free.
    free_faces_dict = dict()
    square_distances_list = np.asarray(square_distances_list)
    for i in range(0, len(in_images)):
        free_faces_dict[i] = (square_distances_list[i], [0] * utilities.pokemon_types_dim)

    engaged_faces = dict()

    print("Beginning stable matching process, please be patient.")
    continue_condition = True
    while len(free_faces_dict) > 0:
        # current_image = (free_faces_dict.keys())
        current_image_index = list(free_faces_dict.keys())[0]
        # print("Matching index: " + str(current_image_index))
        # if current_image_index > 8200:
        #    print("ERROR_APPROACHING")
        # Only the first element is needed per loop
        current_image_contents = free_faces_dict.pop(current_image_index)
        current_image_squared_distances = np.asarray(current_image_contents[0])
        # Which type has this image is engaged to (1), been turned down by (-1), and hasn't proposed yet (0).
        current_image_match_indicators = np.asarray(current_image_contents[1])
        # Propose to the next type available for it
        where_result = np.where(current_image_match_indicators == 0)
        available_types_to_match_indices = np.asarray(where_result).flatten()
        available_types_to_match = current_image_squared_distances[available_types_to_match_indices]
        available_type = available_types_to_match_indices[available_types_to_match.argmin()]
        if len(faces_per_types_list[available_type]) < max_anime_faces_per_type[available_type]:
            # Then, it is really available, so we pair them together.
            current_image_match_indicators[available_type] = 1  # We set it to 1 which means they are paired.
            # This should store the Index(ID) of the image and the HSV value
            faces_per_types_list[available_type].append(current_image_index)
            # Add it to the engaged list.
            engaged_faces[current_image_index] = (current_image_squared_distances, current_image_match_indicators)
        else:  # else, it means there exists some pair with this type.
            # Retrieve all the element indices present in that type list. Then get all of them from engaged_faces.
            possible_contenders = [engaged_faces[x_index] for x_index in faces_per_types_list[available_type]]
            possible_contenders_type_distance = np.asarray(possible_contenders)[:, 0, available_type]
            # NOTE: It is the MAX value since it's the most distant to the HSV mean.
            max_distance_element_index = possible_contenders_type_distance.argmax()
            max_distance_element_id = faces_per_types_list[available_type][max_distance_element_index]
            max_distance_element_value = possible_contenders_type_distance[max_distance_element_index]
            # If this type prefers to have the current_image_index over one that it already has:
            if current_image_squared_distances[available_type] < max_distance_element_value:
                # Now, this type prefers the current_image_index over minimum_element_index, so we swap them
                # minimum_element_index becomes free.
                # Retrieve the distances and match_indicators from the dict
                freed_element = engaged_faces[max_distance_element_id]
                freed_element[1][available_type] = -1  # Set to minus one, since it must no longer be matched to that.
                freed_element[0][available_type] = 1  # NOT NECESSARY, BUT FOR DEBUGGING PURPOSES.
                # Also delete it from the faces_per_type_list[available_type] list.
                # Restore it to the free_faces_dict
                free_faces_dict[max_distance_element_id] = freed_element
                engaged_faces.pop(max_distance_element_id)
                faces_per_types_list[available_type].remove(max_distance_element_id)

                # And current_image_index becomes engaged to this type
                current_image_match_indicators[available_type] = 1  # We set it to 1 which means they are paired.
                # This should store the Index(ID) of the image and the HSV value
                faces_per_types_list[available_type].append(current_image_index)
                # Add it to the engaged list.
                engaged_faces[current_image_index] = (current_image_squared_distances, current_image_match_indicators)
            else:
                current_image_match_indicators[available_type] = -1  # No point in repeating?
                free_faces_dict[current_image_index] = (current_image_squared_distances, current_image_match_indicators)
    return faces_per_types_list


image_per_type_lists = stable_matching(reshaped_image_faces, max_anime_faces_per_type)

print("PASSING TO STORE THEM IN ONE HOT ENCODING")

# Now, we use the image indices from faces_per_types_list and go for each type to assign the two-hot-encodings.
for i_current_type in range(0, utilities.pokemon_types_dim):
    for current_element in image_per_type_lists[i_current_type]:
        # All these indices must get a 2 in this position of their encodings
        Y_full_HSV_faces[current_element] = [0] * utilities.pokemon_types_dim
        Y_full_HSV_faces[current_element][i_current_type] = 2  #

print("Exporting the dataset")
# Now, we have to make a new dataset!
h5f = h5py.File('anime_faces_32_train_HSV_Two_Hot_Encoded_Augmented_With_Types.h5', 'w')
# These two lines below are used when the full data set is to be in one file.
h5f.create_dataset('anime_faces_32_X', data=X_full_HSV_faces[0:number_of_train_elements])
h5f.create_dataset('anime_faces_32_Y', data=Y_full_HSV_faces[0:number_of_train_elements])
h5f.create_dataset('anime_faces_32_X_test', data=X_test_HSV_faces)
h5f.create_dataset('anime_faces_32_Y_test', data=Y_full_HSV_faces[number_of_train_elements:])
h5f.close()

"""
# Now, we can compare all images from the non-labeled data set against the types
ssim_results = []
anime_faces_by_type = [utilities.pokemon_types_dim]
for unlabeled_image in X_full_HSV_faces:
    current_best_ssim = -1  # This is the actual [0,1] value returned by the ssim comparison.
    current_best_ssim_index = -1  # This is the index at which it was obtained the best ssim.
    # compare unlabeled_image to all
"""

# For each non-labeled image

# We run the SSIM comparison, and we save that value. We can either average it or some other way to measure.
# for all the images.







