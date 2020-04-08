
import numpy as np
import math
from PIL import Image
from datetime import datetime
import h5py
import imgaug as ia
import imgaug.augmenters as iaa
import operator

type_to_categorical = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying',
                       'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water']


def prepare_dataset_for_input_layer(in_h5f_dataset_name, in_dataset_x_label="pokedataset32_X",
                                    in_dataset_y_label="pokedataset32_Y"):
    h5f_obj = h5py.File(in_h5f_dataset_name, 'r')
    h5f_x_values = h5f_obj[in_dataset_x_label]
    h5f_y_values = h5f_obj[in_dataset_y_label]
    h5f_y_values = np.reshape(np.asarray(h5f_y_values), newshape=[h5f_y_values.shape[0], 18 * 2])
    # We return them ready to be appended like this: "expanded_X = np.append(X, Y, axis=1)"
    return h5f_x_values, h5f_y_values


def image_flip_left_right(in_image_list, in_types_list):
    out_all_images = in_image_list
    for i_image in in_image_list:
        pixels = i_image[0:3072]
        reshaped_original = np.reshape(pixels, newshape=[32, 32, 3])
        # First, get the flipped left-right.
        flipped_image = np.fliplr(reshaped_original).flatten()
        out_all_images = np.vstack((out_all_images, flipped_image))

    out_types = in_types_list
    out_types = np.vstack((out_types, out_types))  # The types are in the same order as the input, only duplicate them.
    print("total number of images after augmentation: " + str(out_all_images.shape))
    return out_all_images, out_types


def image_augmentation(in_image_list):
    out_all_images = in_image_list
    for i_image in in_image_list:
        pixels = i_image[0:3072]
        # print(str(len(pixels)))
        types = i_image[3071:-1]
        reshaped_original = np.reshape(pixels, newshape=[32, 32, 3])
        # First, get the flipped left-right.
        flipped_image = np.fliplr(reshaped_original).flatten()
        flipped_image = np.append(flipped_image, types)
        print(len(flipped_image))
        out_all_images = np.vstack((out_all_images, flipped_image))

    print("total number of images after augmentation: " + str(out_all_images.shape))
    return out_all_images


def print_pokemon_types(types, in_print_all=True):
    types_as_strings = []
    flat_types = np.asarray(types).flatten()
    flat_types = np.reshape(flat_types, newshape=[2, 18])
    index_and_value = {}
    types_indices = []
    types_values = []  # Floating point values [0 to 1]
    for typearray in flat_types:
        for i in range(0, 18):
            if typearray[i] >= 0.1:
                index_and_value[i] = typearray[i]
                types_as_strings.append(type_to_categorical[i] + " : " + str(typearray[i]))
    if in_print_all:
        print(types_as_strings)  # Print them and exit the function, you could also retrieve them
    return index_and_value


def export_as_atlas(in_image_list, in_reconstructed_image_list, image_width=32, image_height=32, num_channels=3,
                    name_annotations='standard'):
    num_elements = len(in_image_list)
    if num_elements == 0:
        return
    print('Number or elements in in_image_list is: ' + str(num_elements))
    rows = math.ceil(math.sqrt(num_elements))  # ceil to be the highest integer enough.
    print('number of Rows and COlumns to have is: ' + str(rows))
    row_counter = 0
    column_counter = 0
    # Make it big enough to put the original above the reconstructed. (That's why multiplied by 2)
    atlas_image = Image.new('RGB', (image_width*rows, image_height*rows*2), (0, 0, 0))

    for original, reconstructed in zip(in_image_list, in_reconstructed_image_list):
        """if column_counter + (row_counter*rows) >= num_elements:
            break  # This is to stop it as soon as"""

        reshaped_image = np.reshape(
            np.uint8(np.multiply(original.flatten(), 255.)),
            [image_width, image_height, num_channels])

        reshaped_reconstructed = np.reshape(
            np.uint8(np.multiply(reconstructed.flatten(), 255.)),
            [image_width, image_height, num_channels])
        #reshaped_reconstructed = np.asarray(reshaped_reconstructed)

        offset = (column_counter*image_width, row_counter*image_height*2)
        im_original = Image.fromarray(reshaped_image, 'RGB')
        atlas_image.paste(im_original, offset)

        offset = (column_counter * image_width, row_counter * image_height * 2 + image_height)
        im_reconstructed = Image.fromarray(reshaped_reconstructed, 'RGB')
        atlas_image.paste(im_reconstructed, offset)
        column_counter += 1
        # Go to the next row.
        if column_counter == rows:
            column_counter = 0
            row_counter += 1

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H-%M-%S")
    print('saving output atlas image.')
    atlas_image.save('Output image_' + name_annotations + '_' + str(current_time) + '.png')


def export_types_csv(in_original_types, in_predicted_types):
    #
    num_errors = 0
    num_correct = 0
    num_not_present = 0
    num_extra_types = 0
    current_iteration = 0
    correct_indices = []
    for original, predicted in zip(in_original_types, in_predicted_types):
        orig_index_and_values = print_pokemon_types(original, False)
        pred_index_and_values = print_pokemon_types(predicted, False)

        #sorted_orig = sorted(orig_index_and_values.items(), key=operator.itemgetter(1))  # This is unnecessary
        #sorted_pred = sorted(pred_index_and_values.items(), key=operator.itemgetter(1))

        no_errors = True
        for i in orig_index_and_values:
            if i not in pred_index_and_values:
                # Then it has failed at least once.
                num_not_present += 1
                no_errors = False

        for i in pred_index_and_values:
            if i not in orig_index_and_values:
                num_extra_types += 1
                no_errors = False

        if no_errors:
            num_correct += 1
            print('pokemon with both correct types was: ')
            print(orig_index_and_values)
            print(pred_index_and_values)
            correct_indices.append(current_iteration)

        current_iteration += 1

    num_errors = num_not_present + num_extra_types
    print('The total number of errors was: ' + str(num_errors))
    print('Total number of elements with NO error in them: ' + str(num_correct))
    return correct_indices


def generate_all_one_type(in_num_elements, in_type="Fire", in_second_type="None"):
    new_types = np.zeros((in_num_elements, 2, 18), dtype=np.int)

    for elem in new_types:
        if type_to_categorical.count(in_type) > 0:  # This one is just for safety, should be valid, but one never knows
            index = type_to_categorical.index(in_type)
            elem[0][index] = 1  # Set to true the type specified in in_type
        if type_to_categorical.count(in_second_type) > 0:  # This one could be split into 2 different for cycles
            # To speed up the process when only one type is desired.
            index = type_to_categorical.index(in_second_type)
            elem[1][index] = 1  # Set to true the type specified in in_type

    return new_types
