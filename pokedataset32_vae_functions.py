import numpy as np
import math
from PIL import Image
from datetime import datetime
import tensorflow as tf
import tflearn
import h5py
import imgaug as ia
import imgaug.augmenters as iaa
import operator
import matplotlib.colors

type_to_categorical = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying',
                       'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water']

# Params
image_dimension = 32
image_color_dimension = 3
original_dim = 3072  # 32 x 32 RGB images.
pokemon_types_dim = 18 * 2  # 18 *2, since we need space for the two possible types.

glob_z_mean = 0
glob_z_std = 0


def get_network():
    # Number of filters in Autoencoder's order.
    NUM_FILTERS_FIRST = 64
    NUM_FILTERS_SECOND = 64
    NUM_FILTERS_THIRD = 64
    # Filter sizes
    FILTER_SIZE_FIRST = 3  # Filter sizes 5 seem to perform better than 3 or 7, at least with 8-8 filters.
    FILTER_SIZE_SECOND = 3
    FILTER_SIZE_THIRD = 1
    # Strides
    FILTER_STRIDES_FIRST = 1
    FILTER_STRIDES_SECOND = 1
    FILTER_STRIDES_THIRD = 1

    FULLY_CONNECTED_1_UNITS = 256  # with 256 instead of 512 it gets stuck at 0.07, not 0.03
    FULLY_CONNECTED_2_UNITS = 128
    # FULLY_CONNECTED_3_UNITS = 64
    embedded_representation_units = FULLY_CONNECTED_2_UNITS

    DECODER_WIDTH = 8  # With the newly added Conv2d and maxPool layers added, it was reduced from 8 down to 4
    EMBEDDED_VECTOR_SIZE = DECODER_WIDTH * DECODER_WIDTH
    EMBEDDED_VECTOR_TOTAL = EMBEDDED_VECTOR_SIZE * image_color_dimension

    # Building the encoder # The size of the input should be 3108 = 3072 + 18*2
    # data_augmentation=image_aug, omitted cause TFLearn's augmentation can't work well for our input.
    networkInput = tflearn.input_data(shape=[None, original_dim + pokemon_types_dim])
    # Once the data is in, we need to split the pixel data and the types data.
    map_flat = tf.slice(networkInput, [0, 0], [-1, original_dim])
    pokemonTypesFlat = tf.slice(networkInput, [0, original_dim], [-1, -1])

    # We reshape the flat versions to something more like the original.
    mapShape = tf.reshape(map_flat, [-1, image_dimension, image_dimension, image_color_dimension])
    print("mapShape dimensions, before Conv_2D #1 are: " + str(mapShape))
    pokemonTypes = tf.reshape(pokemonTypesFlat, [-1, pokemon_types_dim])

    encoderStructure = tflearn.conv_2d(mapShape, NUM_FILTERS_FIRST, FILTER_SIZE_FIRST,
                                       strides=FILTER_STRIDES_FIRST, activation='relu')
    print("encoderStructure before dropout is: " + str(encoderStructure))
    # encoderStructure = tflearn.dropout(encoderStructure, 0.5)  # Re-add it later with a lower value like: 0.85 0.9
    print("encoderStructure before max_pool_2D #1 is: " + str(encoderStructure))
    encoderStructure = tflearn.max_pool_2d(encoderStructure, 2, strides=2)
    print("encoderStructure before conv_2D #2 is: " + str(encoderStructure))
    encoderStructure = tflearn.conv_2d(encoderStructure, NUM_FILTERS_SECOND, FILTER_SIZE_SECOND,
                                       strides=FILTER_STRIDES_SECOND, activation='relu')
    print("encoderStructure before max_pool_2D #2 is: " + str(encoderStructure))
    encoderStructure = tflearn.max_pool_2d(encoderStructure, 2, strides=2)
    print("encoderStructure before flatten is: " + str(encoderStructure))

    flatStructure = tflearn.flatten(encoderStructure)
    print("flatStructure is = " + str(flatStructure))
    flatStructureSize = flatStructure.shape[1]  # Why is it size 2048 with 8 filters and 1024 with 4?
    print('flatStructureSize = ' + str(flatStructureSize))

    encoder = tf.concat([flatStructure, pokemonTypes], 1)

    encoder = tflearn.fully_connected(encoder, FULLY_CONNECTED_1_UNITS, activation='relu')

    encoder = tflearn.fully_connected(encoder, FULLY_CONNECTED_2_UNITS, activation='relu')

    # embedded representation? Yes.
    # encoder = tflearn.fully_connected(encoder, FULLY_CONNECTED_3_UNITS, activation='relu')

    global glob_z_mean
    global glob_z_std
    glob_z_mean = tflearn.fully_connected(encoder, embedded_representation_units)
    glob_z_std = tflearn.fully_connected(encoder, embedded_representation_units)

    # decoder = tflearn.fully_connected(encoder, FULLY_CONNECTED_2_UNITS, activation='relu')

    decoder = tflearn.fully_connected(encoder, FULLY_CONNECTED_1_UNITS, activation='relu')

    decoder = tflearn.fully_connected(decoder, int(EMBEDDED_VECTOR_TOTAL + pokemon_types_dim), activation='relu')

    decoderStructure = tf.slice(decoder, [0, 0], [-1, EMBEDDED_VECTOR_TOTAL])
    decoderTypes = tf.slice(decoder, [0, EMBEDDED_VECTOR_TOTAL], [-1, -1])
    print("decoder types size is*****: " + str(decoderTypes))

    decoderStructure = tf.reshape(decoderStructure, [-1, DECODER_WIDTH, DECODER_WIDTH,
                                                     image_color_dimension])

    # Decoder's convolution and up-sampling process.
    decoderStructure = tflearn.conv_2d(decoderStructure, NUM_FILTERS_SECOND, FILTER_SIZE_SECOND,
                                       strides=FILTER_STRIDES_SECOND, activation='relu')
    decoderStructure = tflearn.upsample_2d(decoderStructure, 2)

    decoderStructure = tflearn.conv_2d(decoderStructure, NUM_FILTERS_FIRST, FILTER_SIZE_FIRST,
                                       strides=FILTER_STRIDES_FIRST, activation='relu')
    decoderStructure = tflearn.upsample_2d(decoderStructure, 2)

    decoderStructure = tflearn.flatten(decoderStructure)  # With 64 filters, it has 3108*64 = 198,912 connections...

    network = tf.concat([decoderStructure, decoderTypes], 1)

    # Added this layer since maybe it was going from 32,768 to 3,108 units too fast. results were mixed.
    # network = tflearn.fully_connected(network, 8192, activation='relu')

    print("network before the final fully_connected is: " + str(network))
    network = tflearn.fully_connected(network, original_dim + pokemon_types_dim, activation='relu')
    return network


# Define VAE Loss
def vae_loss(y_pred, y_true):
    # https: // github.com / tflearn / tflearn / issues / 72
    global glob_z_mean
    global glob_z_std
    # Reconstruction loss
    encode_decode_loss = y_true * tf.math.log(1e-10 + y_pred) + (1 - y_true) * tf.math.log(1e-10 + 1 - y_pred)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
    # KL Divergence loss
    kl_div_loss = 1 + glob_z_std - tf.square(glob_z_mean) - tf.exp(glob_z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)


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


def image_augmentation(in_image_list, in_types_list, in_flip_lr=True, in_gamma_contrast=True,
                       in_multiply_saturation=True, in_multiply_brightness=True,
                       in_multiply_hue=True, in_gaussian_blur=True):
    out_all_images = []
    out_all_images.extend(in_image_list)
    out_all_types = []
    out_all_types.extend(in_types_list)

    if in_flip_lr:
        images_aug = iaa.Fliplr(1.0)(images=in_image_list)
        out_all_images.extend(images_aug)
    if in_gamma_contrast:
        images_aug = iaa.GammaContrast((0.75, 1.25))(images=in_image_list)
        out_all_images.extend(images_aug)
    if in_multiply_saturation:
        images_aug = iaa.MultiplySaturation((0.5, 1.5), from_colorspace='RGB')(images=in_image_list)
        out_all_images.extend(images_aug)
    if in_multiply_brightness:
        images_aug = iaa.MultiplyBrightness((0.5, 1.5))(images=in_image_list)
        out_all_images.extend(images_aug)
    if in_multiply_hue:
        images_aug = iaa.MultiplyHue((0.8, 1.2))(images=in_image_list)
        out_all_images.extend(images_aug)
    if in_gaussian_blur:
        images_aug = iaa.GaussianBlur(0.75)(images=in_image_list)
        out_all_images.extend(images_aug)

    print("The size of Out_all_images is: " + str(len(out_all_images)))
    for i in range(0, int(len(out_all_images) / len(in_types_list)) - 1):
        out_all_types.extend(in_types_list)

    print("The size of Out_all_types is: " + str(len(out_all_types)))

    print("total number of images after augmentation: " + str(len(out_all_images)))
    return out_all_images, out_all_types


def convert_to_format(in_image_list, in_format_string):
    out_image_list = []
    if 'RGB' == in_format_string:
        for current_image in in_image_list:
            current_image = np.asarray(current_image).flatten()
            current_image = np.true_divide(current_image, 255.)
            out_image_list.append(current_image)
    elif 'HSV' == in_format_string:
        for current_image in in_image_list:
            # NOTE: The division by 255 must be performed BEFORE the conversion to HSV. Corrected.
            current_image = np.true_divide(current_image, 255.)
            current_image = matplotlib.colors.rgb_to_hsv(current_image)
            current_image = np.asarray(current_image).flatten()
            out_image_list.append(current_image)
    elif 'HSV_TO_RGB' == in_format_string:
        for current_image in in_image_list:
            current_image = np.reshape(current_image, [32, 32, 3])
            current_image = matplotlib.colors.hsv_to_rgb(current_image)
            current_image = np.asarray(current_image).flatten()
            # current_image = np.multiply(current_image, 255.)
            out_image_list.append(current_image)
    else:
        print("Error in convert to format: Non valid in_format_string received.")
    return out_image_list  # Check that the changes to its content remain after return.


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


def reconstruct_pixels_and_types(in_encode_decode_sample):
    out_reconstructed_pixels = []
    out_reconstructed_types = []
    for i in range(0, len(in_encode_decode_sample)):
        sample = in_encode_decode_sample[i][0:3072]
        reshaped_sample = np.reshape(sample, [32, 32, 3])
        # https://matplotlib.org/api/_as_gen/matplotlib.colors.hsv_to_rgb.html#matplotlib.colors.hsv_to_rgb
        reshaped_sample = matplotlib.colors.hsv_to_rgb(reshaped_sample)
        pixel_list = reshaped_sample.flatten()
        out_reconstructed_pixels.append(pixel_list)
        reshaped_types = np.reshape(in_encode_decode_sample[i][3072:3108], [2, 18])
        out_reconstructed_types.append(reshaped_types)
    return out_reconstructed_pixels, out_reconstructed_types


def export_as_atlas(in_image_list, in_reconstructed_image_list, image_width=32, image_height=32, num_channels=3,
                    name_annotations='standard'):
    num_elements = len(in_image_list)
    if num_elements == 0:
        return
    print('Number or elements in in_image_list is: ' + str(num_elements))
    rows = math.ceil(math.sqrt(num_elements))  # ceil to be the highest integer enough.
    print('number of Rows and Columns to have is: ' + str(rows))
    row_counter = 0
    column_counter = 0
    # Make it big enough to put the original above the reconstructed. (That's why multiplied by 2)
    atlas_image = Image.new('RGB', (image_width * rows, image_height * rows * 2), (0, 0, 0))

    for original, reconstructed in zip(in_image_list, in_reconstructed_image_list):
        """if column_counter + (row_counter*rows) >= num_elements:
            break  # This is to stop it as soon as"""

        reshaped_image = np.reshape(
            np.uint8(np.multiply(original.flatten(), 255.)),
            [image_width, image_height, num_channels])

        reshaped_reconstructed = np.reshape(
            np.uint8(np.multiply(reconstructed.flatten(), 255.)),
            [image_width, image_height, num_channels])
        # reshaped_reconstructed = np.asarray(reshaped_reconstructed)

        offset = (column_counter * image_width, row_counter * image_height * 2)
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
    print('saving output atlas image in the ImageOutputs folder.')
    atlas_image.save('ImageOutputs/Output image_' + name_annotations + '_' + str(current_time) + '.png')


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

        # sorted_orig = sorted(orig_index_and_values.items(), key=operator.itemgetter(1))  # This is unnecessary
        # sorted_pred = sorted(pred_index_and_values.items(), key=operator.itemgetter(1))

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
