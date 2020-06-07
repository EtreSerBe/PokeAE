import numpy as np
import math
from PIL import Image
from datetime import datetime
import tensorflow as tf
import tflearn
import h5py
import imgaug.augmenters as iaa
import matplotlib.colors

type_to_categorical = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying',
                       'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water']

# Params
image_dimension = 32
image_color_dimension = 3
original_dim = 3072  # 32 x 32 RGB images.
pokemon_types_dim = 18  # 18 *2, since we need space for the two possible types.
currently_using_two_hot_encoding = True

global glob_z_mean
global glob_z_std

# Number of filters in Autoencoder's order.
NUM_FILTERS_FIRST = 256
NUM_FILTERS_SECOND = 256
NUM_FILTERS_THIRD = NUM_FILTERS_SECOND
# Filter sizes
FILTER_SIZE_FIRST = 2  # Filter sizes 5 seem to perform better than 3 or 7, at least with 8-8 filters.
FILTER_SIZE_SECOND = 2
# Strides
FILTER_STRIDES_FIRST = 1
FILTER_STRIDES_SECOND = 1

DECODER_WIDTH = 8  # With the newly added Conv2d and maxPool layers added, it was reduced from 8 down to 4
EMBEDDED_VECTOR_SIZE = DECODER_WIDTH * DECODER_WIDTH
EMBEDDED_VECTOR_TOTAL = EMBEDDED_VECTOR_SIZE * image_color_dimension

# 432 + 18
# FULLY_CONNECTED_1_UNITS = 192  # 468 was great # 228  # with 256 instead of 512 it gets stuck at 0.07, not 0.03
# FULLY_CONNECTED_2_UNITS = 168
# FULLY_CONNECTED_3_UNITS = 128
latent_dimension = 256

# num_types_fully_connected = 64
EMBEDDED_ACTIVATION = 'linear'
LAST_ACTIVATION = 'sigmoid'
ALL_OTHER_ACTIVATIONS = 'leaky_relu'


def get_model_descriptive_name(in_optimizer, in_loss, in_version=''):
    now = datetime.now()
    current_time = now.strftime("%b_%d")  # %b is the code for Short mont version: Dec, Oct, etc.
    out_name = 'model_' + str(current_time) + '_optim_' + in_optimizer + \
               '_loss_' + str(in_loss) + '_last_activ_' + LAST_ACTIVATION + '_latent_' \
               + str(latent_dimension) + '_num_filters_' + str(NUM_FILTERS_FIRST) + '_' + \
               str(NUM_FILTERS_SECOND) \
               + '_decoder_width_' + str(DECODER_WIDTH) + \
               in_version + '.tflearn'
    return out_name


def get_network():
    # Building the encoder # The size of the input should be 3108 = 3072 + 18
    networkInput = tflearn.input_data(shape=[None, original_dim + pokemon_types_dim])
    # Once the data is in, we need to split the pixel data and the types data.
    map_flat = tf.slice(networkInput, [0, 0], [-1, original_dim])
    pokemonTypes = tf.slice(networkInput, [0, original_dim], [-1, -1])

    # We reshape the flat versions to something more like the original.
    mapShape = tf.reshape(map_flat, [-1, image_dimension, image_dimension, image_color_dimension])
    print("mapShape dimensions, before Conv_2D #1 are: " + str(mapShape))

    encoderStructure = tflearn.conv_2d(mapShape, NUM_FILTERS_FIRST, FILTER_SIZE_FIRST,
                                       strides=FILTER_STRIDES_FIRST, activation=ALL_OTHER_ACTIVATIONS)
    print("encoderStructure before max_pool_2D #1 is: " + str(encoderStructure))
    encoderStructure = tflearn.max_pool_2d(encoderStructure, 2, strides=2)

    print("encoderStructure before conv_2D #2 is: " + str(encoderStructure))
    encoderStructure = tflearn.conv_2d(encoderStructure, NUM_FILTERS_SECOND, FILTER_SIZE_SECOND,
                                       strides=FILTER_STRIDES_SECOND, activation=ALL_OTHER_ACTIVATIONS)
    print("encoderStructure before max_pool_2D #2 is: " + str(encoderStructure))
    encoderStructure = tflearn.max_pool_2d(encoderStructure, 2, strides=2)
    print("encoderStructure before flatten is: " + str(encoderStructure))
    """encoderStructure = tflearn.conv_2d(encoderStructure, NUM_FILTERS_THIRD, 2,
                                       strides=2, activation=ALL_OTHER_ACTIVATIONS)"""
    print("encoderStructure after WEIRD CONVOLUTION is: " + str(encoderStructure))  # Should be 4 by 4
    flatStructure = tflearn.flatten(encoderStructure)
    print("flatStructure is = " + str(flatStructure))
    # flatStructure = tflearn.fully_connected(flatStructure, EMBEDDED_VECTOR_TOTAL, activation=ALL_OTHER_ACTIVATIONS)
    # pre_type_dropout_rate = 1.0
    # flatStructure = tflearn.dropout(flatStructure, pre_type_dropout_rate)
    # flatStructure = tflearn.fully_connected(flatStructure, 64, activation=ALL_OTHER_ACTIVATIONS)
    encoder = tf.concat([flatStructure, pokemonTypes], 1)
    # encoder = tflearn.fully_connected(encoder, FULLY_CONNECTED_1_UNITS, activation=ALL_OTHER_ACTIVATIONS)
    """my_custom_layer_instance = PixelPlusTypesLayer(0)  # Param doesn't matter.
    encoder = my_custom_layer_instance(encoder)
    # encoder = tf.concat([encoder, pokemonTypes], 1)
    encoder = tflearn.activation(encoder, activation=ALL_OTHER_ACTIVATIONS)
    print("encoder after custom layer is = " + str(encoder))"""

    global glob_z_mean
    global glob_z_std
    glob_z_mean = tflearn.fully_connected(encoder, latent_dimension, activation=EMBEDDED_ACTIVATION)
    glob_z_std = tflearn.fully_connected(encoder, latent_dimension, activation=EMBEDDED_ACTIVATION)
    #

    # Sampler: Normal (gaussian) random distribution
    eps = tf.random_normal(tf.shape(glob_z_std), dtype=tf.float32, mean=0.0, stddev=1.0,
                           name='epsilon')  # + 0.00001
    z = glob_z_mean + tf.exp(glob_z_std / 2.0) * eps

    # decoder = tflearn.fully_connected(z, FULLY_CONNECTED_1_UNITS, activation=ALL_OTHER_ACTIVATIONS)
    # decoder = tflearn.batch_normalization(decoder)
    # decoder = tflearn.fully_connected(z, 64, activation=ALL_OTHER_ACTIVATIONS)

    decoder = tflearn.fully_connected(z, DECODER_WIDTH * DECODER_WIDTH * NUM_FILTERS_THIRD + pokemon_types_dim,
                                      activation=EMBEDDED_ACTIVATION, scope='decoder_fc_1')

    """decoder = tf.concat([decoder, pokemonTypes], 1)
    decoder_custom_layer_instance = PixelPlusTypesLayer(num_outputs=EMBEDDED_VECTOR_TOTAL)
    decoderStructure = decoder_custom_layer_instance(decoder)"""

    decoderStructure = tf.slice(decoder, [0, 0], [-1, DECODER_WIDTH * DECODER_WIDTH * NUM_FILTERS_THIRD], name='slice_1')
    decoderTypes = tf.slice(decoder, [0, DECODER_WIDTH * DECODER_WIDTH * NUM_FILTERS_THIRD], [-1, -1], name='slice_2')
    # decoderTypes = tflearn.activation(decoderTypes, activation='sigmoid')

    decoderStructure = tf.reshape(decoderStructure, [-1, DECODER_WIDTH, DECODER_WIDTH,
                                                     NUM_FILTERS_THIRD], name='reshape')

    # Decoder's convolution and up-sampling process.
    decoderStructure = tflearn.upsample_2d(decoderStructure, 2)
    decoderStructure = tflearn.conv_2d(decoderStructure, NUM_FILTERS_SECOND, FILTER_SIZE_SECOND,
                                       strides=1, activation=ALL_OTHER_ACTIVATIONS, scope='decoder_conv_1')

    decoderStructure = tflearn.upsample_2d(decoderStructure, 2)
    decoderStructure = tflearn.conv_2d(decoderStructure, NUM_FILTERS_FIRST, FILTER_SIZE_FIRST,
                                       strides=1, activation=ALL_OTHER_ACTIVATIONS, scope='decoder_conv_2')

    decoderStructure = tflearn.upsample_2d(decoderStructure, 2)
    # https://www.tensorflow.org/tutorials/generative/cvae, they use this last layer to return to the original
    decoderStructure = tflearn.conv_2d(decoderStructure, 3, 2, strides=1, activation=ALL_OTHER_ACTIVATIONS,
                                       scope='decoder_conv_3')
    decoderStructure = tflearn.max_pool_2d(decoderStructure, 2, strides=2)
    print("decoder structure size is*****: " + str(decoderStructure))
    # decoderStructure = tflearn.dropout(decoderStructure, 0.95)

    decoderStructure = tflearn.flatten(decoderStructure)
    # decoderTypes = tflearn.fully_connected(decoderTypes, pokemon_types_dim)
    # decoderTypes = tflearn.activation(decoderTypes, activation=ALL_OTHER_ACTIVATIONS)
    network = tf.concat([decoderStructure, decoderTypes], 1)
    # decoderTypes = tflearn.fully_connected(network, pokemon_types_dim, activation=ALL_OTHER_ACTIVATIONS)

    # network = tflearn.fully_connected(network, original_dim)  # , activation=ALL_OTHER_ACTIVATIONS)
    # network = tf.concat([decoderStructure, decoderTypes], 1)
    network = tflearn.flatten(network)
    network = tflearn.activation(network, activation=LAST_ACTIVATION)
    # network = tf.clip_by_value(network, -0.999, 0.9999)
    # network = tflearn.fully_connected(network, original_dim + pokemon_types_dim, activation=LAST_ACTIVATION)
    return network


# Define VAE Loss
def vae_loss(y_pred, y_true):
    # https://github.com/tflearn/tflearn/issues/72
    global glob_z_mean
    global glob_z_std
    glob_kld_weight = 1.0
    encode_decode_weight = 1.0
    # Reconstruction loss
    # But this is BINARY cross entropy, right?
    # https://peltarion.com/knowledge-center/documentation/
    # modeling-view/build-an-ai-model/loss-functions/binary-crossentropy
    encode_decode_loss = y_true * tf.math.log(1e-7 + y_pred) \
                         + (1 - y_true) * tf.math.log(1e-7 + 1 - y_pred)

    # encode_decode_loss_types = tf.slice(tf.abs(y_pred+y_true), [0, original_dim], [-1, -1])  # to check

    # Now only use the values ABOVE 1
    # encode_decode_loss_types = encode_decode_loss_types - 1
    # encode_decode_loss_types = tf.clip_by_value(encode_decode_loss_types, 0.0, 1.0)
    # Now it only should have the ones that were correct, right?
    encode_decode_loss_pixels = -tf.reduce_sum(encode_decode_loss, 1)
    encode_decode_loss_pixels *= encode_decode_weight
    # encode_decode_loss_types = -tf.reduce_sum(encode_decode_loss_types, 1)
    # Now it should tell the loss that this is good, which is why it's negative here.
    # encode_decode_loss_types *= 0.0  # It did nothing good
    # KL Divergence loss
    kl_div_loss = 1 + glob_z_std - tf.square(glob_z_mean) - tf.exp(glob_z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    out_kl_div_loss = tf.reduce_mean(encode_decode_loss_pixels + glob_kld_weight * kl_div_loss)
    return out_kl_div_loss  # + tf.reduce_mean(encode_decode_loss_types)


# Define VAE Loss
def vae_loss_mean_square(y_pred, y_true):
    # https://github.com/tflearn/tflearn/issues/72
    global glob_z_mean
    global glob_z_std
    kl_weight = 1.00
    pixels_weight = 1.0
    types_weight = 1.0
    # Reconstruction loss
    square_error = tf.square(y_pred - y_true)
    encode_decode_loss_pixels = tf.slice(square_error, [0, 0], [-1, original_dim])

    y_true_types = tf.slice(y_true, [0, original_dim], [-1, -1])
    y_pred_types = tf.slice(y_pred, [0, original_dim], [-1, -1])

    encode_decode_loss_types = y_true_types * tf.math.log(1e-7 + y_pred_types) \
                               + (1 - y_true_types) * tf.math.log(1e-7 + 1 - y_pred_types)

    encode_decode_loss_pixels = tf.reduce_mean(tf.reduce_sum(encode_decode_loss_pixels, 1))
    encode_decode_loss_types = -tf.reduce_mean(tf.reduce_sum(encode_decode_loss_types, 1))
    encode_decode_loss_pixels *= pixels_weight
    encode_decode_loss_types *= types_weight

    # final_encode_decode_loss = encode_decode_loss_pixels + encode_decode_loss_types

    # KL Divergence loss
    kl_div_loss = 1 + glob_z_std - tf.square(glob_z_mean) - tf.exp(glob_z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    out_kl_div_loss = tf.reduce_mean(kl_div_loss)
    return encode_decode_loss_pixels + encode_decode_loss_types + out_kl_div_loss * kl_weight


def vae_loss_abs_error(y_pred, y_true):
    # https://github.com/tflearn/tflearn/issues/72
    global glob_z_mean
    global glob_z_std
    # Reconstruction loss
    # But this is cross entropy, right? We can't use it right now
    encode_decode_loss = -tf.reduce_sum(tf.abs(y_pred - y_true), 1)
    # KL Divergence loss
    kl_div_loss = 1 + glob_z_std - tf.square(glob_z_mean) - tf.exp(glob_z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)


def get_generative_network(in_trained_model):
    input_noise = tflearn.input_data(shape=[None, latent_dimension], name='input_noise')
    decoder = tflearn.fully_connected(input_noise,
                                      DECODER_WIDTH * DECODER_WIDTH * NUM_FILTERS_THIRD + pokemon_types_dim,
                                      activation=EMBEDDED_ACTIVATION, scope='decoder_fc_1', reuse=True)

    decoderStructure = tf.slice(decoder, [0, 0], [-1, DECODER_WIDTH * DECODER_WIDTH * NUM_FILTERS_THIRD],
                                name='slice_1')
    decoderTypes = tf.slice(decoder, [0, DECODER_WIDTH * DECODER_WIDTH * NUM_FILTERS_THIRD], [-1, -1], name='slice_2')
    decoderStructure = tf.reshape(decoderStructure, [-1, DECODER_WIDTH, DECODER_WIDTH,
                                                     NUM_FILTERS_THIRD], name='reshape')

    # Decoder's convolution and up-sampling process.
    decoderStructure = tflearn.upsample_2d(decoderStructure, 2)
    decoderStructure = tflearn.conv_2d(decoderStructure, NUM_FILTERS_SECOND, FILTER_SIZE_SECOND,
                                       strides=1, activation=ALL_OTHER_ACTIVATIONS, scope='decoder_conv_1', reuse=True)

    decoderStructure = tflearn.upsample_2d(decoderStructure, 2)
    decoderStructure = tflearn.conv_2d(decoderStructure, NUM_FILTERS_FIRST, FILTER_SIZE_FIRST,
                                       strides=1, activation=ALL_OTHER_ACTIVATIONS, scope='decoder_conv_2', reuse=True)

    decoderStructure = tflearn.upsample_2d(decoderStructure, 2)
    # https://www.tensorflow.org/tutorials/generative/cvae, they use this last layer to return to the original
    decoderStructure = tflearn.conv_2d(decoderStructure, 3, 2, strides=1, activation=ALL_OTHER_ACTIVATIONS,
                                       scope='decoder_conv_3', reuse=True)
    decoderStructure = tflearn.max_pool_2d(decoderStructure, 2, strides=2)

    decoderStructure = tflearn.flatten(decoderStructure)
    network = tf.concat([decoderStructure, decoderTypes], 1)
    network = tflearn.activation(network, activation=LAST_ACTIVATION)
    generator_model = tflearn.DNN(network, session=in_trained_model.session)
    return generator_model


class PixelPlusTypesLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(PixelPlusTypesLayer, self).__init__()
        self.num_outputs = num_outputs

    # input shape should be x+18 poke-type information, that's why it's -18.
    def build(self, input_shape):
        print('input_shape in build function size is: ' + str(input_shape))
        # custom_weights = {'W': tf.Variable(tf.random_normal([input_shape[-1], 1 + 18]))}
        self.custom_weight = {'W_custom': tf.Variable(tf.random_normal(shape=[1, input_shape[-1] - 18, 19]),
                                                      trainable=True, dtype=np.float)}
        self.bias = {'b': tf.Variable(tf.random_normal([input_shape[-1] - 18]), trainable=True, dtype=np.float)}

    def call(self, in_input, **kwargs):
        image_information_size = int(in_input.shape[-1]) - pokemon_types_dim
        print('image_information_size in call function size is: ' + str(image_information_size))
        print('in_input in call function size is: ' + str(in_input))
        print('self.custom_weight size is: ' + str(self.custom_weight))
        print('self.bias size is: ' + str(self.bias))
        input_1 = tf.slice(in_input, begin=[0, 0], size=[-1, image_information_size])
        print('input_1 size is: ' + str(input_1))
        input_2 = tf.slice(in_input, begin=[0, image_information_size], size=[-1, -1])
        print('input_2 size is: ' + str(input_2))
        # Input_1 should be the 3072 values from the image, #2 the 18 values representing the poke-types
        # We must now have a [3072, 19] matrix, were the row is the image value to which it will output,
        # element 0 is the value it had and elements 1 to 18 are the types

        input_1_transposed = input_1
        input_1_transposed = tf.reshape(input_1_transposed, shape=[-1, image_information_size, 1])
        print('input_1_transposed size is: ' + str(input_1_transposed))
        input_2_repeat = tf.repeat(input_2, repeats=input_1.shape[-1], axis=0)
        input_2_repeat = tf.reshape(input_2_repeat, shape=[-1, image_information_size, pokemon_types_dim])
        print('input_2_repeat size is: ' + str(input_2_repeat))
        full_matrix = tf.concat([input_1_transposed, input_2_repeat], axis=2)  # this should be 3072 by 19 now
        print('full_matrix size is: ' + str(full_matrix))
        # My output must be 3072, 1 in size, so input must be [3072, x] * [y, 1], right? it doesn't sound right
        # https://stackoverflow.com/questions/40670370/dot-product-of-two-vectors-in-tensorflow
        output = tf.reduce_sum(tf.multiply(full_matrix, self.custom_weight['W_custom']), axis=2)
        print('output after Keras.DOT size is: ' + str(output))
        output = tf.reshape(output, shape=[-1, in_input.shape[-1] - pokemon_types_dim]) + self.bias['b']
        print('output after reshape is: ' + str(output))
        # output = tf.concat([output, input_2], axis=1)
        # print('output after concat size is: ' + str(output))
        return output


def predict_batches(in_complete_set_to_predict, in_trained_model, in_number_of_chunks=10):
    out_encode_decode_sample = []
    num_chunks = in_number_of_chunks
    chunk_size = int(len(in_complete_set_to_predict) / num_chunks)
    print('number of samples is: ' + str(len(in_complete_set_to_predict)) + ' chunk size is: ' + str(chunk_size))
    for i in range(0, num_chunks - 1):
        current_slice = in_complete_set_to_predict[chunk_size * i:chunk_size * (i + 1)]
        out_encode_decode_sample.extend(in_trained_model.predict(current_slice))

    current_slice = in_complete_set_to_predict[chunk_size * num_chunks:]
    print('number of samples in last slice is: ' + str(len(current_slice)))
    out_encode_decode_sample.extend(in_trained_model.predict(current_slice))
    return out_encode_decode_sample


def prepare_dataset_for_input_layer(in_h5f_dataset_name, in_dataset_x_label="pokedataset32_X",
                                    in_dataset_y_label="pokedataset32_Y",
                                    in_use_two_hot_encoding=True):
    h5f_obj = h5py.File(in_h5f_dataset_name, 'r')
    h5f_x_values = h5f_obj[in_dataset_x_label]
    h5f_y_values = h5f_obj[in_dataset_y_label]
    global pokemon_types_dim  # This is used to configure for the first time it runs.
    global currently_using_two_hot_encoding
    if currently_using_two_hot_encoding != in_use_two_hot_encoding:
        print('WARNING: Difference between type_hot_encodings detected.')
        currently_using_two_hot_encoding = in_use_two_hot_encoding
    if in_use_two_hot_encoding:
        pokemon_types_dim = len(type_to_categorical)
        h5f_y_values = np.reshape(np.asarray(h5f_y_values), newshape=[h5f_y_values.shape[0], 18])
    else:
        pokemon_types_dim = len(type_to_categorical) * 2
        h5f_y_values = np.reshape(np.asarray(h5f_y_values), newshape=[h5f_y_values.shape[0], 18 * 2])
    # We return them ready to be appended like this: "expanded_X = np.append(X, Y, axis=1)"
    return h5f_x_values, h5f_y_values


def print_pokemon_types(types, in_print_all=True):
    types_as_strings = []
    flat_types = np.asarray(types).flatten()
    if not currently_using_two_hot_encoding:  # Only do this if it was with 2 18-size vector for the types.
        flat_types = np.reshape(flat_types, newshape=[2, 18])
    else:
        flat_types = np.reshape(flat_types, newshape=[1, 18])
    index_and_value = {}
    for typearray in flat_types:
        for i in range(0, pokemon_types_dim):
            if typearray[i] >= 0.15:
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
        if not currently_using_two_hot_encoding:
            reshaped_types = np.reshape(in_encode_decode_sample[i][3072:3072 + pokemon_types_dim], [2, 18])
        else:
            reshaped_types = np.reshape(in_encode_decode_sample[i][3072:3072 + pokemon_types_dim], [1, 18])
        out_reconstructed_types.append(reshaped_types)
    return out_reconstructed_pixels, out_reconstructed_types


def export_as_atlas(in_image_list, in_reconstructed_image_list, image_width=32, image_height=32, num_channels=3,
                    name_annotations='standard', name_prefix=''):
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
    current_time = now.strftime("%Y-%m-%d %H-%M")
    print('saving output atlas image in the ImageOutputs folder.')
    atlas_image.save('ImageOutputs/' + name_prefix + 'Image_'
                     + '_' + str(current_time) + name_annotations + '.png')


def export_multi_type_atlas(in_image_list, in_reconstructed_image_list, in_forced_image_list,
                            image_width=32, image_height=32,
                            num_channels=3, name_annotations='', name_prefix=''):
    num_elements = len(in_image_list)
    if num_elements == 0:
        return
    rows = math.ceil(math.sqrt(num_elements))  # ceil to be the highest integer enough.
    row_counter = 0
    column_counter = 0
    # Make it big enough to put the original above the reconstructed. (That's why multiplied by 2)
    atlas_image = Image.new('RGB', (image_width * rows, image_height * rows * 3), (0, 0, 0))

    for original, reconstructed, forced in zip(in_image_list, in_reconstructed_image_list, in_forced_image_list):
        """if column_counter + (row_counter*rows) >= num_elements:
            break  # This is to stop it as soon as"""

        reshaped_image = np.reshape(
            np.uint8(np.multiply(original.flatten(), 255.)),
            [image_width, image_height, num_channels])

        reshaped_reconstructed = np.reshape(
            np.uint8(np.multiply(reconstructed.flatten(), 255.)),
            [image_width, image_height, num_channels])

        reshaped_forced = np.reshape(
            np.uint8(np.multiply(forced.flatten(), 255.)),
            [image_width, image_height, num_channels])
        # reshaped_reconstructed = np.asarray(reshaped_reconstructed)

        offset = (column_counter * image_width, row_counter * image_height * 3)
        im_original = Image.fromarray(reshaped_image, 'RGB')
        atlas_image.paste(im_original, offset)

        offset = (column_counter * image_width, row_counter * image_height * 3 + image_height)
        im_reconstructed = Image.fromarray(reshaped_reconstructed, 'RGB')
        atlas_image.paste(im_reconstructed, offset)

        offset = (column_counter * image_width, row_counter * image_height * 3 + image_height * 2)
        im_forced = Image.fromarray(reshaped_forced, 'RGB')
        atlas_image.paste(im_forced, offset)
        column_counter += 1
        # Go to the next row.
        if column_counter == rows:
            column_counter = 0
            row_counter += 1

    now = datetime.now()
    current_time = now.strftime("%Y-%b-%d %H-%M")
    print('saving output Multi type atlas image in the ImageOutputs folder.')
    atlas_image.save('ImageOutputs/' + name_prefix + 'Image_'
                     + name_annotations + '_' + str(current_time) + '.png')


def read_types_from_csv(in_csv_reader_dictionary, in_use_two_hot_encoding=True):
    out_type_labels = []
    if in_use_two_hot_encoding:
        for row in in_csv_reader_dictionary:
            # print(row)
            # Make a variable with the 18 spaces in 0.
            one_hot_type = [0] * 18
            type_string = str(row['Type1'])
            first_type = type_to_categorical.index(type_string)
            one_hot_type[first_type] = 1  # Set it to one, as it possesses this type.
            # Check if it has a second type:
            if row['Type2'] != '':  # if it does, then add it.
                type_string = str(row['Type2'])
                second_type = type_to_categorical.index(type_string)
                one_hot_type[second_type] = 1  # Set it to one, as it ALSO possesses this type.
            else:
                one_hot_type[first_type] = 2  # Set it to TWO, as it only possesses one type.
            out_type_labels.append(one_hot_type)
    else:
        for row in in_csv_reader_dictionary:
            # print(row)
            # Make a variable with the 18 spaces in 0.
            one_hot_type = [0] * 18
            one_hot_type_2 = [0] * 18
            # We only want the 2nd and 3rd (if any) columns
            type_string = str(row['Type1'])
            first_type = type_to_categorical.index(type_string)
            one_hot_type[first_type] = 1  # Set it to one, as it possesses this type.
            # Check if it has a second type:
            if row['Type2'] != '':  # if it does, then add it.
                type_string = str(row['Type2'])
                second_type = type_to_categorical.index(type_string)
                one_hot_type_2[second_type] = 1  # Set it to one, as it ALSO possesses this type.
                out_type_labels.append([one_hot_type, one_hot_type_2])
            else:
                out_type_labels.append([one_hot_type, one_hot_type])  # Add the same type TWICE.
    # In any case, return the list with the labels.
    return out_type_labels


def export_types_csv(in_original_types, in_predicted_types):
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
            # print('pokemon with both correct types was: ')
            # print(orig_index_and_values)
            # print(pred_index_and_values)
            correct_indices.append(current_iteration)

        current_iteration += 1

    num_errors = num_not_present + num_extra_types
    print('The total number of errors was: ' + str(num_errors) + ' from which ExtraTypes were: --- ' +
          str(num_extra_types) + ' and Missing original types were: --- ' + str(num_not_present))
    print('Total number of elements with NO error in them: ' + str(num_correct))
    return correct_indices


def generate_all_one_type(in_num_elements, in_type="Fire", in_second_type="None"):
    if currently_using_two_hot_encoding:
        new_types = np.zeros((in_num_elements, 1, len(type_to_categorical)), dtype=np.int)
    else:
        new_types = np.zeros((in_num_elements, 2, len(type_to_categorical)), dtype=np.int)

    for elem in new_types:
        if type_to_categorical.count(in_type) > 0:  # This one is just for safety, should be valid, but one never knows
            index = type_to_categorical.index(in_type)
            elem[0][index] = 1  # Set to true the type specified in in_type
            if type_to_categorical.count(in_second_type) > 0:  # This one could be split into 2 different for cycles
                # To speed up the process when only one type is desired.
                index = type_to_categorical.index(in_second_type)
                if not currently_using_two_hot_encoding:
                    elem[1][index] = 1  # Set to true the type specified in in_type
                else:
                    elem[0][index] = 1
            else:
                if not currently_using_two_hot_encoding:
                    elem[1][index] = 1  # If mono-type, repeat it in the second one. P.E: Pikachu is electric electric.
                else:
                    elem[0][index] += 1  # Now it should be set to 2 for mono-types. P.E: Pikachu is electric electric
                    print('Monotype Pokemon has two instead of 1 in the encoding: ' + str(elem[0][index]))

    return new_types


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


def zero_center(in_image_list):
    overall_mean = 0.0
    output_images = []
    for image in in_image_list:
        overall_mean += np.mean(image)
    overall_mean /= len(in_image_list)
    for image in in_image_list:
        image = image - overall_mean
        output_images.append(image)
    return output_images
