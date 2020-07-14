"""The Dataset is stored in a csv file, so we can use TFLearn load_csv() function to load the data from
file into a python list. We specify 'target_column' argument to indicate that our labels (survived or not)
 are located in the first column (id: 0). The function will return a tuple: (data, labels)."""

import os
import csv
import h5py
import glob
import numpy as np
import pokedataset32_vae_functions as utilities
from PIL import Image
from imgaug import augmenters as iaa

# full RGB; full HSV, and train HSV augmented are the indispensable ones.
# Important parameters for the data set creation. Modifying them will change the final set generated.
image_format_to_use = "HSV"
full_dataset = False
use_augmentation = True  # NOTE: Right now, if augmentation is used, the full dataset option should be disabled.
use_type_swap = False
use_two_hot_encoding = True
num_noise_per_pokemon = 2

# This is only used for the Pokemon images with swapped types, which are a special dataset for testing.
if not use_type_swap:
    source_folder = 'poke3232dataset/'
    csv_type_file = 'pokemontypes_2020.csv'
else:
    source_folder = 'poke3232dataset_type_swapped/'
    csv_type_file = 'pokemontypes_2020_type_swapped.csv'


# As seen in https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def sortKeyFunction(s):
    print(os.path.basename(s))
    pokenumber = os.path.basename(s)[:-7]
    if "-" in pokenumber:
        pokenumber = pokenumber.split("-")[0]+".5"

    print(pokenumber)
    s = pokenumber
    return float(s)  # :-4 is to not return the extension


# Load CSV file, indicate that the first column represents labels
# Now, we can check for the
my_file = open(csv_type_file)
csv_reader_dict = csv.DictReader(my_file)
# csv_reader_object = csv.reader(my_file)

encoded_type_labels = []

type_to_categorical = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying',
                       'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water']

encoded_type_labels = utilities.read_types_from_csv(csv_reader_dict, use_two_hot_encoding)

print('Finished types encoding')
print(encoded_type_labels)
encoded_type_labels = np.asarray(encoded_type_labels)

# Now, encoded_type_labels has all the one_hot encodings for all the elements.
# We just have to put it into the same file as the pixel data and that's it.
pixel_data = []
image_list = []
white_background_image_list = []
noise_background_image_list = []
for i in range(0, num_noise_per_pokemon):
    noise_background_image_list.append(list())

# This was important, if not re-sorted, it used a non-numeric order, which was not desired in this case.
filename_list = glob.glob(source_folder+'*.png')
filename_list.sort(key=sortKeyFunction)

# Now, the images need to be augmented BEFORE converting it to the HSV color space.
im_background = Image.new('RGBA', (32, 32), (255, 255, 255, 0))

for filename in filename_list:
    with Image.open(filename) as image:
        # Note, it is always converted to RBG, to ignore the Alpha channel
        # and because augmentation library (aleju/imgaug) works on RGB color space.
        im = image.convert('RGBA')

        pixel_matrix_wb = utilities.blend_alpha_images(im_background, im)
        white_background_image_list.append(pixel_matrix_wb)

        for i in range(0, num_noise_per_pokemon):
            im_noise = utilities.generate_random_noise(32, 32, 3)
            pixel_matrix_noise = utilities.blend_alpha_images(im_noise, im)
            noise_background_image_list[i].append(pixel_matrix_noise)

        im = image.convert('RGB')
        pixel_matrix = np.asarray(im.getdata())  # Make the Width*Height*Depth matrices
        pixel_matrix = np.reshape(pixel_matrix, newshape=[32, 32, 3])
        pixel_matrix = pixel_matrix.astype(dtype=np.uint8)
        image_list.append(pixel_matrix)  # Need them all stored in one container for augmentation.


if use_augmentation:
    print("total images before augmentation is: " + str(len(image_list)))
    if not full_dataset:
        training_elements = int((len(image_list) / 100) * 85)  # This will give us 15% for testing
        num_test_elements = len(image_list) - training_elements
        test_images_list = image_list[training_elements:]  # First assign test ones, to avoid losing info.
        test_images_list.extend(white_background_image_list[training_elements:])
        test_labels_list = encoded_type_labels[training_elements:]
        test_labels_list = np.concatenate((test_labels_list, test_labels_list), axis=0)

        image_list = image_list[0:training_elements]
        encoded_type_labels = encoded_type_labels[0:training_elements]
        image_list.extend(white_background_image_list[0:training_elements])
        # We need it to be twice given black + white backgrounds.
        encoded_type_labels = np.concatenate((encoded_type_labels, encoded_type_labels), axis=0)
        if num_noise_per_pokemon > 0:
            test_noisy_images_list = noise_background_image_list[0][training_elements:]
            test_noisy_labels_list = test_labels_list[0:num_test_elements]
            for i in range(1, num_noise_per_pokemon):
                test_noisy_images_list.extend(noise_background_image_list[i][training_elements:])
                # One per noisy poke lap.
                test_noisy_labels_list = np.concatenate((test_noisy_labels_list,
                                                         test_noisy_labels_list[0:num_test_elements]), axis=0)
            train_noisy_images_list = noise_background_image_list[0][0:training_elements]
            train_noisy_labels_list = encoded_type_labels[0:training_elements]
            for i in range(1, num_noise_per_pokemon):
                train_noisy_images_list.extend(noise_background_image_list[i][0:training_elements])
                train_noisy_labels_list = np.concatenate((train_noisy_labels_list,
                                                          train_noisy_labels_list[0:training_elements]), axis=0)

            test_noisy_images_data, test_noisy_labels_data = utilities.image_augmentation(test_noisy_images_list,
                                                                                          test_noisy_labels_list)
            train_noisy_images_data, train_noisy_labels_data = utilities.image_augmentation(train_noisy_images_list,
                                                                                            train_noisy_labels_list)
            test_pixel_noisy_data = np.asarray(test_noisy_images_data).astype(dtype=np.dtype('Float32'))
            train_pixel_noisy_data = np.asarray(train_noisy_images_data).astype(dtype=np.dtype('Float32'))
            test_pixel_noisy_data = utilities.convert_to_format(test_pixel_noisy_data, image_format_to_use)
            train_pixel_noisy_data = utilities.convert_to_format(train_pixel_noisy_data, image_format_to_use)

        print("getting test data augmented.")
        test_pixel_data, test_label_data = utilities.image_augmentation(test_images_list,
                                                                        test_labels_list, in_flip_lr=True)
        test_pixel_data = np.asarray(test_pixel_data).astype(dtype=np.dtype('Float32'))  # Float64 by default.
        test_pixel_data = utilities.convert_to_format(test_pixel_data, image_format_to_use)

    print("getting non-test data augmented.")
    # Now, do the augmentation for the non-test images. If no split was specified, this will contain all images.
    pixel_data, label_data = utilities.image_augmentation(image_list, encoded_type_labels, in_flip_lr=True)
    print("data augmentation successful.")
else:  # If no augmentation will be applied.
    image_list.extend(white_background_image_list)
    encoded_type_labels = np.concatenate((encoded_type_labels, encoded_type_labels), axis=0)
    pixel_data = image_list  # Only pass the variables to the correct names.
    label_data = encoded_type_labels

pixel_data = np.asarray(pixel_data).astype(dtype=np.dtype('Float32'))  # Float64 by default.

# Now, we need to put them in the correct format according to the desired set.
pixel_data = utilities.convert_to_format(pixel_data, image_format_to_use)

# unison_shuffled_copies(pixel_data, encoded_type_labels)  # Shuffled along the first axis only.

# Finally, put it into a h5f dataset and that's it.
if full_dataset:
    h5f = h5py.File('pokedataset32_full_' + image_format_to_use +
                    ('_Two_Hot_Encoded' if use_two_hot_encoding else '') +
                    ('_Augmented' if use_augmentation else '') +
                    ('_Type_Swapped' if use_type_swap else '') + '.h5', 'w')
    # These two lines below are used when the full data set is to be in one file.
    h5f.create_dataset('pokedataset32_X', data=pixel_data)
    h5f.create_dataset('pokedataset32_Y', data=label_data)
    h5f.close()
else:  # If it has train and test separation.
    h5f = h5py.File('pokedataset32_train_' + image_format_to_use +
                    ('_Two_Hot_Encoded' if use_two_hot_encoding else '') +
                    ('_Augmented' if use_augmentation else '') +
                    ('_Type_Swapped' if use_type_swap else '') + '.h5', 'w')
    # These four lines below are for the data split into train and test portions.
    h5f.create_dataset('pokedataset32_X', data=pixel_data)
    h5f.create_dataset('pokedataset32_Y', data=label_data)
    h5f.create_dataset('pokedataset32_X_test', data=test_pixel_data)
    h5f.create_dataset('pokedataset32_Y_test', data=test_label_data)
    h5f.close()
    if num_noise_per_pokemon > 0:
        h5f_NOISE = h5py.File('pokedataset32_train_NOISE_' + image_format_to_use +
                              ('_Two_Hot_Encoded' if use_two_hot_encoding else '') +
                              ('_Augmented' if use_augmentation else '') +
                              ('_Type_Swapped' if use_type_swap else '') + '.h5', 'w')
        # These four lines below are for the data split into train and test portions.
        h5f_NOISE.create_dataset('pokedataset32_X', data=train_pixel_noisy_data)
        h5f_NOISE.create_dataset('pokedataset32_Y', data=train_noisy_labels_data)
        h5f_NOISE.create_dataset('pokedataset32_X_test', data=test_pixel_noisy_data)
        h5f_NOISE.create_dataset('pokedataset32_Y_test', data=test_noisy_labels_data)
        h5f_NOISE.close()
