"""The Dataset is stored in a csv file, so we can use TFLearn load_csv() function to load the data from
file into a python list. We specify 'target_column' argument to indicate that our labels (survived or not)
 are located in the first column (id: 0). The function will return a tuple: (data, labels)."""

import os
import csv
import h5py
import glob
import numpy as np
import PokeAE.pokedataset32_vae_functions as utilities
from PIL import Image
from imgaug import augmenters as iaa


# Important parameters for the data set creation. Modifying them will change the final set generated.
image_format_to_use = "HSV"
full_dataset = False
use_augmentation = True


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
my_file = open('pokemontypes.csv')
csv_reader_dict = csv.DictReader(my_file)
# csv_reader_object = csv.reader(my_file)

one_hot_labels = []

type_to_categorical = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying',
                       'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water']
print(len(type_to_categorical))  # Must be 18 in size.

csv_reader_dict
current_count = 1

for row in csv_reader_dict:
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
        one_hot_labels.append([one_hot_type, one_hot_type_2])
    else:
        one_hot_labels.append([one_hot_type, one_hot_type])  # Add the same type TWICE.

print('Finished one hot encoding')
print(one_hot_labels)
one_hot_labels = np.asarray(one_hot_labels)

# Now, one_hot_labels has all the one_hot encodings for all the elements.
# We just have to put it into the same file as the pixel data and that's it.

source_folder = 'poke3232dataset/'

pixel_data = []
image_list = []

# This was important, if not re-sorted, it used a non-numeric order, which was not desired in this case.
filename_list = glob.glob(source_folder+'*.png')
filename_list.sort(key=sortKeyFunction)

# Now, the images need to be augmented BEFORE converting it to the HSV color space.
for filename in filename_list:
    with Image.open(filename) as image:
        # Note, it is always converted to RBG, to ignore the Alpha channel
        # and because augmentation library (aleju/imgaug) works on RGB color space.
        im = image.convert('RGB')
        pixel_matrix = np.asarray(im.getdata())  # Make the Width*Height*Depth matrices
        pixel_matrix = np.reshape(pixel_matrix, newshape=[32, 32, 3])
        pixel_matrix = pixel_matrix.astype(dtype=np.uint8)
        image_list.append(pixel_matrix)  # Need them all stored in one container for augmentation.

if use_augmentation:
    print("total images before augmentation is: " + str(len(image_list)))
    if not full_dataset:
        training_elements = int((len(image_list) / 100) * 85)  # This will give us 15% for testing
        test_images_list = image_list[training_elements:]  # First assign test ones, to avoid losing info.
        test_labels_list = one_hot_labels[training_elements:]
        image_list = image_list[0:training_elements]
        one_hot_labels = one_hot_labels[0:training_elements]

        print("getting test data augmented.")
        test_pixel_data, test_label_data = utilities.image_augmentation(test_images_list,
                                                                        test_labels_list, in_flip_lr=True,
                                                                        in_gamma_contrast=False,
                                                                        in_multiply_saturation=False,
                                                                        in_multiply_brightness=False,
                                                                        in_multiply_hue=False,
                                                                        in_gaussian_blur=False
                                                                        )
        test_pixel_data = np.asarray(test_pixel_data).astype(dtype=np.float)  # Float64 by default.
        test_pixel_data = utilities.convert_to_format(test_pixel_data, image_format_to_use)

    print("getting non-test data augmented.")
    # Now, do the augmentation for the non-test images. If no split was specified, this will contain all images.
    pixel_data, label_data = utilities.image_augmentation(image_list, one_hot_labels, in_flip_lr=True,
                                                          in_gamma_contrast=False,
                                                          in_multiply_saturation=False,
                                                          in_multiply_brightness=False,
                                                          in_multiply_hue=False, in_gaussian_blur=False
                                                          )
    print("data augmentation successful.")
else:  # If no augmentation will be applied.
    pixel_data = image_list  # Only pass the variables to the correct names.
    label_data = one_hot_labels

pixel_data = np.asarray(pixel_data).astype(dtype=np.float)  # Float64 by default.

# Now, we need to put them in the correct format according to the desired set.
pixel_data = utilities.convert_to_format(pixel_data, image_format_to_use)

# unison_shuffled_copies(pixel_data, one_hot_labels)  # Shuffled along the first axis only.

# Finally, put it into a h5f dataset and that's it.
if full_dataset:
    h5f = h5py.File('pokedataset32_full_' + image_format_to_use +
                    ('_Augmented' if use_augmentation else '') + '.h5', 'w')
    # These two lines below are used when the full data set is to be in one file.
    h5f.create_dataset('pokedataset32_X', data=pixel_data)
    h5f.create_dataset('pokedataset32_Y', data=label_data)
    h5f.close()
else:  # If it has train and test separation.
    h5f = h5py.File('pokedataset32_train_' + image_format_to_use +
                    ('_Augmented' if use_augmentation else '') + '.h5', 'w')
    # These four lines below are for the data split into train and test portions.
    h5f.create_dataset('pokedataset32_X', data=pixel_data)
    h5f.create_dataset('pokedataset32_Y', data=label_data)
    h5f.create_dataset('pokedataset32_X_test', data=test_pixel_data)
    h5f.create_dataset('pokedataset32_Y_test', data=test_label_data)
    h5f.close()
