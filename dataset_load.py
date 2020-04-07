"""The Dataset is stored in a csv file, so we can use TFLearn load_csv() function to load the data from
file into a python list. We specify 'target_column' argument to indicate that our labels (survived or not)
 are located in the first column (id: 0). The function will return a tuple: (data, labels)."""

"""import numpy as np
import tflearn

# Download the Titanic dataset
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

from tflearn.data_utils import load_csv
data, labels = load_csv('pokemontypes.csv', target_column=0, columns_to_ignore=0,
                        categorical_labels=True, n_classes=18)
"""
import os
import csv
import h5py
import glob
from PIL import Image
import numpy as np


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

    print(current_count)
    current_count += 1
    print(one_hot_type)
    one_hot_labels.append([one_hot_type, one_hot_type_2])


print('Finished one hot encoding')
print(one_hot_labels)
one_hot_labels = np.asarray(one_hot_labels)

# Now, one_hot_labels has all the one_hot encodings for all the elements.
# We just have to put it into the same file as the pixel data and that's it.

source_folder = 'poke3232dataset/'

pixel_data = []

filename_list = glob.glob(source_folder+'*.png')
filename_list.sort(key=sortKeyFunction)
print('The filenames ordered by number is: ')
print(filename_list)


# X is the pixel data for the pokemon.
for filename in filename_list:
    with Image.open(filename) as image:
        im = image.convert('RGB')
        pixel_list = np.asarray(im.getdata()).flatten()
        pixel_list = np.true_divide(pixel_list, 255.)  # Very important to normalize your inputs!
        print(filename)
        # NOTE: Check if the pixels are triplets or are already flattened.
        print('the number of pixels each image has is: ' + str(len(pixel_list)))
        pixel_data.append(pixel_list)  # add it to the variable with all the information.

# Should be 809 or something.
# print('The length of pixel_data is: ' + str(len(pixel_data)))  # correctly printed.

# Make a 3-fold cross-validation split, so it's stable during development.
pixel_data = np.asarray(pixel_data)
print(pixel_data)

# unison_shuffled_copies(pixel_data, one_hot_labels)  # Shuffled along the first axis only.

# NOTE: Make a better way to automatize this latter. (26/2/2020)
pixels_1 = pixel_data[0:370]
pixels_2 = pixel_data[370:741]
pixels_3 = pixel_data[741:891]
labels_1 = one_hot_labels[0:370]
labels_2 = one_hot_labels[370:741]
labels_3 = one_hot_labels[741:891]

pixels = np.concatenate((pixels_1, pixels_2))
labels = np.concatenate((labels_1, labels_2))

print(len(pixels))
print(len(labels))
print(len(pixels_3))
print(len(labels_3))

# Finally, put it into a h5f dataset and that's it.
h5f = h5py.File('pokedataset32_12_3_RGB.h5', 'w')
h5f.create_dataset('pokedataset32_X', data=pixels)
h5f.create_dataset('pokedataset32_Y', data=labels)
h5f.create_dataset('pokedataset32_X_test', data=pixels_3)
h5f.create_dataset('pokedataset32_Y_test', data=labels_3)
h5f.close()

# build_hdf5_image_dataset


"""
# Load path/class_id image file:
# dataset_file = 'my__pokemon_dataset.txt'
dataset_file = 'poke3232dataset'

# Build a HDF5 dataset (only required once)
from tflearn.data_utils import build_hdf5_image_dataset
build_hdf5_image_dataset(dataset_file, image_shape=(32, 32), mode='folder', output_path='poke3232_dataset.h5',
                         categorical_labels=True, normalize=True)"""
