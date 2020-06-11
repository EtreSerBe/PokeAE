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

# full RGB; full HSV, and train HSV augmented are the indispensable ones.
# Important parameters for the data set creation. Modifying them will change the final set generated.
image_format_to_use = "HSV"
full_dataset = True
use_augmentation = False
use_type_information = False
use_two_hot_encoding = True

# This is only used for the Pokemon images with swapped types, which are a special dataset for testing.
if not use_type_information:
    source_folder = 'C:/Users/Adrián González/Desktop/anime faces dataset/'
    csv_type_file = ''
else:
    source_folder = 'C:/Users/Adrián González/Desktop/anime faces dataset/'
    csv_type_file = 'anime_faces_types.csv'
    # Load CSV file, indicate that the first column represents labels
    # Now, we can check for the
    my_file = open(csv_type_file)
    csv_reader_dict = csv.DictReader(my_file)


# As seen in https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def sortKeyFunction(s):
    print(os.path.basename(s))
    number = os.path.basename(s)[:-4]

    print(number)
    s = number
    return float(s)  # :-4 is to not return the extension


# csv_reader_object = csv.reader(my_file)
filename_list = glob.glob(source_folder+'*.jpg')
filename_list.sort(key=sortKeyFunction)

for filename in filename_list:
    with Image.open(filename) as image:
        # Note, it is always converted to RBG, to ignore the Alpha channel
        # and because augmentation library (aleju/imgaug) works on RGB color space.
        im = image.convert('RGB')
        reduced_im = im.resize((32, 32), resample=Image.BICUBIC)
        number = os.path.basename(filename)[:-4]
        reduced_im.save(source_folder + "32_32/" + number + "_32.jpg")
