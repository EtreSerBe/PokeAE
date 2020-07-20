from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import tflearn
import matplotlib.colors
import pokedataset32_vae_functions as utilities
import matplotlib.pyplot as plt

use_noise = False
current_dataset = 'pokedataset'
# current_dataset = 'anime_faces_'

use_anime_with_types = False
if not use_anime_with_types or current_dataset == 'pokedataset':
    X_full_HSV, Y_full_HSV, X_full_RGB, Y_full_RGB, X, Y, test_X, test_Y = utilities.ready_all_data_sets(
        current_dataset)
    if use_noise:
        X_noisy_HSV, Y_noisy_HSV = \
            utilities.prepare_dataset_for_input_layer("pokedataset32_train_NOISE_HSV_Two_Hot_Encoded_Augmented.h5")

        X_noisy_HSV_test, Y_noisy_HSV_test = \
            utilities.prepare_dataset_for_input_layer("pokedataset32_train_NOISE_HSV_Two_Hot_Encoded_Augmented.h5",
                                                      in_dataset_x_label="pokedataset32_X_test",
                                                      in_dataset_y_label="pokedataset32_Y_test")

        X_plus_noise = np.concatenate((X, X_noisy_HSV), axis=0)
        Y_plus_noise = np.concatenate((Y, Y_noisy_HSV), axis=0)
        Y_plus_noise = Y_plus_noise * 0.5
else:
    X, Y = utilities.prepare_dataset_for_input_layer(
        'anime_faces_32_train_HSV_Two_Hot_Encoded_Augmented_With_Types.h5', in_dataset_x_label='anime_faces_32_X',
        in_dataset_y_label='anime_faces_32_Y')
    test_X, test_Y = utilities.prepare_dataset_for_input_layer(
        'anime_faces_32_train_HSV_Two_Hot_Encoded_Augmented_With_Types.h5', in_dataset_x_label='anime_faces_32_X_test',
        in_dataset_y_label='anime_faces_32_Y_test')
    X_full_RGB, Y_full_RGB = utilities.prepare_dataset_for_input_layer(
        'anime_faces_32_full_RGB_Two_Hot_Encoded.h5', in_dataset_x_label='anime_faces_32_X',
        in_dataset_y_label='anime_faces_32_Y')

    X_first_half = X[0:int(len(X) / 2)]
    Y_first_half = Y[0:int(len(Y) / 2)]
    test_X_first_half = test_X[0:int(len(test_X) / 2)]
    test_Y_first_half = test_Y[0:int(len(test_Y) / 2)]
    """X_second_half = X[int(len(X) / 2):]
    Y_second_half = Y[int(len(Y) / 2):]
    test_X_second_half = test_X[int(len(test_X) / 2):]
    test_Y_second_half = test_Y[int(len(test_Y) / 2):]"""
    X_full_HSV = np.concatenate((X_first_half, test_X_first_half), axis=0)
    Y_full_HSV = np.concatenate((Y_first_half, test_Y_first_half), axis=0)
    Y_full_RGB = Y_full_HSV  # Replace it, since RGB was not saved with types.


Y = Y * 0.5
test_Y = test_Y * 0.5
Y_full_HSV = Y_full_HSV * 0.5  # np.clip(Y_full_HSV, 0.0, 1.0)
Y_full_RGB = Y_full_RGB * 0.5

small_X = np.concatenate((X[0:200], test_X[0:200]), axis=0)
small_Y = np.concatenate((Y[0:200], test_Y[0:200]), axis=0)
expanded_small_X = np.append(small_X, small_Y, axis=1)
len_X_div_2 = int(len(X)/2)
small_X_RGB = np.concatenate((X_full_RGB[0:200], X_full_RGB[len_X_div_2:len_X_div_2+200]), axis=0)

# Now we add the extra info from the Ys.
if use_noise:
    expanded_X = np.append(X_plus_noise, Y_plus_noise, axis=1)  # It already contains the Flip-left-right augmentation.
else:
    expanded_X = np.append(X, Y, axis=1)  # It already contains the Flip-left-right augmentation.
expanded_test_X = np.append(test_X, test_Y, axis=1)
expanded_full_X_HSV = np.append(X_full_HSV, Y_full_HSV, axis=1)  # Used to print everyone in the image.


print("getting network to load model*******************")
network_instance = utilities.get_network()

predict_full_dataset = True
optimizer_name = 'adam'
loss_name = 'vae_loss'
loaded_model_name = utilities.get_model_descriptive_name(optimizer_name, loss_name,
                                                         in_version='_anime_labels_V3_poke4_no_noise5')
final_model_name = utilities.get_model_descriptive_name(optimizer_name, loss_name,
                                                        in_version='_anime_labels_V3_poke4_no_noise5')
save_images = False

network_instance = tflearn.regression(network_instance,
                                      optimizer=optimizer_name,
                                      metric='R2',
                                      loss=utilities.vae_loss,
                                      learning_rate=0.0000170)  # adagrad? #adadelta #nesterov did good,

model = tflearn.DNN(network_instance)
print("LOADING MODEL.")
model.load("saved_models/" + loaded_model_name)
# Variable so you don't have to go all the way down just to change that
model_save_name = "saved_models/" + final_model_name
reconstructed_pixels = []
reconstructed_types = []

for lap in range(0, 1):
    # Now, continue the training with VERY SMALL batch sizes, so it can learn specifics about each pokemon.
    model.fit(expanded_X, Y_targets=expanded_X,
              n_epoch=50,
              shuffle=True,
              show_metric=True,
              snapshot_epoch=True,
              batch_size=128,
              # validation_set=0.15,  # It also accepts a float < 1 to performs a data split over training data.
              validation_set=(expanded_test_X, expanded_test_X),
              # We use it for validation for now. But also test.
              run_id='encoder_decoder')

    # Now we print how it has progressed to see if we want to keep these changes.
    # print("getting samples to show on screen.")
    if predict_full_dataset:
        predicted_Y = Y_full_RGB
        exporting_RGB = X_full_RGB
        encode_decode_sample = utilities.predict_batches(expanded_full_X_HSV, model, in_samples_per_batch=64)
    else:
        predicted_Y = small_Y
        exporting_RGB = small_X_RGB
        encode_decode_sample = utilities.predict_batches(expanded_small_X, model, in_samples_per_batch=64)

    reconstructed_pixels = []
    reconstructed_types = []
    reconstructed_pixels, reconstructed_types = utilities.reconstruct_pixels_and_types(encode_decode_sample)
    correct_indices = utilities.export_types_csv(predicted_Y, reconstructed_types)

    if save_images:
        print("Exporting reconstructed pokemon as an image.")
        utilities.export_as_atlas(exporting_RGB, reconstructed_pixels, name_annotations='standard_retrain_' + str(lap))

        # This is used to export an image only containing the ones whose types were correctly predicted by the NN.
        correct_X_RGB = [exporting_RGB[i] for i in correct_indices]
        correct_reconstructed_pixels = [reconstructed_pixels[i] for i in correct_indices]
        utilities.export_as_atlas(correct_X_RGB, correct_reconstructed_pixels, name_annotations='correct')

    # Compare original images with their reconstructions.
    f, a = plt.subplots(2, 10, figsize=(20, 2), squeeze=False)  # figsize=(50, 2),
    for i in range(5):
        # reshaped_pokemon = np.multiply(reshaped_pokemon, 255.0)
        reshaped_pokemon = np.reshape(np.asarray(exporting_RGB[-(i*3+1)]), [1024, 3])
        reshaped_pokemon = np.asarray(reshaped_pokemon).flatten()
        temp = [[ii] for ii in list(reshaped_pokemon)]  # WTH? Python, you're drunk haha.
        print("ORIGINAL Types for Pokemon " + str(i) + " are: ")
        utilities.print_pokemon_types(predicted_Y[-(i*3+1)])
        a[0][i].imshow(np.reshape(temp, (32, 32, 3)))
        temp = [[ii] for ii in list(reconstructed_pixels[-(i*3+1)])]
        a[1][i].imshow(np.reshape(temp, (32, 32, 3)))
        print("Types for Pokemon " + str(i) + " are: ")
        utilities.print_pokemon_types(reconstructed_types[-(i*3+1)])

        reshaped_pokemon = np.reshape(np.asarray(exporting_RGB[(i * 3 + 1)]), [1024, 3])
        reshaped_pokemon = np.asarray(reshaped_pokemon).flatten()
        temp = [[ii] for ii in list(reshaped_pokemon)]  # WTH? Python, you're drunk haha.
        print("ORIGINAL Types for Pokemon " + str(i+5) + " are: ")
        utilities.print_pokemon_types(predicted_Y[(i * 3 + 1)])
        a[0][i+5].imshow(np.reshape(temp, (32, 32, 3)))
        temp = [[ii] for ii in list(reconstructed_pixels[(i * 3 + 1)])]
        a[1][i+5].imshow(np.reshape(temp, (32, 32, 3)))
        print("Types for Pokemon " + str(i+5) + " are: ")
        utilities.print_pokemon_types(reconstructed_types[(i * 3 + 1)])
    f.show()
    plt.draw()
    # input('Press E to exit')
    plt.waitforbuttonpress()

print('waiting for button press to save the model')

if not save_images:
    print("Exporting reconstructed pokemon as an image.")
    utilities.export_as_atlas(exporting_RGB, reconstructed_pixels, name_annotations='standard_retrain_final')
    correct_indices = utilities.export_types_csv(predicted_Y, reconstructed_types)

print("Now overwriting the model")
model.save(model_save_name)
