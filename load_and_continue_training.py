from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import tflearn
import matplotlib.colors
import PokeAE.pokedataset32_vae_functions as utilities
import matplotlib.pyplot as plt

X_full_HSV, Y_full_HSV = utilities.prepare_dataset_for_input_layer('pokedataset32_full_HSV_Two_Hot_Encoded.h5')

X_full_RGB, Y_full_RGB = utilities.prepare_dataset_for_input_layer('pokedataset32_full_RGB_Two_Hot_Encoded.h5')

X, Y = utilities.prepare_dataset_for_input_layer('pokedataset32_train_HSV_Two_Hot_Encoded_Augmented.h5')

test_X, test_Y = utilities.prepare_dataset_for_input_layer('pokedataset32_train_HSV_Two_Hot_Encoded_Augmented.h5',
                                                           in_dataset_x_label='pokedataset32_X_test',
                                                           in_dataset_y_label='pokedataset32_Y_test')

Y = Y * 0.5
test_Y = test_Y * 0.5
Y_full_HSV = Y_full_HSV * 0.5  # np.clip(Y_full_HSV, 0.0, 1.0)
Y_full_RGB = Y_full_RGB * 0.5

# Now we add the extra info from the Ys.
expanded_X = np.append(X, Y, axis=1)  # It already contains the Flip-left-right augmentation.
expanded_test_X = np.append(test_X, test_Y, axis=1)
expanded_full_X_HSV = np.append(X_full_HSV, Y_full_HSV, axis=1)  # Used to print everyone in the image.


print("getting network to load model*******************")
network_instance = utilities.get_network()

optimizer_name = 'adam'
loss_name = 'vae_loss'
loaded_model_name = utilities.get_model_descriptive_name(optimizer_name, loss_name, in_version='_2by2_V2')
final_model_name = utilities.get_model_descriptive_name(optimizer_name, loss_name, in_version='_2by2_V2')
save_images = False


network_instance = tflearn.regression(network_instance,
                                      optimizer=optimizer_name,
                                      # optimizer='rmsprop',
                                      metric='R2',
                                      loss=utilities.vae_loss,
                                      # loss=utilities.vae_loss_abs_error,
                                      learning_rate=0.0001)  # adagrad? #adadelta #nesterov did good,

model = tflearn.DNN(network_instance)
print("LOADING MODEL.")
model.load("Saved models/" + loaded_model_name)
# Variable so you don't have to go all the way down just to change that
model_save_name = "Saved models/" + final_model_name
reconstructed_pixels = []
reconstructed_types = []

for lap in range(0, 3):
    # Now, continue the training with VERY SMALL batch sizes, so it can learn specifics about each pokemon.
    model.fit(expanded_X, Y_targets=expanded_X,
              n_epoch=5,
              shuffle=True,
              show_metric=True,
              snapshot_epoch=True,
              batch_size=32,
              # validation_set=0.15,  # It also accepts a float < 1 to performs a data split over training data.
              validation_set=(expanded_test_X, expanded_test_X),
              # We use it for validation for now. But also test.
              run_id='encoder_decoder')

    # Now we print how it has progressed to see if we want to keep these changes.
    # print("getting samples to show on screen.")
    encode_decode_sample = utilities.predict_batches(expanded_full_X_HSV, model, in_number_of_chunks=10)

    reconstructed_pixels = []
    reconstructed_types = []

    reconstructed_pixels, reconstructed_types = utilities.reconstruct_pixels_and_types(encode_decode_sample)
    correct_indices = utilities.export_types_csv(Y_full_RGB, reconstructed_types)

    if save_images:
        print("Exporting reconstructed pokemon as an image.")
        utilities.export_as_atlas(X_full_RGB, reconstructed_pixels, name_annotations='standard_retrain_' + str(lap))

        # This is used to export an image only containing the ones whose types were correctly predicted by the NN.
        correct_X_RGB = [X_full_RGB[i] for i in correct_indices]
        correct_reconstructed_pixels = [reconstructed_pixels[i] for i in correct_indices]
        utilities.export_as_atlas(correct_X_RGB, correct_reconstructed_pixels, name_annotations='correct')

    # Compare original images with their reconstructions.
    f, a = plt.subplots(2, 10, figsize=(20, 2), squeeze=False)  # figsize=(50, 2),
    for i in range(5):
        # reshaped_pokemon = np.multiply(reshaped_pokemon, 255.0)
        reshaped_pokemon = np.reshape(np.asarray(X_full_RGB[-(i*3+1)]), [1024, 3])
        reshaped_pokemon = np.asarray(reshaped_pokemon).flatten()
        temp = [[ii] for ii in list(reshaped_pokemon)]  # WTH? Python, you're drunk haha.
        print("ORIGINAL Types for Pokemon " + str(i) + " are: ")
        utilities.print_pokemon_types(Y_full_RGB[-(i*3+1)])
        a[0][i].imshow(np.reshape(temp, (32, 32, 3)))
        temp = [[ii] for ii in list(reconstructed_pixels[-(i*3+1)])]
        a[1][i].imshow(np.reshape(temp, (32, 32, 3)))
        print("Types for Pokemon " + str(i) + " are: ")
        utilities.print_pokemon_types(reconstructed_types[-(i*3+1)])

        reshaped_pokemon = np.reshape(np.asarray(X_full_RGB[(i * 3 + 1)]), [1024, 3])
        reshaped_pokemon = np.asarray(reshaped_pokemon).flatten()
        temp = [[ii] for ii in list(reshaped_pokemon)]  # WTH? Python, you're drunk haha.
        print("ORIGINAL Types for Pokemon " + str(i+5) + " are: ")
        utilities.print_pokemon_types(Y_full_RGB[(i * 3 + 1)])
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
# plt.waitforbuttonpress()

if not save_images:
    print("Exporting reconstructed pokemon as an image.")
    utilities.export_as_atlas(X_full_RGB, reconstructed_pixels, name_annotations='standard_retrain_final')
    correct_indices = utilities.export_types_csv(Y_full_RGB, reconstructed_types)

print("Now overwriting the model")
model.save(model_save_name)
