from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import tflearn
import matplotlib.colors
import pokedataset32_vae_functions as utilities
import matplotlib.pyplot as plt

current_dataset = 'pokedataset'
# current_dataset = 'anime_faces_'

X_full_HSV, Y_full_HSV, X_full_RGB, Y_full_RGB, X, Y, test_X, test_Y = utilities.ready_all_data_sets(current_dataset)

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
expanded_X = np.append(X, Y, axis=1)  # It already contains the Flip-left-right augmentation.
expanded_test_X = np.append(test_X, test_Y, axis=1)
expanded_full_X_HSV = np.append(X_full_HSV, Y_full_HSV, axis=1)  # Used to print everyone in the image.


print("getting network to load model*******************")
network_instance = utilities.get_network()

predict_full_dataset = True
optimizer_name = 'adam'
loss_name = 'vae_loss'
# V5 IS THE SAFE ONE, V6 IS CHANGING VAE_LOSS
# V7 got to 0.85 but has the KL divergence weight reduced, so it's not the safest haha.
# V9 has 0.05 in KL_divergence weight now. But it reached 0.90 R2! V10 to 0.91
# V11 is trying to recover KL_Divergence cost to 1.0, starting with 0.15, it went from 0.91 to 0.90
# with 0.2 went from 0.9 to 0.89
# V11 didn't survive going back to 1.0 KL_Divergence cost, it has only 0.84 R2.
# V12 Was the best after returning to 1.0 KL_divergence and it has around 0.83 R2.
# The ones with sigmoid instead of relu don't perform as well.
# I'm moving to train with pokemon over with both V10 and V12, see how it goes.
# In the generative aspect, V12 proved WAY better than V10 over anime faces. So, pokemon training on V12 first.
loaded_model_name = utilities.get_model_descriptive_name(optimizer_name, loss_name, in_version='_anime_faces_V12_Poke4')
final_model_name = utilities.get_model_descriptive_name(optimizer_name, loss_name, in_version='_anime_faces_V12_Poke4')
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
model.load("saved_models/" + loaded_model_name)
# Variable so you don't have to go all the way down just to change that
model_save_name = "saved_models/" + final_model_name
reconstructed_pixels = []
reconstructed_types = []

for lap in range(0, 3):
    # Now, continue the training with VERY SMALL batch sizes, so it can learn specifics about each pokemon.
    model.fit(expanded_X, Y_targets=expanded_X,
              n_epoch=3,
              shuffle=True,
              show_metric=True,
              snapshot_epoch=True,
              batch_size=16,
              # validation_set=0.15,  # It also accepts a float < 1 to performs a data split over training data.
              validation_set=(expanded_test_X, expanded_test_X),
              # We use it for validation for now. But also test.
              run_id='encoder_decoder')

    # Now we print how it has progressed to see if we want to keep these changes.
    # print("getting samples to show on screen.")
    if predict_full_dataset:
        predicted_X = X
        predicted_Y = Y_full_RGB
        exporting_RGB = X_full_RGB
        encode_decode_sample = utilities.predict_batches(expanded_full_X_HSV, model, in_samples_per_batch=64)
    else:
        predicted_X = small_X
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
