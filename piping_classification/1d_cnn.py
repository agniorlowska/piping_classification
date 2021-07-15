import os
import classification as clf
import numpy as np
import functions as fc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import my_tools as mt
from IPython.display import display, Audio

import tensorflow as tf
from tensorflow import keras

#some global variables
SAMPLING_RATE = 22050
VALID_SPLIT = 0.1
TEST_SPLIT = 0.3
SHUFFLE_SEED =  43
SAMPLES_TO_DISPLAY = 10
SCALE = 0.5

BATCH_SIZE = 128
EPOCHS = 50

directory = "/home/agnieszka/dataset_beehive/queen_noqueen_folds/fold2"
target_names = ['queen', 'no queen']

names = os.listdir(directory)
audio_paths = []
labels = []
for label, name in enumerate(names):
    print("Processing speaker {}".format(name,))
    
    speaker_sample_paths = [
        os.path.join(directory, filepath)
        for filepath in os.listdir(directory)
        if filepath.endswith(".wav")
    ]
    q = fc.queen_info(name)
    print(q)
    labels.append(q)
    print(len(labels))
    print(len(speaker_sample_paths))
    audio_paths = speaker_sample_paths
    
print(
    "Found {} files belonging to {} classes.".format(len(audio_paths), len(set(labels)))
)

rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)

# Split into training and validation
num_val_samples = int(VALID_SPLIT * len(audio_paths))
num_test_samples = int(TEST_SPLIT * len(audio_paths))
print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
train_audio_paths = audio_paths[:-num_val_samples]
train_labels = labels[:-num_val_samples]


print("Using {} files for validation.".format(num_val_samples))
valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]

print("Using {} files for testing".format(num_test_samples))
test_audio_paths = audio_paths[-num_test_samples:]
test_labels = labels[-num_test_samples:]

train_ds = fc.paths_and_labels_to_dataset(train_audio_paths, train_labels)
train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
    BATCH_SIZE
)

valid_ds = fc.paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)

test_ds = fc.paths_and_labels_to_dataset(test_audio_paths, test_labels)
test_ds = test_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)


#Transform audio wave to the frequency domain using `audio_to_fft`
train_ds = train_ds.map(
    lambda x, y: (fc.audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

valid_ds = valid_ds.map(
    lambda x, y: (fc.audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
)
valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)

test_ds = test_ds.map(
    lambda x, y: (fc.audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)


model = clf.make_model_1DCNN((SAMPLING_RATE // 2, 1), len(names))
model.summary()

model.compile(
    optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model_save_filename = "model.h5"

earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
    model_save_filename, monitor="val_accuracy", save_best_only=True
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=(valid_ds),
    callbacks=[earlystopping_cb, mdlcheckpoint_cb],
)

print(model.evaluate(valid_ds))
y_pred = []

print(test_audio_paths)
print(test_labels)

for audios, labels in test_ds.take(1):
    # Get the signal FFT
    #ffts = fc.audio_to_fft(audios)
    #print(len(ffts))
    # Predict
    y_pred = model.predict(audios)
    y_pred = np.argmax(y_pred, axis=-1)
    # Take random samples
    print(len(audios))
    print(audios)
    print(len(labels))
    print(labels)
    print(len(y_pred))
    print(y_pred)
    audios = audios.numpy()
    labels = labels.numpy()
    cnf_matrix = confusion_matrix(labels, y_pred)
    np.set_printoptions(precision=2)
    
    mt.plot_confusion_matrix(cnf_matrix, classes=target_names,
                          title='Confusion matrix:')
    plt.show()
    print ('\nClasification report for fold:\n', 
           classification_report(labels, y_pred, target_names=target_names ))
 

