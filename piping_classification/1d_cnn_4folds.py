import os
import functions as fc
from sklearn.metrics import accuracy_score
import classification as clf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import my_tools as mt
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






##piping vs quacking
#fold1_directory = "/home/agnieszka/queen_noqueen_folds/fold1"
#fold2_directory = "/home/agnieszka/queen_noqueen_folds/fold2"
#fold3_directory = "/home/agnieszka/queen_noqueen_folds/fold3"
#fold4_directory = "/home/agnieszka/queen_noqueen_folds/fold4"
#target_names = ['queen', 'no queen']

#queen vs no queen
#fold1_directory = "/home/agnieszka/Desktop/folds_little/fold1"
#fold2_directory = "/home/agnieszka/Desktop/folds_little/fold2"
#fold3_directory = "/home/agnieszka/Desktop/folds_little/fold3"
#fold4_directory = "/home/agnieszka/Desktop/folds_little/fold4"
#target_names = ['queen', 'no queen']

#4 label
# =============================================================================
fold1_directory = "/home/agnieszka/Desktop/folds_piping_natural_beehive/fold1"
fold2_directory = "/home/agnieszka/Desktop/folds_piping_natural_beehive/fold2"
fold3_directory = "/home/agnieszka/Desktop/folds_piping_natural_beehive/fold3"
fold4_directory = "/home/agnieszka/Desktop/folds_piping_natural_beehive/fold4"
target_names = ['piping', 'queen', 'no queen']
# =============================================================================



names = os.listdir(fold1_directory)
fold1_audio_paths = []
fold1_labels = []
for label, name in enumerate(names):
    speaker_sample_paths = [
        os.path.join(fold1_directory, filepath)
        for filepath in os.listdir(fold1_directory)
        if filepath.endswith(".wav")
    ]
    q = fc.queen_info(name)
    fold1_labels.append(q)
    fold1_audio_paths = speaker_sample_paths
    
print(
    "Fold1: Found {} files belonging to {} classes.".format(len(fold1_audio_paths), len(set(fold1_labels)))
)

rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(fold1_audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(fold1_labels)
############################################
names = os.listdir(fold2_directory)
fold2_audio_paths = []
fold2_labels = []
for label, name in enumerate(names):
    speaker_sample_paths = [
        os.path.join(fold2_directory, filepath)
        for filepath in os.listdir(fold2_directory)
        if filepath.endswith(".wav")
    ]
    q = fc.queen_info(name)
    fold2_labels.append(q)
    fold2_audio_paths = speaker_sample_paths
    
print(
    "Fold2: Found {} files belonging to {} classes.".format(len(fold2_audio_paths), len(set(fold2_labels)))
)

rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(fold2_audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(fold2_labels)
############################################
names = os.listdir(fold3_directory)
fold3_audio_paths = []
fold3_labels = []
for label, name in enumerate(names):
    speaker_sample_paths = [
        os.path.join(fold3_directory, filepath)
        for filepath in os.listdir(fold3_directory)
        if filepath.endswith(".wav")
    ]
    q = fc.queen_info(name)
    fold3_labels.append(q)
    fold3_audio_paths = speaker_sample_paths
    
print("Fold3: Found {} files belonging to {} classes.".format(len(fold3_audio_paths), len(set(fold3_labels)))
)

rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(fold3_audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(fold3_labels)
########################################
names = os.listdir(fold4_directory)
fold4_audio_paths = []
fold4_labels = []
for label, name in enumerate(names):
    speaker_sample_paths = [
        os.path.join(fold4_directory, filepath)
        for filepath in os.listdir(fold4_directory)
        if filepath.endswith(".wav")
    ]
    q = fc.queen_info(name)
    fold4_labels.append(q)
    fold4_audio_paths = speaker_sample_paths
    
print(
    "Fold4: Found {} files belonging to {} classes.".format(len(fold4_audio_paths), len(set(fold4_labels)))
)

rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(fold4_audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(fold4_labels)

# Split into training, validation and testing for all folds
audio_paths_1 = fold2_audio_paths + fold3_audio_paths + fold4_audio_paths
audio_paths_2 = fold1_audio_paths + fold3_audio_paths + fold4_audio_paths
audio_paths_3 = fold1_audio_paths + fold2_audio_paths + fold4_audio_paths
audio_paths_4 = fold1_audio_paths + fold2_audio_paths + fold3_audio_paths

labels_1 = fold2_labels + fold3_labels + fold4_labels
labels_2 = fold1_labels + fold3_labels + fold4_labels
labels_3 = fold1_labels + fold2_labels + fold4_labels
labels_4 = fold1_labels + fold2_labels + fold3_labels

train_ds_1, valid_ds_1, test_ds_1 = fc.prepare_training_testing(audio_paths_1, labels_1, fold1_audio_paths, fold1_labels)
train_ds_2, valid_ds_2, test_ds_2 = fc.prepare_training_testing(audio_paths_2, labels_2, fold2_audio_paths, fold2_labels)
train_ds_3, valid_ds_3, test_ds_3 = fc.prepare_training_testing(audio_paths_3, labels_3, fold3_audio_paths, fold3_labels)
train_ds_4, valid_ds_4, test_ds_4 = fc.prepare_training_testing(audio_paths_4, labels_4, fold4_audio_paths, fold4_labels)



# Create the model
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

#Training and prediction 
y_pred_1, fin_labels_1 = fc.training_and_evaluation(model, train_ds_1, valid_ds_1, test_ds_1, earlystopping_cb, mdlcheckpoint_cb, target_names)
y_pred_2, fin_labels_2 = fc.training_and_evaluation(model, train_ds_2, valid_ds_2, test_ds_2, earlystopping_cb, mdlcheckpoint_cb, target_names)
y_pred_3, fin_labels_3 = fc.training_and_evaluation(model, train_ds_3, valid_ds_3, test_ds_3, earlystopping_cb, mdlcheckpoint_cb, target_names)
y_pred_4, fin_labels_4 = fc.training_and_evaluation(model, train_ds_4, valid_ds_4, test_ds_4, earlystopping_cb, mdlcheckpoint_cb, target_names)


rounded_predictions = np.concatenate((y_pred_1, y_pred_2, y_pred_3, y_pred_4))
rounded_labels = np.concatenate((fin_labels_1, fin_labels_2, fin_labels_3, fin_labels_4))
    
cnf_matrix = confusion_matrix(rounded_labels, rounded_predictions)
np.set_printoptions(precision=2)
mt.plot_confusion_matrix(cnf_matrix, classes=target_names,
                          title='Confusion matrix:')
print ('\nClasification report:\n', classification_report(rounded_labels, rounded_predictions, target_names=target_names))
print('Accuracy: ', accuracy_score(rounded_labels, rounded_predictions))
df = mt.get_classification_report(rounded_labels, rounded_predictions)
print(df)




    