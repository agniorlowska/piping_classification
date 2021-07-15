import data_augmentation as da

fold1_directory = "/home/agnieszka/Desktop/folds_piping_natural_beehive/fold1"
fold2_directory = "/home/agnieszka/Desktop/folds_piping_natural_beehive/fold2"
fold3_directory = "/home/agnieszka/Desktop/folds_piping_natural_beehive/fold3"
fold4_directory = "/home/agnieszka/Desktop/folds_piping_natural_beehive/fold4"

n_chunks = 27

audios, labels = da.data_augmentation(fold1_directory, fold2_directory, fold3_directory, fold4_directory, n_chunks)