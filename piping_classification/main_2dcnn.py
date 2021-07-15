import run 

#here you can choose the classification method:
experiment = 0  # 
                # 0 - tooting vs quacking classification
                #  3-label classification (piping/queen/no queen)

random = 1 #
           # 0 - 4-fold cross validation
           # 1 - 70-30 random split

#here you can choose the approach of feature extraction:
mode = 2 #
         # 0 - mean-STFT + DA
         # 1 - mean-STFT
         # 2 - MFCCS
         # 3 - STFT without mean spectrogram (input size 513x44)

#here you can choose value of B:
n_chunks = 27 
#some cnn performance properties
num_epochs = 50
num_batch_size = 145



if experiment == 0 and random == 0:
    fold1_directory = "/home/agnieszka/Desktop/folds_only_piping/fold1"
    fold2_directory = "/home/agnieszka/Desktop/folds_only_piping/fold2"
    fold3_directory = "/home/agnieszka/Desktop/folds_only_piping/fold3"
    fold4_directory = "/home/agnieszka/Desktop/folds_only_piping/fold4"
    target_names = ['tooting', 'quacking']
    class_names = ['tooting', 'quacking']
    n_outputs = 2
    run.four_folds(fold1_directory, fold2_directory, fold3_directory, fold4_directory, n_chunks, mode, n_outputs, num_batch_size, num_epochs, class_names, target_names)    
elif experiment == 0 and random == 1:
    toot_directory = "/home/agnieszka/Desktop/folds_only_piping/fold1"
    quack_directory = "/home/agnieszka/Desktop/folds_only_piping/fold3"
    target_names = ['tooting', 'quacking']
    class_names = ['tooting', 'quacking']
    n_outputs = 2
    run.random_split_2label(toot_directory, quack_directory, n_chunks, mode, n_outputs, num_batch_size, num_epochs, class_names, target_names)
elif experiment == 1 and random == 0:
     fold1_directory = "/home/agnieszka/foldscnn/fold1"
     fold2_directory = "/home/agnieszka/foldscnn/fold2"
     fold3_directory = "/home/agnieszka/foldscnn/fold3"
     fold4_directory = "/home/agnieszka/foldscnn/fold4"    
     class_names= ['piping', 'queen', 'no queen' ]
     target_names=['piping', 'queen', 'no queen' ]
     n_outputs = 3
     run.four_folds(fold1_directory, fold2_directory, fold3_directory, fold4_directory, n_chunks, mode, n_outputs, num_batch_size, num_epochs, class_names, target_names)
elif experiment == 1 and random == 1:
     queen_directory = "/home/agnieszka/queen_noqueen/queen"
     noqueen_directory = "/home/agnieszka/queen_noqueen/noqueen"
     pip_directory = "/home/agnieszka/piping_quacking_dataset"    
     class_names= ['piping', 'queen', 'no queen' ]
     target_names=['piping', 'queen', 'no queen' ]
     n_outputs = 3
     run.random_split_3label(queen_directory, noqueen_directory, pip_directory, n_chunks, mode, n_outputs, num_batch_size, num_epochs, class_names, target_names)









