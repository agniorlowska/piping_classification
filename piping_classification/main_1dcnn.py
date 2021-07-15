import run 

experiment = 0  # 
                # 0 - tooting vs quacking classification
                #  3-label classification (piping/queen/no queen)
                
random = 1 #
           # 0 - 4-fold cross validation
           # 1 - 70-30 random split
                
if experiment == 0 and random == 0:
    fold1_directory = "/home/agnieszka/Desktop/folds_only_piping/fold1"
    fold2_directory = "/home/agnieszka/Desktop/folds_only_piping/fold2"
    fold3_directory = "/home/agnieszka/Desktop/folds_only_piping/fold3"
    fold4_directory = "/home/agnieszka/Desktop/folds_only_piping/fold4"
    target_names = ['tooting', 'quacking']
    run.four_folds_1DCNN(fold1_directory, fold2_directory, fold3_directory, fold4_directory, target_names)    
elif experiment == 0 and random == 1:
    directory = "/home/agnieszka/Desktop/folds_only_piping/fold1"
    target_names = ['tooting', 'quacking']
    n_outputs = 2
    run.random_split_1DCNN(directory, target_names)

elif experiment == 1 and random == 0:
     fold1_directory = "/home/agnieszka/foldscnn/fold1"
     fold2_directory = "/home/agnieszka/foldscnn/fold2"
     fold3_directory = "/home/agnieszka/foldscnn/fold3"
     fold4_directory = "/home/agnieszka/foldscnn/fold4"    
     target_names=['piping', 'queen', 'no queen' ]
     run.four_folds_1DCNN(fold1_directory, fold2_directory, fold3_directory, fold4_directory, target_names)    
elif experiment == 1 and random == 1:
    directory = "/home/agnieszka/queen_noqueen/queen" 
    target_names=['piping', 'queen', 'no queen' ]
    run.random_split_1DCNN(directory, target_names)
               