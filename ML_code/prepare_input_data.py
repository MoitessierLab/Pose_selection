# Author: Anne Labarre
# McGill University, Montreal, QC, Canada

import random
import rdkit
import os
import pickle
import torch
import pandas as pd
import numpy as np
import shutil
import time

import math
from sklearn import preprocessing
import xgboost as xgb
import matplotlib.pyplot as plt

def split_set(args):

    #if os.path.isfile(args.CSV_data_dir + args.list_trainset) is True and os.path.isfile(args.CSV_data_dir + args.list_testset) is True and os.path.isfile(args.CSV_data_dir + args.list_valset) is True:

    train_list = []
    val_list = []
    test_list = []

    # Load the train set
    with open(args.CSV_data_dir + args.list_trainset, 'rb') as list_f:
        print('| Training set selected from %-86s |' % args.list_trainset)
        for line in list_f:
            train_list.append(line.strip().decode('UTF-8'))  # store the PDB name in the train set in a list
        chunks = []
        counterNaN_train=0

        for protein_id in train_list:  # for each pdb, read its CSV file and store it in a dataframe
            if os.path.isfile(args.CSV_data_dir + protein_id + args.CSV_suffix) is True:
            #if os.path.isfile(args.CSV_data_dir + protein_id + '_good' + args.CSV_suffix) is True:
                chunk = pd.read_csv(args.CSV_data_dir + protein_id + args.CSV_suffix, index_col=False)
                #chunk = pd.read_csv(args.CSV_data_dir + protein_id + '_good' + args.CSV_suffix, index_col=False) #, chunksize = 500, engine = "python") # index_col=False)
                if chunk.isnull().values.any(): # check for NaN instances
                    train_list.remove(protein_id)
                    counterNaN_train+=1
                else:
                    #chunk['Label'] = 1
                    chunks.append(chunk)
                    del chunk
                    
                    # if os.path.isfile(args.CSV_data_dir + protein_id + '_bad' + args.CSV_suffix) is True:
                    #     chunk = pd.read_csv(args.CSV_data_dir + protein_id + '_bad' + args.CSV_suffix, index_col=False) #, chunksize = 500, engine = "python") # index_col=False)
                    # 
                    #     if chunk.isnull().values.any(): # check for NaN instances
                    #         train_list.remove(protein_id)
                    #         counterNaN_train+=1
                    #     else:
                    #         chunk['Label'] = 0
                    #         chunks.append(chunk)
                    #         del chunk
            else:
                train_list.remove(protein_id)
                #print('| Could not find %s%s%-50s |' % ( args.CSV_data_dir, protein_id, args.CSV_suffix ))
        train_set = pd.concat(chunks).dropna()
        del chunks

    # Load the validation set
    with open(args.CSV_data_dir + args.list_valset, 'rb') as list_f:
        print('| Validation set selected from %-83s  |' % args.list_valset)
        for line in list_f:
            val_list.append(line.strip().decode('UTF-8'))  # store the PDB names in the val set in a list
        chunks = []
        counterNaN_val = 0

        for protein_id in val_list:  # for each pdb, read its CSV file and store it in a dataframe
            if os.path.isfile(args.CSV_data_dir + protein_id + args.CSV_suffix) is True:
            #if os.path.isfile(args.CSV_data_dir + protein_id + '_good' + args.CSV_suffix) is True:
                chunk = pd.read_csv(args.CSV_data_dir + protein_id + args.CSV_suffix, index_col=False)
                #chunk = pd.read_csv(args.CSV_data_dir + protein_id + '_good' + args.CSV_suffix, index_col=False)
                if chunk.isnull().values.any():
                    val_list.remove(protein_id)
                    counterNaN_val+=1
                else:
                    #chunk['Label'] = 1
                    chunks.append(chunk)
                    del chunk

                    # if os.path.isfile(args.CSV_data_dir + protein_id + '_bad' + args.CSV_suffix) is True:
                    #     chunk = pd.read_csv(args.CSV_data_dir + protein_id  + '_bad' + args.CSV_suffix, index_col=False)
                    #     if chunk.isnull().values.any():
                    #         val_list.remove(protein_id)
                    #         counterNaN_val += 1
                    #     else:
                    #         chunk['Label'] = 0
                    #         chunks.append(chunk)
                    #         del chunk
            else:
                val_list.remove(protein_id)
                #print('| Could not find %s%s%-50s |' % (args.CSV_data_dir, protein_id, args.CSV_suffix))
        val_set = pd.concat(chunks).dropna()
        del chunks

    # Load the test set
    with open(args.CSV_data_dir + args.list_testset, 'rb') as list_f:
        print('| Testing set selected from %-86s  |' % args.list_testset)
        for line in list_f:
            test_list.append(line.strip().decode('UTF-8'))  # store the PDB names in the test set in a list

        chunks = []
        counterNaN_test = 0

        for protein_id in test_list:  # for each pdb, read its CSV file and store it in a dataframe
            if os.path.isfile(args.CSV_data_dir + protein_id + args.CSV_suffix) is True:
            #if os.path.isfile(args.CSV_data_dir + protein_id + '_good' + args.CSV_suffix) is True:
                chunk = pd.read_csv(args.CSV_data_dir + protein_id + args.CSV_suffix, index_col=False)
                #chunk = pd.read_csv(args.CSV_data_dir + protein_id + '_good' + args.CSV_suffix, index_col=False)
                
                if chunk.isnull().values.any():
                    test_list.remove(protein_id)
                    counterNaN_test+=1
                else:
                    #chunk['Label'] = 1
                    chunks.append(chunk)
                    del chunk

                    # if os.path.isfile(args.CSV_data_dir + protein_id + '_bad' + args.CSV_suffix) is True:
                    #     chunk = pd.read_csv(args.CSV_data_dir + protein_id + '_bad' + args.CSV_suffix, index_col=False)
                    # 
                    #     if chunk.isnull().values.any():
                    #         test_list.remove(protein_id)
                    #         counterNaN_test += 1
                    #     else:
                    #         chunk['Label'] = 0
                    #         chunks.append(chunk)
                    #         del chunk
            else:
                test_list.remove(protein_id)
                #print('| Could not find %s%s%-50s |' % (args.CSV_data_dir, protein_id, args.CSV_suffix))
        test_set = pd.concat(chunks).dropna()
        del chunks

    print('| Total number of PDBs: %-91s |' % (len(train_list) + len(val_list) + len(test_list)))
    print('|   Number of PDBs in the Train set:              %-8s (%.1f%%)                                                  |' % (len(train_list), (100 * len(train_list) / (len(train_list) + len(val_list) + len(test_list)) )))
    print('|   Number of PDBs in the Validation set:         %-8s (%.1f%%)                                                  |' % (len(val_list), (100 * len(test_list) / (len(train_list) + len(val_list) + len(test_list)) )))
    print('|   Number of PDBs in the Test set:               %-8s (%.1f%%)                                                  |' % (len(test_list), (100 * len(val_list) / (len(train_list) + len(val_list) + len(test_list)) )))
    print('|   Number of NaN CSV in the Train set:           %-7s                                                           |' % str(counterNaN_train))
    print('|   Number of NaN CSV in the Validation set:      %-7s                                                           |' % str(counterNaN_val))
    print('|   Number of NaN CSV in the Test set:            %-7s                                                           |' % str(counterNaN_test))
    print('| Total number of poses:                          %-7s                                                           |' % (len(train_set.index) + (len(val_set.index) + len(test_set.index))))
    print('|   Number of poses in the Train set:             %-8s (%.1f%%)                                                  |' % (len(train_set.index), (100 * len(train_set.index) / (len(train_set.index) + len(val_set.index) + len(test_set.index)) )))
    print('|   Number of poses in the Validation set:        %-8s (%.1f%%)                                                  |' % (len(val_set.index), (100 * len(val_set.index) / (len(train_set.index) + len(val_set.index) + len(test_set.index)) )))
    print('|   Number of poses in the Test set:              %-8s (%.1f%%)                                                  |' % (len(test_set.index), (100 * len(test_set.index) / (len(train_set.index) + len(val_set.index) + len(test_set.index)) )))
    print('|-------------------------------------------------------------------------------------------------------------------|')

# ----------- Randomize the data set
    print('| Randomizing the Training set, Validation set, and the Testing set                                                 |')
    print('|-------------------------------------------------------------------------------------------------------------------|')
    train_set = train_set.sample(frac=1)
    val_set = val_set.sample(frac=1)
    test_set = test_set.sample(frac=1)

# ----------- Split the data set into train_X train_y test_X test_y

    if args.features_train != "" :
        features_to_keep = args.features_train.split(',')
        features_to_drop = [feat for feat in train_set.columns if feat not in features_to_keep]
    else:
        features_to_drop = args.features.split(',')

    train_set_y = train_set['Label']
    val_set_y = val_set['Label']
    test_set_y = test_set['Label']
    train_set_X = train_set.drop(columns=features_to_drop, axis=1)
    val_set_X = val_set.drop(columns=features_to_drop, axis=1)
    test_set_X = test_set.drop(columns=features_to_drop, axis=1)
    features_X = train_set_X.columns

    # Print the list of dropped features
    print('| Dropping features:                                                                                                |')
    features_to_drop_buffered = list(features_to_drop)
    j = math.ceil(len(features_to_drop) / 5)
    while len(features_to_drop_buffered) < (j * 5):
        features_to_drop_buffered.append('')
    for i in range(0, j):
        print('| %-22s %-22s %-22s %-22s %-22s|' % (
        features_to_drop_buffered[j * 0 + i], features_to_drop_buffered[j * 1 + i], features_to_drop_buffered[j * 2 + i], features_to_drop_buffered[j * 3 + i],
        features_to_drop_buffered[j * 4 + i]))
    print('|-------------------------------------------------------------------------------------------------------------------|')

    # Print the list of remaining features for training
    print('| Using features for training:                                                                                      |')
    features_buffered = list(train_set_X.columns)  # this is an "Index" data type, which cannot be appended with strings
    j = math.ceil(len(features_buffered) / 5)
    while len(features_buffered) < (j * 5):
        features_buffered.append('')
    for i in range(0, j):
        print('| %-22s %-22s %-22s %-22s %-22s|' % (
        features_buffered[j * 0 + i], features_buffered[j * 1 + i], features_buffered[j * 2 + i], features_buffered[j * 3 + i], features_buffered[j * 4 + i]))
    print('|-------------------------------------------------------------------------------------------------------------------|')

# ----------- Normalize the data set

    if args.normalize == 'yes':
        train_set_X = pd.DataFrame(preprocessing.normalize(train_set_X))
        val_set_X = pd.DataFrame(preprocessing.normalize(val_set_X))
        test_set_X = pd.DataFrame(preprocessing.normalize(test_set_X))
        train_set_X.columns = features_X
        val_set_X.columns = features_X
        test_set_X.columns = features_X
        print('| Normalizing the dataset                                                                                           |')
        print('|-------------------------------------------------------------------------------------------------------------------|')
    elif args.normalize == 'no':
        print('| Dataset was not normalized                                                                                        |')
        print('|-------------------------------------------------------------------------------------------------------------------|')

    return train_set_X, train_set_y, val_set_X, val_set_y, test_set_X, test_set_y, features_X

    # else:
    #     print('| Could not find input train/val/test list or directory.                                                            |')
    #     print('| Please specify valid keywords (--CSV_data_dir or --list_valset or --list_trainset or --list_testset)              |')
    #     print('|-------------------------------------------------------------------------------------------------------------------|')
    # 
    #     return 0, 0, 0, 0, 0, 0, 0
# Splits the poses into correct (RMSD <= 2Angs) and incorrect (RMSD > 2 Angs)
def filter_set(args):
    # construct a random number generator - rng
    rng = np.random.default_rng(12345)
    protein_list = []
    # read the list of pdb id's

    # Use the splitting from https://arxiv.org/html/2308.09639v2#S5
    if os.path.isfile(args.CSV_data_dir + args.list_trainset) is True and os.path.isfile(args.CSV_data_dir + args.list_testset) is True:

        print('| Training set selected from %-86s |' % args.list_trainset)
        with open(args.CSV_data_dir + args.list_trainset, 'rb') as list_f:
            for line in list_f:
                protein_list.append(line.strip().decode('UTF-8'))
        print('| Testing set selected from %-86s  |' % args.list_testset)
        with open(args.CSV_data_dir + args.list_testset, 'rb') as list_f:
            for line in list_f:
                protein_list.append(line.strip().decode('UTF-8'))

    # Randomly split the pdb list into train and test, ensuring that no poses from a PDB are both in the train and the test
    else:

        # I have: 12697 = 10792 + 1905
        # I should have: 13508

        print('| Splitting the set randomly                                                                                        |')
        print('|   Train set: list_trainset_random.txt                                                                             |')
        print('|   Test set: list_testset_random.txt                                                                               |')
        protein_list = os.listdir(args.CSV_data_dir) # stores the name of the files in raw_data_dir
        protein_list = [x for x in protein_list if os.path.getsize(args.CSV_data_dir + '/' + x) > 0] # removes empty files
        protein_list = [x.replace('_poses_output-results_bitstring.csv', '') for x in protein_list
                        if '_poses_output-results_bitstring.csv' in x] # filters out files that don't contain '_poses_output-results_bitstring.csv' and removes that extension. Only the PDB id will remain

        # randomly split the set into 85% train and 15% test sets
        random.shuffle(protein_list)
        split_index = int(0.85 * len(protein_list))
        train_list = protein_list[:split_index]
        test_list = protein_list[split_index:]
        print('| Total number of PDBs: %-91s |' % (len(protein_list))) # print as a string with minimum 5 char (right aligned). If there are less than 5 char, then add white spaces to pad on the left
        print('|   Number of PDBs in the train set: %-78s |' % (len(train_list)))
        print('|   Number of PDBs in the test set: %-78s  |' % (len(test_list)))

        with open(args.CSV_data_dir + 'list_trainset_random.txt', 'w') as train_f:
            for line in train_list:
                train_f.write(f"{line}\n")

        with open(args.CSV_data_dir + 'list_testset_random.txt', 'w') as test_f:
            for line in test_list:
                test_f.write(f"{line}\n")


    protein_number = 0
    total_count_good = 0
    total_count_bad = 0
    for protein_id in protein_list:
        if protein_number == 0 and args.verbose > 2:
            print('|-------------------------------------------------------------------------------------------------------------------|')
            #print('|     # | Protein name                      | Min Energy      | Max Energy      | # Poses kept    | # Poses removed |')
            print('|     # | Protein name                      | Min Energy      | Max Energy      | # Poses kept    | # Poses removed | <1.0    | 1.0-2.0 | 2.0-3.0 | 3.0-4.0 | 4.0-5.0 | >5.0    |')
        protein_number += 1
        print('| %5s | %-33s | ' % (protein_number, protein_id), end='') # end with empty string (not a newline character)
        initial_conf = True
        reference_energy = 150
        sanitization = True
        # time.sleep(1)
        energies = []
        rmsd = []
#        # Combine the files into a single file starting with the minimized Xray pose (name: $pdb_all_poses.sdf).
#        # Remove those too high in energy and update the lowest energy observed so far
#        # if os.path.exists(args.raw_data_dir + protein_id + '_all_poses.sdf') is False:
#        #     print('1123: file ', args.raw_data_dir + protein_id + '_all_poses.sdf', ' does not exist')

        try:

            ligand = pd.read_csv(args.CSV_data_dir + protein_id + '_poses_output-results_bitstring.csv', index_col=False) # load the pdb in a dataframe
            number_total_poses = len(ligand.index)

            # remove a pose if there is a feature outside the quartiles
            numeric_ligand = ligand.select_dtypes(include=['number']) # remove string containing columns, only needed on mcurie
            df_cutoff = numeric_ligand.quantile([0.025, 0.975], axis=0)  # calculates the quantiles
            rows_to_keep = pd.Series([True] * len(ligand))

            #df_cutoff = ligand.quantile([0.025, 0.975], axis=0) # calculates the quantiles
            for feature in numeric_ligand.columns:

                if feature not in ('Name', 'Energy', 'RMSD', 'Label'): # remove values outside quantiles
                    top = df_cutoff.loc[0.975, feature]
                    bottom = df_cutoff.loc[0.025, feature]

                    # Keep rows that are between the values at the 2.5% and 97.5% quantiles
                    #ligand = ligand[ligand[feature].between(bottom, top)]
                    rows_to_keep &= ligand[feature].between(bottom, top)

            ligand = ligand[rows_to_keep]

            # Filter the energies
            ligand = ligand[(ligand['Energy'] > -150) & (ligand['Energy'] < 0)] # remove poses that are way too low or positive
            energy_min = ligand["Energy"].min() # get the new minium energy
            ligand = ligand[ligand['Energy'] < (float(energy_min) + args.energy_threshold)] # remove poses that have a higher energy that energy_min + energy_threshold
            energy_max = ligand["Energy"].max()  # get the maximum energy after filtering
            # number_filtered_poses = len(ligand.index)

            # # Select low RMSD and low/high energy poses
            # #ligand_RMSD_2 = ligand[ligand['RMSD'] < 2].sort_values(by='energy')
            # ligand_RMSD_2_low_E = ligand[(ligand['RMSD'] < 2) & (ligand['Energy'] < (energy_min + 10))]
            # ligand_RMSD_2_low_E = ligand_RMSD_2_low_E.sort_values(by='Energy')
            # ligand_RMSD_2_low_E = ligand_RMSD_2_low_E.head(10)
            # ligand_RMSD_2_low_E['Label'] = 1
            #
            # ligand_RMSD_2_high_E = ligand[(ligand['RMSD'] < 2) & (ligand['Energy'] > (energy_max - 10))]
            # ligand_RMSD_2_high_E = ligand_RMSD_2_high_E.sort_values(by='Energy')
            # ligand_RMSD_2_high_E = ligand_RMSD_2_high_E.tail(10)
            # ligand_RMSD_2_high_E['Label'] = 0
            #
            # ligand_filtered = pd.concat([ligand_RMSD_2_low_E, ligand_RMSD_2_high_E], ignore_index=True)
            # number_filtered_poses = len(ligand_filtered.index)

            # Select low Energy but low/high RMSD poses
            # ligand_high_RMSD = ligand[(ligand['RMSD'] > 2.5) & (ligand['Energy'] < (energy_min + 15))]
            # ligand_high_RMSD = ligand_high_RMSD.sort_values(by='Energy')
            # ligand_high_RMSD = ligand_high_RMSD.head(5)
            # 
            # ligand_low_RMSD = ligand[(ligand['RMSD'] < 1.75) & (ligand['Energy'] < (energy_min + 15))]
            # ligand_low_RMSD = ligand_low_RMSD.sort_values(by='Energy')
            # ligand_low_RMSD = ligand_low_RMSD.head(len(ligand_high_RMSD.index)) #same number of poses as high rmsd/low energy to have a balanced set
            # 
            # ligand_filtered = pd.concat([ligand_low_RMSD, ligand_high_RMSD], ignore_index=True)
            # number_filtered_poses = len(ligand_filtered.index)
            # 
            # # print('%-15s | ' % (energy_min), end='')
            # # print('%-15s | ' % (energy_max), end='')
            # print('%-15s | ' % (len(ligand_low_RMSD.index)), end='')
            # print('%-15s | ' % (len(ligand_high_RMSD.index)), end='')
            # print('%-15s | ' % (number_filtered_poses), end='')
            # #print('%-15s | ' % (number_total_poses - number_filtered_poses))
            # print('%-15s | ' % (number_total_poses - number_filtered_poses), end='')
            # 
            # RMSD_0to1 = (ligand['RMSD'] <= 1).sum()
            # RMSD_1to2 = ((ligand['RMSD'] > 1) & (ligand['RMSD'] <= 2)).sum()
            # RMSD_2to3 = ((ligand['RMSD'] > 2) & (ligand['RMSD'] <= 3)).sum()
            # RMSD_3to4 = ((ligand['RMSD'] > 3) & (ligand['RMSD'] <= 4)).sum()
            # RMSD_4to5 = ((ligand['RMSD'] > 4) & (ligand['RMSD'] <= 5)).sum()
            # RMSD_above5 = (ligand['RMSD'] > 5).sum()
            # 
            # print('%-7s | ' % (RMSD_0to1), end='')
            # print('%-7s | ' % (RMSD_1to2), end='')
            # print('%-7s | ' % (RMSD_2to3), end='')
            # print('%-7s | ' % (RMSD_3to4), end='')
            # print('%-7s | ' % (RMSD_4to5), end='')
            # print('%-7s | ' % (RMSD_above5))

            ligand.to_csv(args.CSV_data_dir + protein_id + '_filtered_RMSD_25.csv', index=False)
            #ligand_filtered.to_csv(args.CSV_data_dir + protein_id + '_filtered_RMSD_25.csv', index=False)

        except:
            print('problem reading ' + protein_id + '_poses_output-results_bitstring.csv               |')
            continue
def pair_good_bad_poses_substract(args):

    protein_list = []
    protein_list_train = []
    protein_list_test = []
    # read the list of pdb id's

    # Use the splitting from https://arxiv.org/html/2308.09639v2#S5
    if os.path.isfile(args.CSV_data_dir + args.list_trainset) is True and os.path.isfile(args.CSV_data_dir + args.list_testset) is True:

        print('| Training set selected from %-86s |' % args.list_trainset)
        with open(args.CSV_data_dir + args.list_trainset, 'rb') as list_f:
            for line in list_f:
                protein_list.append(line.strip().decode('UTF-8'))
                protein_list_train.append(line.strip().decode('UTF-8'))
        print('| Testing set selected from %-86s  |' % args.list_testset)
        with open(args.CSV_data_dir + args.list_testset, 'rb') as list_f:
            for line in list_f:
                protein_list.append(line.strip().decode('UTF-8'))
                protein_list_test.append(line.strip().decode('UTF-8'))

    # Randomly split the pdb list into train and test, ensuring that no poses from a PDB are both in the train and the test
    else:

        # I have: 12697 = 10792 + 1905
        # I should have: 13508

        print('| Splitting the set randomly                                                                                        |')
        print('|   Train set: list_trainset_random.txt                                                                             |')
        print('|   Test set: list_testset_random.txt                                                                               |')
        protein_list = os.listdir(args.CSV_data_dir)  # stores the name of the files in raw_data_dir
        protein_list = [x for x in protein_list if os.path.getsize(args.CSV_data_dir + '/' + x) > 0]  # removes empty files
        protein_list = [x.replace('_good' + args.CSV_suffix, '') for x in protein_list
                        if
                        '_good' + args.CSV_suffix in x]  # filters out files that don't contain '_poses_output-results_bitstring.csv' and removes that extension. Only the PDB id will remain

        # randomly split the set into 85% train and 15% test sets
        random.shuffle(protein_list)
        split_index = int(0.85 * len(protein_list))
        train_list = protein_list[:split_index]
        test_list = protein_list[split_index:]
        print('| Total number of PDBs: %-91s |' % (len(protein_list)))  # print as a string with minimum 5 char (right aligned). If there are less than 5 char, then add white spaces to pad on the left
        print('|   Number of PDBs in the train set: %-78s |' % (len(train_list)))
        print('|   Number of PDBs in the test set: %-78s  |' % (len(test_list)))

        with open(args.CSV_data_dir + 'list_trainset_random.txt', 'w') as train_f:
            for line in train_list:
                train_f.write(f"{line}\n")

        with open(args.CSV_data_dir + 'list_testset_random.txt', 'w') as test_f:
            for line in test_list:
                test_f.write(f"{line}\n")
    eng = pd.DataFrame()
    protein_number = 0
    protein_good_bad = 0
    protein_not_empty = 0
    protein_25 = 0
    protein_final = 0
    above = []
    below = []
##    label_1 = 0
##    label_0 = 0
    for protein_id in protein_list:
        if protein_number == 0 and args.verbose > 2:
            print('|-------------------------------------------------------------------------------------------------------------------|')
            print('|     # | Protein name                      | Min Energy      | Max Energy      | # Good poses     | # Bad poses    |')
            #print('|     # | Protein name            | Min Energy    | Corr. RMSD    | Min RMSD      | Corr. Energy  | Match/Mismatch? |')
        protein_number += 1
        print('| %5s | %-23s | ' % (protein_number, protein_id), end='') # end with empty string (not a newline character)
        initial_conf = True
        reference_energy = 150
        sanitization = True
        # time.sleep(1)
        energies = []
        rmsd = []
#        # Combine the files into a single file starting with the minimized Xray pose (name: $pdb_all_poses.sdf).
#        # Remove those too high in energy and update the lowest energy observed so far
#        # if os.path.exists(args.raw_data_dir + protein_id + '_all_poses.sdf') is False:
#        #     print('1123: file ', args.raw_data_dir + protein_id + '_all_poses.sdf', ' does not exist')




        if os.path.isfile(args.CSV_data_dir + protein_id + '_good' + args.CSV_suffix) is True and os.path.isfile(args.CSV_data_dir + protein_id + '_bad' + args.CSV_suffix) is True:
            protein_good_bad += 1
            ligand_good_poses = pd.read_csv(args.CSV_data_dir + protein_id + '_good' + args.CSV_suffix, index_col=False)  # load the pdb in a dataframe
            ligand_bad_poses = pd.read_csv(args.CSV_data_dir + protein_id + '_bad' + args.CSV_suffix, index_col=False)
            with open(args.CSV_data_dir + protein_id + '_good' + args.CSV_suffix, 'rb') as file:
                content = file.read()
                if b'\x00' in content:
                    print(f"Null character found in {protein_id}")
                    continue
                else:
                    try:
                        ligand_good_poses = pd.read_csv(args.CSV_data_dir + protein_id + '_good' + args.CSV_suffix, index_col=False)
                    except:
                        print("An unknown error occured with this file")
                        continue

            with open(args.CSV_data_dir + protein_id + '_bad' + args.CSV_suffix, 'rb') as file:
                content = file.read()
                if b'\x00' in content:
                    print(f"Null character found in {protein_id}")
                    continue
                else:
                    try:
                        ligand_bad_poses = pd.read_csv(args.CSV_data_dir + protein_id + '_bad' + args.CSV_suffix, index_col=False)
                    except:
                        print("An unknown error occured with this file")
                        continue

            ligand_good_poses['RMSD'] = 1.75
            ligand_bad_poses['RMSD'] = 2.5
            ligand_good_poses = ligand_good_poses.dropna()
            ligand_bad_poses = ligand_bad_poses.dropna()
    ##            number_total_poses = len(ligand.index)

            if (len(ligand_good_poses.index) > 0):
                if (len(ligand_bad_poses.index) > 0):

                    protein_not_empty += 1

                    #eng = pd.concat([eng, ligand_good_poses[['Name', 'Energy']]], axis=0)
                    #eng = pd.concat([eng, ligand_bad_poses[['Name', 'Energy']]], axis=0)

                    ligand_good_poses = ligand_good_poses.sample(frac=1)
                    ligand_bad_poses = ligand_bad_poses.sample(frac=1)
                    ligand_good_poses['Name'] = 0
                    ligand_bad_poses['Name'] = 0

                    if len(ligand_good_poses.index) == len(ligand_bad_poses.index):
                        substraction_gb = ligand_good_poses.subtract(ligand_bad_poses).round(3)
                        substraction_gb['Label'] = 0
                        substraction_bg = ligand_bad_poses.subtract(ligand_good_poses).round(3)
                        substraction_bg['Label'] = 1

                        eng = pd.concat([eng, substraction_gb['Energy']], axis=0)
                        #eng = pd.concat([eng, substraction_bg['Energy']], axis=0)

                        substraction = pd.concat([substraction_gb.iloc[:12], substraction_bg.iloc[13:]], ignore_index=True)
                        substraction = substraction.sample(frac=1)
                        print(' ')
                        protein_final += 1

                        if (substraction_gb['Energy'] > 100).any():
                            print('Above 100:    ' + protein_id)
                            above.append(protein_id)
                        if (substraction_gb['Energy'] < -100).any():
                            print('Below -100:   ' + protein_id)
                            below.append(protein_id)
                    else:
                        print(' Good and bad poses do not have the same number of lines                                         |')

                    #substraction.to_csv(args.CSV_data_dir + protein_id + '_good_bad_substract_filtered_' + args.CSV_suffix2, index=False)
                    substraction.to_csv(args.CSV_data_dir + protein_id + '_good_bad_substract.csv', index=False)


                else:
                    print(protein_id + '_bad_poses.csv is empty                                                       |')
            else:
                print(protein_id + '_good_poses.csv is empty                                                      |')

        else:
            print(protein_id + args.CSV_suffix + ' does not exist                                                 |')


    # eng['Energy'] = eng['Energy'].clip(lower=-100, upper=100)
    # plt.figure(figsize=(10, 6))
    # plt.hist(eng['Energy'], bins=200)
    # plt.savefig('hist_before_subs.png')
    # plt.clf()

    print('proteins read:     ' + str(protein_number))
    print('files found:       ' + str(protein_good_bad))
    print('files with data:   ' + str(protein_not_empty))
    print('total written:     ' + str(protein_final))

    print(above)
    print(below)

    # percent_train = len(mismatch_train) / mismatch_counter * 100
    # percent_test = len(mismatch_test) / mismatch_counter * 100
    # percent_1 = label_1 / (label_0 + label_1) * 100
    # percent_0 = label_0 / (label_0 + label_1) * 100

    # print('|-------------------------------------------------------------------------------------------------------------------|')
    # print('| Number of mismatched pairs: %-85s |' % str(mismatch_counter) )
    # print('| Number of mismatched pairs in the train set: %5s (%.0f%s)                                                          |' % ( str(len(mismatch_train)) , percent_train, "%"))
    # print('| Number of mismatched pairs in the train set: %5s (%.0f%s)                                                          |' % ( str(len(mismatch_test)) , percent_test, "%"))
    # print('| Number of "1" labels: %5s (%.0f%s)                                                                                 |' % ( str(label_1), percent_1, "%"))
    # print('| Number of "0" labels: %5s (%.0f%s)                                                                                 |' % ( str(label_0), percent_0 , "%"))

    # with open(args.CSV_data_dir + 'train.txt', 'w') as f:
    #     for line in mismatch_train:
    #         f.write(f"{line}\n")

    # with open(args.CSV_data_dir + 'test.txt', 'w') as f:
    #     for line in mismatch_test:
    #         f.write(f"{line}\n")



            
def make_pairs_from_top_10(args):

    train_list = []
    #test_list = []

    # Use the splitting from https://arxiv.org/html/2308.09639v2#S5
    if os.path.isfile(args.CSV_data_dir + args.list_trainset) is True: # and os.path.isfile(args.CSV_data_dir + args.list_testset) is True:

        print('| Training set selected from %-86s |' % args.list_trainset)
        with open(args.CSV_data_dir + args.list_trainset, 'rb') as list_f:
            for line in list_f:
                train_list.append(line.strip().decode('UTF-8'))
                #print(line.decode('UTF-8') + ' loaded')
        #print('| Testing set selected from %-86s  |' % args.list_testset)
        #with open(args.CSV_data_dir + args.list_testset, 'rb') as list_f:
        #    for line in list_f:
        #        test_list.append(line.strip().decode('UTF-8'))
    else:
        print('| Please provide a train and test list. |')

    if args.model == 'XGBc':
        model = xgb.XGBClassifier()
        model.load_model(args.model_name + '.json')
        #feature_names = model.get_booster().feature_names
    elif args.model == 'LRc':
        with open(args.model_name + '.pkl', 'rb') as pickle_file:
            model = pickle.load(pickle_file)
    feature_names = ['Energy','MatchScore','GLY_BB_vdW','GLY_BB_ES','GLY_BB_HB','ALA_BB_vdW','ALA_BB_ES','ALA_BB_HB','ALA_SC_vdW','ALA_SC_ES','VAL_BB_vdW','VAL_BB_ES','VAL_BB_HB','VAL_SC_vdW','VAL_SC_ES','LEU_BB_vdW','LEU_BB_ES','LEU_BB_HB','LEU_SC_vdW','LEU_SC_ES','ILE_BB_vdW','ILE_BB_ES','ILE_BB_HB','ILE_SC_vdW','ILE_SC_ES','PRO_BB_vdW','PRO_BB_ES','PRO_BB_HB','PRO_SC_vdW','PRO_SC_ES','ASP_BB_vdW','ASP_BB_ES','ASP_BB_HB','ASP_SC_vdW','ASP_SC_ES','ASP_SC_HB','GLU_BB_vdW','GLU_BB_ES','GLU_BB_HB','GLU_SC_vdW','GLU_SC_ES','GLU_SC_HB','ASN_BB_vdW','ASN_BB_ES','ASN_BB_HB','ASN_SC_vdW','ASN_SC_ES','ASN_SC_HB','GLN_BB_vdW','GLN_BB_ES','GLN_BB_HB','GLN_SC_vdW','GLN_SC_ES','GLN_SC_HB','LYS_BB_vdW','LYS_BB_ES','LYS_BB_HB','LYS_SC_vdW','LYS_SC_ES','LYS_SC_HB','ARG_BB_vdW','ARG_BB_ES','ARG_BB_HB','ARG_SC_vdW','ARG_SC_ES','ARG_SC_HB','TRP_BB_vdW','TRP_BB_ES','TRP_BB_HB','TRP_SC_vdW','TRP_SC_ES','TYR_BB_vdW','TYR_BB_ES','TYR_BB_HB','TYR_SC_vdW','TYR_SC_ES','TYR_SC_HB','PHE_BB_vdW','PHE_BB_ES','PHE_BB_HB','PHE_SC_vdW','PHE_SC_ES','HIS_BB_vdW','HIS_BB_ES','HIS_BB_HB','HIS_SC_vdW','HIS_SC_ES','HIS_SC_HB','CYS_BB_vdW','CYS_BB_ES','CYS_BB_HB','CYS_SC_vdW','CYS_SC_ES','CYS_SC_HB','SER_BB_vdW','SER_BB_ES','SER_BB_HB','SER_SC_vdW','SER_SC_ES','SER_SC_HB','THR_BB_vdW','THR_BB_ES','THR_BB_HB','THR_SC_vdW','THR_SC_ES','THR_SC_HB','MET_BB_vdW','MET_BB_ES','MET_BB_HB','MET_SC_vdW','MET_SC_ES','MET_SC_HB','Bonds','Angles','Torsions','Angles_M','Tors_M','vdW_14','Elec_14','vdW_15','Elec_15','vdW','Elec','HBonds','HB_M','Bond_M','Wat_vdW','Wat_Elec','Wat_HB','No_Wat']

    total = 0
    accuracies = pd.DataFrame(columns=['RMSD threshold', 'Lowest energy', 'Lowest RMSD', 'Random', 'Predicted'])
    accuracies['RMSD threshold'] = np.arange(0, 5.25, 0.25)
    accuracies['Lowest energy'] = 0
    accuracies['Lowest RMSD'] = 0
    accuracies['Random'] = 0
    accuracies['Predicted'] = 0

    for protein_id in train_list:

        if os.path.isfile(args.CSV_data_dir + protein_id + args.CSV_suffix) and (os.stat(args.CSV_data_dir + protein_id + args.CSV_suffix).st_size != 0):
            ligand = pd.read_csv(args.CSV_data_dir + protein_id + args.CSV_suffix, index_col=False)  # load the pdb in a dataframe
            #print(ligand)
            ligand = ligand.dropna()

            if len(ligand.index) == 10:

                total = total + 1
                # PDBdata = pd.DataFrame(columns=['Energies', 'Label', 'RMSD', 'Label_ML', 'RMSD_ML'])
                # PDBdata['Energies'] = pdb['Energy']
                # PDBdata['Label'] = pdb['Label']
                # PDBdata['RMSD'] = pdb['RMSD']

                #ligand = ligand.sample(frac=1)
                ligand['Name'] = 0 # need to set it to 0 otherwise you cant substract

                predicted_label_matrix = np.zeros([10, 10]) # substraction matrix
                label_matrix = np.zeros([10, 10])
                #            j    j    j            ---> x-axis
                # matrix = [[A], [B], [C],  --> i   ---> y-axis
                #           [D], [E], [F],  --> i
                #           [G], [H], [I]]  --> i

                for i, row in enumerate(predicted_label_matrix): # go through each row i eg. for i=1 --> [A,B,C]
                    for j, value in enumerate(row): # go through each value j in a row i eg. for i=1 --> A,B,C
                        #sub = (ligand.iloc[j] - ligand.iloc[i]).round(3)
                        # generating the good/bad pairs of subtracted features
                        # eg. for i=1 --> A-A, A-B, A-C
                        # the result of the subtraction of features, eg. run 1 - run 1,...,10 (column - row), run 2 - run 1,...,10
                        sub = (ligand.iloc[i] - ligand.iloc[j]).round(3)
                        # if the RMSD difference is positive, this means that we substracted a bad (ie. high RMSD) - good pair (ie. low RMSD) --> Label = 1
                        if sub['RMSD'] > 0: # eg. RMSD 3.0 (inactive) - RMSD 0.2 (active) = +2.8 --> Label = 1 (inactive - active)
                            sub['Label'] = 1
                        elif sub['RMSD'] < 0: # eg. RMSD 0.2 (active) - RMSD 3.0 (inactive) = -2.8 --> Label = 0 (active - inactive)
                            sub['Label'] = 0
                        elif sub['RMSD'] == 0:
                            ## not sure
                            sub['Label'] = 1
                        # split the subtracted pairs into the model input (X) and output (y)
                        sub_X = sub[feature_names].to_frame().transpose() # substracted features for 1 good/bad pair
                        sub_y = sub['Label'] # labels
                        # predict if the pair is a Good - Bad or a Bad - Good
                        predicted_label_matrix[i][j] = model.predict(sub_X)
                        predicted_label_matrix[i][i] = 1
                        label_matrix[i][j] = sub_y
                        #prediction = model.predict(sub_X)
                #print(protein_id)
                #print(label_matrix) # real matrix
                #print(predicted_label_matrix) # prediction matrix
                #print(np.sum(label_matrix, axis=0)) #--> sums the columns
                # print(np.sum(label_matrix, axis=1)) #--> sums the rows
                # print("Column with the highest sum:", np.argmax(np.sum(label_matrix, axis=0)))
                #print(predicted_label_matrix)
                column_sums = np.sum(predicted_label_matrix, axis=0) # sums the columns --> the most number of 1s ie. the highest sum is the most active (column# starts at 0)
                #row_sums = np.sum(predicted_label_matrix, axis=1) # sums the rows --> the least number of 1s ie. the most inactives
                #print(column_sums)
                #print(row_sums)
                max_sum_column_index = np.argmax(column_sums)
                #print("Column with the highest sum:", max_sum_column_index)

                max_value = np.max(column_sums) # Find the maximum value
                max_indices = np.where(column_sums == max_value)[0] # Get all indices where the sum is equal to the max value
                random_max_index = np.random.choice(max_indices) # Randomly select one of the indices
                #print("Random column with the highest sum:", random_max_index)

                ligand['RMSD_ML'] = 3
                ligand.at[random_max_index, 'RMSD_ML'] = 0
                
                
                # ligand.loc[max_indices, 'RMSD_ML'] = 0
                # if len(max_indices) > 1: # if there are two RMSD_ML predictions that are possible, select the one with the lowest energy
                #     max_index_with_lowest_energy = ligand.loc[ligand['RMSD_ML'] == 0, 'Energy'].idxmin()
                #     ligand['RMSD_ML'] = 3
                #     ligand.at[max_index_with_lowest_energy, 'RMSD_ML'] = 0
                #print(ligand['RMSD_ML'])

                random = ligand.sample()
                for index, value in accuracies['RMSD threshold'].items():

                    min_energy = ligand['Energy'].idxmin()

                    # Lowest energy
                    if ligand.loc[min_energy, 'RMSD'] <= value:
                        accuracies.at[index, 'Lowest energy'] += 1

                    # Lowest RMSD
                    if ligand['RMSD'].min() <= value:
                        accuracies.at[index, 'Lowest RMSD'] += 1

                    # Random
                    if random['RMSD'].iloc[0] <= value:
                        # accuracies['Random'] = accuracies['Random'] + 1
                        accuracies.at[index, 'Random'] += 1

                    # Predicted
                    if ligand.loc[ligand['RMSD_ML'].idxmin(), 'RMSD'] <= value:
                        accuracies.at[index, 'Predicted'] += 1
            else:
                #print(len(ligand.index))
                print(protein_id + ' is empty                                                                   |')
    
    print('Total number of PDBs: ' + str(total))
    print('Accuracies raw:')
    print(accuracies)

    accuracies['Lowest energy'] = (accuracies['Lowest energy'] / total) * 100
    accuracies['Lowest RMSD'] = (accuracies['Lowest RMSD'] / total) * 100
    accuracies['Random'] = (accuracies['Random'] / total) * 100
    accuracies['Predicted'] = (accuracies['Predicted'] / total) * 100

    # print('Accuracies %:')
    # print(accuracies)

    plt.plot(accuracies['RMSD threshold'], accuracies['Lowest energy'], label="Lowest energy", color='blue')
    plt.plot(accuracies['RMSD threshold'], accuracies['Lowest RMSD'], label="Lowest RMSD", color='red')
    plt.plot(accuracies['RMSD threshold'], accuracies['Random'], label="Random", color='orange')
    plt.plot(accuracies['RMSD threshold'], accuracies['Predicted'], label="Predicted", color='green')

    ax = plt.gca()

    font1 = {'color': 'black', 'size': 14}
    font2 = {'color': 'black', 'size': 12}

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    # Set the x axis label of the current axis.
    plt.xlabel('RMSD (Å)', fontdict=font1)
    # Set the y axis label of the current axis.
    plt.ylabel('Accuracy (%)', fontdict=font1)
    # show a legend on the plot
    plt.legend(prop={'size': 12}, loc='lower right')
    # show gridlines
    plt.grid()
    # Set a title of the current axes.
    plt.title(args.model, fontdict=font1)
    # Add extra white space at the bottom so that the label is not cut off
    plt.subplots_adjust(bottom=0.13)

    print('Accuracies %:')
    print(accuracies)
    plt.savefig('Accuracy_pred_' + args.model + '.png')

def select_mismatch_pairs(args):

    protein_list = []
    protein_list_train = []
    protein_list_test = []
    mismatch_train = []
    mismatch_test = []
    # read the list of pdb id's

    # Use the splitting from https://arxiv.org/html/2308.09639v2#S5
    if os.path.isfile(args.CSV_data_dir + args.list_trainset) is True and os.path.isfile(args.CSV_data_dir + args.list_testset) is True:

        print('| Training set selected from %-86s |' % args.list_trainset)
        with open(args.CSV_data_dir + args.list_trainset, 'rb') as list_f:
            for line in list_f:
                protein_list.append(line.strip().decode('UTF-8'))
                protein_list_train.append(line.strip().decode('UTF-8'))
        print('| Testing set selected from %-86s  |' % args.list_testset)
        with open(args.CSV_data_dir + args.list_testset, 'rb') as list_f:
            for line in list_f:
                protein_list.append(line.strip().decode('UTF-8'))
                protein_list_test.append(line.strip().decode('UTF-8'))

    # Randomly split the pdb list into train and test, ensuring that no poses from a PDB are both in the train and the test
    else:

        # I have: 12697 = 10792 + 1905
        # I should have: 13508

        print('| Splitting the set randomly                                                                                        |')
        print('|   Train set: list_trainset_random.txt                                                                             |')
        print('|   Test set: list_testset_random.txt                                                                               |')
        protein_list = os.listdir(args.CSV_data_dir)  # stores the name of the files in raw_data_dir
        protein_list = [x for x in protein_list if os.path.getsize(args.CSV_data_dir + '/' + x) > 0]  # removes empty files
        protein_list = [x.replace('_top10.csv', '') for x in protein_list
                        if
                        '_top10.csv' in x]  # filters out files that don't contain '_poses_output-results_bitstring.csv' and removes that extension. Only the PDB id will remain

        # randomly split the set into 85% train and 15% test sets
        random.shuffle(protein_list)
        split_index = int(0.85 * len(protein_list))
        train_list = protein_list[:split_index]
        test_list = protein_list[split_index:]
        print('| Total number of PDBs: %-91s |' % (len(protein_list)))  # print as a string with minimum 5 char (right aligned). If there are less than 5 char, then add white spaces to pad on the left
        print('|   Number of PDBs in the train set: %-78s |' % (len(train_list)))
        print('|   Number of PDBs in the test set: %-78s  |' % (len(test_list)))

        with open(args.CSV_data_dir + 'list_trainset_random.txt', 'w') as train_f:
            for line in train_list:
                train_f.write(f"{line}\n")

        with open(args.CSV_data_dir + 'list_testset_random.txt', 'w') as test_f:
            for line in test_list:
                test_f.write(f"{line}\n")

    protein_number = 0
    mismatch_counter = 0
    label_1 = 0
    label_0 = 0
    for protein_id in protein_list:
        if protein_number == 0 and args.verbose > 2:
            print('|-------------------------------------------------------------------------------------------------------------------|')
            #print('|     # | Protein name                      | Min Energy      | Max Energy      | # Poses kept    | # Poses removed |')
            print('|     # | Protein name            | Min Energy    | Corr. RMSD    | Min RMSD      | Corr. Energy  | Match/Mismatch? |')
        protein_number += 1
        print('| %5s | %-23s | ' % (protein_number, protein_id), end='') # end with empty string (not a newline character)
        initial_conf = True
        reference_energy = 150
        sanitization = True
        # time.sleep(1)
        energies = []
        rmsd = []
#        # Combine the files into a single file starting with the minimized Xray pose (name: $pdb_all_poses.sdf).
#        # Remove those too high in energy and update the lowest energy observed so far
#        # if os.path.exists(args.raw_data_dir + protein_id + '_all_poses.sdf') is False:
#        #     print('1123: file ', args.raw_data_dir + protein_id + '_all_poses.sdf', ' does not exist')

        try:

            ligand = pd.read_csv(args.CSV_data_dir + protein_id + '_top10.csv', index_col=False)  # load the pdb in a dataframe
            ligand = ligand.dropna()
            number_total_poses = len(ligand.index)

            if len(ligand.index) > 0:

                # get the lowest Energy and RMSD
                min_energy = ligand['Energy'].min()
                min_energy_idx = ligand['Energy'].idxmin()
                min_RMSD = ligand['RMSD'].min()
                min_RMSD_idx = ligand['RMSD'].idxmin()

                print('%-13s | ' % min_energy, end='') # Min Energy
                print('%-13s | ' % ligand.at[min_energy_idx, 'RMSD'], end='') # Corresponding RMSD
                print('%-13s | ' % min_RMSD, end='')  # Min RMSD
                print('%-13s | ' % ligand.at[min_RMSD_idx, 'Energy'], end='')  # Corresponding Energy

                # # Does the minimum Energy correspond to the minimum RMSD and,
                # # if not, is the minimum RMSD below 2
                # if min_energy_idx != min_RMSD_idx :
                #     if min_RMSD < 2 :
                #         # write these poses as mismatched pair
                #         ligand.at[min_energy_idx, 'Label'] = 0
                #         ligand.at[min_RMSD_idx, 'Label'] = 1

                # Does the minimum Energy correspond to an RMSD value below 2 and,
                # if not, is there and RMSD value below 2
                if (ligand.at[min_energy_idx, 'RMSD'] > 2) & ((ligand['RMSD'] < 2).any()) :
                        # write these poses as mismatched pair
                        ligand.at[min_energy_idx, 'Label'] = 0
                        ligand.at[min_RMSD_idx, 'Label'] = 1

                        mismatch_counter += 1
                        print('MISMATCH        |')

                        # Remove all other lines
                        indexes_to_keep = [min_RMSD_idx, min_energy_idx]
                        mismatch = ligand[ligand.index.isin(indexes_to_keep)]

                        if protein_id in protein_list_train:
                            mismatch_train.append(protein_id)
                        elif protein_id in protein_list_test:
                            mismatch_test.append(protein_id)

                        # write the mismatched pair
                        if args.mismatch == 'pair':
                            mismatch.to_csv(args.CSV_data_dir + protein_id + '_mismatch.csv', index=False)

                        elif args.mismatch == 'substract':

                            # Get the average difference between actives and inactives

                            # Scramble
                            mismatch['Name'] = 0
                            #mismatch = mismatch.assign(Name=0) # Required because you can't substract strings


                            mismatch.sample(frac=1)
                            #mismatch_substract = subtract_rows(mismatch.iloc[0], mismatch.iloc[1])
                            mismatch_substract = (mismatch.iloc[0] - mismatch.iloc[1]).round(3) # row 1 - row 2
                            #print(mismatch_substract)
                            if (mismatch['Label'].iloc[0] == 1) & (mismatch['Label'].iloc[1] == 0): # if row 1 was the active, then the Label is 1
                                mismatch_substract['Label'] = 1
                                label_1 += 1
                            elif (mismatch['Label'].iloc[0] == 0) & (mismatch['Label'].iloc[1] == 1): # if row 2 was the active, then the Label is 0
                                mismatch_substract['Label'] = 0
                                label_0 += 1

                            ms_df = mismatch_substract.to_frame().transpose()
                            ms_df.to_csv(args.CSV_data_dir + protein_id + '_mismatch_substract.csv', index=False)

                        elif args.mismatch == 'add':
                            if (mismatch['Label'].iloc[0] == 1) & (mismatch['Label'].iloc[1] == 0):

                                mismatch.to_csv(args.CSV_data_dir + protein_id + '_mismatch_add.csv', index=False)

                else:
                    print('match           |')
            else:
                print(protein_id + ' is empty                                                                   |')

        except:
            print('problem reading ' + protein_id + '_top10.csv                                                  |')
            continue

    percent_train = len(mismatch_train) / mismatch_counter * 100
    percent_test = len(mismatch_test) / mismatch_counter * 100
    percent_1 = label_1 / (label_0 + label_1) * 100
    percent_0 = label_0 / (label_0 + label_1) * 100

    print('|-------------------------------------------------------------------------------------------------------------------|')
    print('| Number of mismatched pairs: %-85s |' % str(mismatch_counter) )
    print('| Number of mismatched pairs in the train set: %5s (%.0f%s)                                                          |' % ( str(len(mismatch_train)) , percent_train, "%"))
    print('| Number of mismatched pairs in the train set: %5s (%.0f%s)                                                          |' % ( str(len(mismatch_test)) , percent_test, "%"))
    print('| Number of "1" labels: %5s (%.0f%s)                                                                                 |' % ( str(label_1), percent_1, "%"))
    print('| Number of "0" labels: %5s (%.0f%s)                                                                                 |' % ( str(label_0), percent_0 , "%"))

    with open(args.CSV_data_dir + 'train.txt', 'w') as f:
        for line in mismatch_train:
            f.write(f"{line}\n")

    with open(args.CSV_data_dir + 'test.txt', 'w') as f:
        for line in mismatch_test:
            f.write(f"{line}\n")
