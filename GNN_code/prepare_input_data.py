# Author: Nic Moitessier
# McGill University, Montreal, QC, Canada
# pip install torch
#pip install torch_geometric
import random
import rdkit
import os
import pickle
import torch
import pandas as pd
import numpy as np
import shutil
import time

from rdkit import Chem
from rdkit.Chem import rdmolops
from scipy.spatial import distance_matrix

from featurizer import get_mol_charge, get_mol_nrot, get_mol_natoms, get_node_features, get_edge_features, get_edge_features_pairs, get_labels, \
                       get_node_features_from_graphs, get_edge_features_pairs_from_graphs
from utils import split_list, add_formal_charges, scramble_atom_list, scramble_atom_list_pairs, get_Forecaster_graph


# Prepare the complex protein-ligand and dump as pickled files.
# TODO: what about tautomers (same name would likely overwrite...)
# TODO: what about waters (especially those displaced)?
# TODO: labels atoms interacting (to be kept) and those displacing water (to be kept)
# TODO: Remove atoms not interacting or within 2 bonds from atoms interacting.

def generate_complexes(args):
    protein_list = []
    train_protein_list = []
    test_protein_list = []
    val_protein_list = []
    protein_list = []

    # read the list of pdb id's or simply take all the files in the raw_data folder.
    if os.path.isfile(args.raw_data_dir + args.list_trainset) is True and os.path.isfile(args.raw_data_dir + args.list_testset) is True:
        print('| Training set selected from %86s |' % args.list_trainset)
        with open(args.raw_data_dir + args.list_trainset, 'rb') as list_f:
            for line in list_f:
                train_protein_list.append(line.strip().decode('UTF-8'))
        print('| Testing set selected from %86s  |' % args.list_trainset)
        with open(args.raw_data_dir + args.list_testset, 'rb') as list_f:
            for line in list_f:
                test_protein_list.append(line.strip().decode('UTF-8'))
        if args.list_valset != 'none':
            with open(args.raw_data_dir + args.list_valset, 'rb') as list_f:
                for line in list_f:
                    val_protein_list.append(line.strip().decode('UTF-8'))

    else:
        print('| training and testing sets selected randomly                                                                       |')
        protein_list = os.listdir(args.raw_data_dir)
        protein_list = [x for x in protein_list if os.path.getsize(args.raw_data_dir + '/' + x) > 0]
        protein_list = [x.replace('_good_poses.sdf', '') for x in protein_list if '_good_poses.sdf' in x]

        # randomize the list of proteins then split into train and test
        random.shuffle(protein_list)
        train_protein_list, test_protein_list = split_list(protein_list, args.split_train_test)

    print('| Proteins in the training set: %5.0f %77s |' % (len(train_protein_list), ' '))
    print('|------------------------- - - -      %77s |' % ' ')
    for i in range(len(train_protein_list)):
        print('| %-113s |' % train_protein_list[i])

    print('|-------------------------------      %77s |' % ' ')
    print('| Proteins in the testing set:  %5.0f %77s |' % (len(test_protein_list), ' '))
    print('|------------------------- - - -      %77s |' % ' ')
    for i in range(len(test_protein_list)):
        print('| %-113s |' % test_protein_list[i])
    if args.list_valset != 'none':
        print('|-------------------------------      %77s |' % ' ')
        print('| Proteins in the validation set:  %5.0f %74s |' % (len(val_protein_list), ' '))
        print('|------------------------- - - -      %77s |' % ' ')
        for i in range(len(val_protein_list)):
            print('| %-113s |' % val_protein_list[i])
    print('|-------------------------------------------------------------------------------------------------------------------|')

    Chem.rdmolops.AdjustQueryParameters.aromatizeIfPossible = True
    graphMolFileName = "none"
    graphProteinFileName = "none"
    graphMol = {}
    graphProtein = {}
    print('| Preparing the complexes from ligand and protein sdf files                                                         |')

    for thisSet in range(3):
        protein_number = 0
        count_actives = 0
        count_inactives = 0
        set_name = 'training'
        if thisSet == 0:
            protein_list = train_protein_list
            set_name = 'training'
        elif thisSet == 1:
            protein_list = test_protein_list
            set_name = 'testing'
        elif thisSet == 2:
            protein_list = val_protein_list
            set_name = 'validation'
        list_keys = []

        for protein_name in protein_list:
            print('|-----------------------------------------------------------------%50s|' % ' ')
            print('| Protein: %-85s                    |' % protein_name)
            protein_number += 1

            EvdW = 0
            Eelec = 0
            Ehbond = 0
            Ewater = 0
            MScore = 0
            RankScore = 0
            graphs_found = False

            for molFile in os.listdir(args.raw_data_dir):
                # check the presence of "_" so that ACE_ and ACE2_ would be different
                if molFile.find(protein_name + "_") != -1 and molFile.find(".sdf") != -1:
                    # Check whether the graphs are provided if forecaster graphs is set to True
                    if args.forecaster_graphs is False:
                        graphs_found = True
                    else:
                        for molFile2 in os.listdir(args.raw_data_dir):
                            if args.forecaster_graphs is True and molFile2.find(protein_name + "_protein_pro_graph.txt") != -1:
                                graphs_found = True
                                graphProteinFileName = args.raw_data_dir + protein_name + "_protein_pro_graph.txt"
                                graphProtein = get_Forecaster_graph(0, graphProteinFileName, True)
                                break

                    if graphs_found is False:
                        print('| Forecaster generated graph file for the protein is missing %-54s |' % ' ')
                        continue

                    if (molFile.find("_actives") != -1 and molFile.find("_fixed.sdf") != -1) or molFile.find("_good_poses.sdf") != -1:
                        graphs_found = False
                        if args.forecaster_graphs is False:
                            graphs_found = True
                        else:
                            for molFile2 in os.listdir(args.raw_data_dir):
                                if args.forecaster_graphs is True and molFile2.find(protein_name + "_smarted-graph.txt") != -1:
                                    graphs_found = True
                                    graphMolFileName = args.raw_data_dir + molFile2
                                    break
                        if graphs_found is False:
                            print('| Forecaster generated graph file for the ligand is missing  %-54s |' % ' ')
                            continue

                        if args.verbose > 2:
                            print('| Extracting poses from file: %-85s |' % molFile)

                        try:
                            if os.path.isfile(args.raw_data_dir + protein_name + '_binding_site_V3000.sdf'):
                                protein = Chem.MolFromMolFile(args.raw_data_dir + protein_name + '_binding_site_V3000.sdf', removeHs=False)
                            else:
                                protein = Chem.MolFromMolFile(args.raw_data_dir + protein_name + '_protein_pro_binding_site_V3000.sdf', removeHs=False)
                        except:
                            continue

                        if protein is None or protein.GetNumAtoms() == 0:
                            print('| problem with protein file   %-85s |' % ' ')
                            continue

                        add_formal_charges(protein)
                        protein = Chem.RemoveAllHs(protein)

                        if (Chem.SanitizeMol(protein, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                             Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                             Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                             Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                             Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                                             Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
                                             catchErrors=True)) != 0:
                            continue

                        count = 0
                        ligands = Chem.ForwardSDMolSupplier(args.raw_data_dir + molFile, removeHs=False, sanitize=False)

                        for ligand in ligands:
                            if ligand is None or ligand.GetNumAtoms() == 0:
                                continue
                            if count >= args.max_number_of_poses:
                                break
                            add_formal_charges(ligand)

                            rmsd = -1
                            if args.train_mode == 'docking':
                                rmsd = float(ligand.GetProp('FR_FITTED_RMSD'))

                            if args.consider_energy is True:
                                EvdW = float(ligand.GetProp('FR_FITTED_vdW'))
                                Eelec = float(ligand.GetProp('FR_FITTED_Elec'))
                                Ehbond = float(ligand.GetProp('FR_FITTED_Elec_M')) + float(ligand.GetProp('FR_FITTED_Bond_M'))
                                Ewater = float(ligand.GetProp('FR_FITTED_Water_vdW')) + float(ligand.GetProp('FR_FITTED_Water_Elec')) + \
                                         float(ligand.GetProp('FR_FITTED_Water_HBonds'))
                                MScore = float(ligand.GetProp('FR_FITTED_MScore'))
                                RankScore = float(ligand.GetProp('FR_FITTED_Score'))

                            # stereogenic Hs are not removed by removeHs=True or Chem.RemoveHs()
                            try:
                                ligand = Chem.RemoveAllHs(ligand)
                            except:
                                print('| Something is wrong with this molecule, it will be skipped                                                         |')
                                break
                            if (Chem.SanitizeMol(ligand, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                                 Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                                 Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                                 Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                                 Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                                                 Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
                                                 catchErrors=True)) != 0:
                                break

                            if args.forecaster_graphs:
                                graphMol = get_Forecaster_graph(0, graphMolFileName, False)

                            mol_charge = get_mol_charge(ligand)
                            nrot = get_mol_nrot(ligand)
                            n_atoms_lig = get_mol_natoms(ligand)

                            count = count + 1
                            count_actives = count_actives + 1

                            # We remove _DockingRun1 in the name
                            if len(ligand.GetProp('_Name').split("_")) > 2:
                                ligandName = ligand.GetProp('_Name').split("_")[1]
                                interacting_atoms_lig, interacting_atoms_prot = get_interacting_atoms(ligand, protein, args)

                                interaction_pairs = get_interacting_pairs(ligand, protein, interacting_atoms_lig, interacting_atoms_prot, args)

                                if args.verbose > 2:
                                    print('| Active ligands:   #%-6s %-20s in %-25s (%5s out of %5s - %10s set) |'
                                          % (str(count), ligandName, protein_name, protein_number, len(protein_list), set_name), flush=True)

                                if args.forecaster_graphs is False:
                                    complex_mol = (protein, ligand, interaction_pairs, interacting_atoms_lig, interacting_atoms_prot,
                                                   mol_charge, nrot, n_atoms_lig, rmsd, EvdW, Eelec, Ehbond, Ewater, MScore, RankScore)
                                else:
                                    complex_mol = (protein, ligand, graphProtein, graphMol, interaction_pairs, interacting_atoms_lig, interacting_atoms_prot,
                                                   mol_charge, nrot, n_atoms_lig, rmsd, EvdW, Eelec, Ehbond, Ewater, MScore, RankScore)

                                with open(args.data_dir + protein_name + '_active_' + ligandName + '_' + str(count) + '.pkl', 'wb') as complex_f:
                                    pickle.dump(complex_mol, complex_f)

                                list_keys.append(protein_name + '_active_' + ligandName + '_' + str(count) + '.pkl')

                        print('| Number of actives:   %-7.0f    cum: %-35.0f                                          |' % (count, count_actives), flush=True)

                    if (molFile.find("_inactives") != -1 and molFile.find("_fixed.sdf") != -1) or molFile.find("_bad_poses") != -1:

                        graphs_found = False
                        if args.forecaster_graphs is False:
                            graphs_found = True
                        else:
                            for molFile2 in os.listdir(args.raw_data_dir):
                                if args.forecaster_graphs is True and molFile2.find(protein_name + "_smarted-graph.txt") != -1:
                                    graphs_found = True
                                    graphMolFileName = args.raw_data_dir + molFile2
                                    break
                        if graphs_found is False:
                            print('| Forecaster generated graph file for the ligand is missing  %-54s |' % ' ')
                            continue

                        if args.verbose > 2:
                            print('| Extracting poses from file: %-85s |' % molFile)

                        try:
                            if os.path.isfile(args.raw_data_dir + protein_name + '_binding_site_V3000.sdf'):
                                protein = Chem.MolFromMolFile(args.raw_data_dir + protein_name + '_binding_site_V3000.sdf', removeHs=False)
                            else:
                                protein = Chem.MolFromMolFile(args.raw_data_dir + protein_name + '_protein_pro_binding_site_V3000.sdf', removeHs=False)
                        except:
                            continue

                        if protein is None or protein.GetNumAtoms() == 0:
                            print('| problem with protein file   %-85s |' % ' ')
                            continue

                        add_formal_charges(protein)
                        protein = Chem.RemoveAllHs(protein)

                        if (Chem.SanitizeMol(protein, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                             Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                             Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                             Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                             Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                                             Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
                                             catchErrors=True)) != 0:
                            continue

                        count = 0
                        count2 = 0
                        ligands = Chem.ForwardSDMolSupplier(args.raw_data_dir + molFile, removeHs=False, sanitize=False)
                        for ligand in ligands:
                            if ligand is None or ligand.GetNumAtoms() == 0:
                                continue
                            if count >= args.max_number_of_poses:
                                break

                            add_formal_charges(ligand)

                            rmsd = -1
                            if args.train_mode == 'docking':
                                rmsd = float(ligand.GetProp('FR_FITTED_RMSD'))

                            if args.consider_energy is True:
                                EvdW = float(ligand.GetProp('FR_FITTED_vdW'))
                                Eelec = float(ligand.GetProp('FR_FITTED_Elec'))
                                Ehbond = float(ligand.GetProp('FR_FITTED_Elec_M')) + float(ligand.GetProp('FR_FITTED_Bond_M'))
                                Ewater = float(ligand.GetProp('FR_FITTED_Water_vdW')) + float(ligand.GetProp('FR_FITTED_Water_Elec')) + \
                                         float(ligand.GetProp('FR_FITTED_Water_HBonds'))
                                MScore = float(ligand.GetProp('FR_FITTED_MScore'))
                                RankScore = float(ligand.GetProp('FR_FITTED_Score'))

                            try:
                                ligand = Chem.RemoveAllHs(ligand)
                            except:
                                print('| Something is wrong with this molecule, it will be skipped                                                         |')
                                break

                            if (Chem.SanitizeMol(ligand, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                                 Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                                 Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                                 Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                                 Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                                                 Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
                                                 catchErrors=True)) != 0:
                                break

                            # Get the forecaster graph
                            if args.forecaster_graphs:
                                graphMol = get_Forecaster_graph(0, graphMolFileName, False)

                            mol_charge = get_mol_charge(ligand)
                            nrot = get_mol_nrot(ligand)
                            n_atoms_lig = get_mol_natoms(ligand)

                            count = count + 1
                            count2 = count2 + 1
                            count_inactives = count_inactives + 1
                            # We remove _DockingRun1 in the name
                            if len(ligand.GetProp('_Name').split("_")) > 2:
                                ligandName = ligand.GetProp('_Name').split("_")[1]
                                interacting_atoms_lig, interacting_atoms_prot = get_interacting_atoms(ligand, protein, args)
                                
                                interaction_pairs = get_interacting_pairs(ligand, protein, interacting_atoms_lig, interacting_atoms_prot, args)

                                if args.verbose > 2:
                                    print('| Inactive ligands: #%-6s %-20s in %-27s (%5s out of %5s - training set) |'
                                          % (str(count), ligandName, protein_name, protein_number, len(train_protein_list)), flush=True)

                                if args.forecaster_graphs is False:
                                    complex_mol = (protein, ligand, interaction_pairs, interacting_atoms_lig, interacting_atoms_prot,
                                                   mol_charge, nrot, n_atoms_lig, rmsd, EvdW, Eelec, Ehbond, Ewater, MScore, RankScore)
                                else:
                                    complex_mol = (protein, ligand, graphProtein, graphMol, interaction_pairs, interacting_atoms_lig, interacting_atoms_prot,
                                                   mol_charge, nrot, n_atoms_lig, rmsd, EvdW, Eelec, Ehbond, Ewater, MScore, RankScore)

                                with open(args.data_dir + protein_name + '_inactive_' + ligandName + '_' + str(count) + '.pkl', 'wb') as complex_f:
                                    pickle.dump(complex_mol, complex_f)

                                list_keys.append(protein_name + '_inactive_' + ligandName + '_' + str(count) + '.pkl')
                        print('| Number of inactives: %-7.0f    cum: %-35.0f                                          |' % (count2, count_inactives), flush=True)

            if protein_number % 100 == 0:
                if set == 0:
                    # This clears the file and resaves the key list
                    open(args.keys_dir + args.train_keys, "w").close()
                    with open(args.keys_dir + args.train_keys, 'wb') as keys_f:
                        pickle.dump(list_keys, keys_f)
                elif set == 1:
                    open(args.keys_dir + args.test_keys, "w").close()
                    with open(args.keys_dir + args.test_keys, 'wb') as keys_f:
                        pickle.dump(list_keys, keys_f)
                elif set == 2:
                    open(args.keys_dir + args.val_keys, "w").close()
                    with open(args.keys_dir + args.val_keys, 'wb') as keys_f:
                        pickle.dump(list_keys, keys_f)

            if protein_number >= args.max_num_of_systems and args.max_num_of_systems > 0:
                break

        if thisSet == 0:
            # This clears the file and resaves the key list
            open(args.keys_dir + args.train_keys, "w").close()
            with open(args.keys_dir + args.train_keys, 'wb') as keys_f:
                pickle.dump(list_keys, keys_f)
        elif thisSet == 1:
            open(args.keys_dir + args.test_keys, "w").close()
            with open(args.keys_dir + args.test_keys, 'wb') as keys_f:
                pickle.dump(list_keys, keys_f)
        elif thisSet == 2:
            open(args.keys_dir + args.val_keys, "w").close()
            with open(args.keys_dir + args.val_keys, 'wb') as keys_f:
                pickle.dump(list_keys, keys_f)
#        open(args.keys_dir + args.train_keys, "w").close()
#        with open(args.keys_dir + args.train_keys, 'wb') as keys_f:
#            pickle.dump(train_keys, keys_f)
            
        if thisSet == 0:
            print('| Training set done                                                                                                 |')
        elif thisSet == 1:
            print('| Testing set done                                                                                                  |')
        elif thisSet == 2:
            print('| Validation set done                                                                                               |')
        print('|-------------------------------------------------------------------------------------------------------------------|')


def generate_complexes_predict(args):

    good_poses_keys = []
    bad_poses_keys = []
    Chem.rdmolops.AdjustQueryParameters.aromatizeIfPossible = True

    if args.good_and_bad_pairs is False:
        print('| Preparing the complexes from ligand and protein sdf files                                                         |')
    else:
        print('| Preparing the complexes from pairs of poses and protein sdf files                                                 |')
    print('|-----------------------------------------------------------------%50s|' % ' ')
    print('| Protein: %-85s                    |' % args.protein_input)

    EvdW = 0
    Eelec = 0
    Ehbond = 0
    Ewater = 0
    MScore = 0
    RankScore = 0
    graphMolFileName = "none"
    graphProteinFileName = "none"
    graphMol = {}
    graphProtein = {}

    if args.second_pose_input == 'none':
        args.first_same_as_second = True
        args.first_pose_input = args.first_pose_input

    try:
        graphs_found = False
        if os.path.isfile(args.raw_data_dir + args.protein_input + '_protein_pro_binding_site_V3000.sdf'):
            protein = Chem.MolFromMolFile(args.raw_data_dir + args.protein_input + '_protein_pro_binding_site_V3000.sdf', removeHs=False)
            print('| Protein file loaded: %-90s   |' % (args.raw_data_dir + args.protein_input + '_protein_pro_binding_site_V3000.sdf'))
        elif os.path.isfile(args.raw_data_dir + args.protein_input + '_binding_site_V3000.sdf'):
            protein = Chem.MolFromMolFile(args.raw_data_dir + args.protein_input + '_binding_site_V3000.sdf', removeHs=False)
            print('| Protein file loaded: %-90s   |' % (args.raw_data_dir + args.protein_input + '_binding_site_V3000.sdf'))
        else:
            print('| The protein file does not exist                                                                                   |')
            filename = args.raw_data_dir + args.protein_input + '_binding_site_V3000.sdf'
            print('|   Searched for: %-85s             |' % filename)
            filename = args.raw_data_dir + args.protein_input + '_protein_pro_binding_site_V3000.sdf'
            print('|            and: %-85s             |' % filename)
            return False

        # Check whether the graphs are provided if forecaster graphs is set to True
        if args.forecaster_graphs is False:
            graphs_found = True
        else:
            for molFile2 in os.listdir(args.raw_data_dir):
                # print(molFile2, args.raw_data_dir + args.protein_input + '_protein_pro_graph.txt')
                if args.forecaster_graphs is True and molFile2.find(args.protein_input + '_protein_pro_graph.txt') != -1:
                    graphs_found = True
                    graphProteinFileName = args.raw_data_dir + args.protein_input + '_protein_pro_graph.txt'
                    graphProtein = get_Forecaster_graph(0, graphProteinFileName, True)
                    print('| Graph file loaded: %-92s   |' % (args.raw_data_dir + args.protein_input + '_protein_pro_graph.txt'))
                    # print('GraphProtein: \n', graphProtein)
                    break
                if args.forecaster_graphs is True and molFile2.find(args.protein_input + '_graph.txt') != -1:
                    graphs_found = True
                    graphProteinFileName = args.raw_data_dir + args.protein_input + '_graph.txt'
                    graphProtein = get_Forecaster_graph(0, graphProteinFileName, True)
                    print('| Graph file loaded: %-92s   |' % (args.raw_data_dir + args.protein_input + '_graph.txt'))
                    # print('GraphProtein: \n', graphProtein)
                    break

        if graphs_found is False:
            print('| Forecaster-generated graph file for the protein is missing %-54s |' % ' ')
            filename = args.raw_data_dir + args.protein_input + '_protein_pro_graph.txt'
            print('|   Searched for: %-85s             |' % filename)
            filename = args.raw_data_dir + args.protein_input + '_graph.txt'
            print('|            and: %-85s             |' % filename)
            return False

        add_formal_charges(protein)
        protein = Chem.RemoveAllHs(protein)
        print('| Hydrogen atoms and charges fixed on proteins               %-54s |' % ' ')

        if (Chem.SanitizeMol(protein, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                      Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                      Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                      Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                      Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                                      Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
                             catchErrors=True)) != 0:
            print('| Problem with the protein file                                                                                     |')
            return False

        if os.path.isfile(args.raw_data_dir + args.first_pose_input + '.sdf'):
            ligands = Chem.ForwardSDMolSupplier(args.raw_data_dir + args.first_pose_input + '.sdf', removeHs=False, sanitize=False)
            print('| Ligand file loaded: %-93s |' % (args.raw_data_dir + args.first_pose_input + '.sdf'))
        elif os.path.isfile(args.raw_data_dir + args.first_pose_input):
            ligands = Chem.ForwardSDMolSupplier(args.raw_data_dir + args.first_pose_input, removeHs=False, sanitize=False)
            print('| Ligand file loaded: %-93s |' % (args.raw_data_dir + args.first_pose_input))
        else:
            print('| Ligand file is missing                                     %-54s |' % ' ')
            filename = args.raw_data_dir + args.first_pose_input + '.sdf'
            print('|   Searched for: %-85s             |' % filename)
            filename = args.raw_data_dir + args.first_pose_input
            print('|            and: %-85s             |' % filename)
            return False

        graphs_found = False
        if args.forecaster_graphs is False:
            graphs_found = True
        else:
            for molFile2 in os.listdir(args.raw_data_dir):
                if args.forecaster_graphs is True and molFile2.find(args.first_pose_input + "-graph.txt") != -1:
                    graphs_found = True
                    graphMolFileName = args.raw_data_dir + molFile2
                    break
                elif args.forecaster_graphs is True and molFile2.find(args.first_pose_input.removesuffix('.sdf') + "-graph.txt") != -1:
                    graphs_found = True
                    graphMolFileName = args.raw_data_dir + molFile2
                    break
        if graphs_found is False:
            print('| Forecaster-generated graph file for the ligand is missing  %-54s |' % ' ')
            filename = args.raw_data_dir + args.first_pose_input + '-graph.txt'
            print('|   Searched for: %-85s             |' % filename)
            return False

        count = 0
        for ligand in ligands:
            if ligand is None or ligand.GetNumAtoms() == 0:
                continue
            #print('| Reading ligand #%-3s         %-85s |' % (str(count + 1), ' '), flush=True)

            add_formal_charges(ligand)

            rmsd = float(ligand.GetProp('FR_FITTED_RMSD'))
            if rmsd == 0 or rmsd > 90:
                rmsd = 99
            # print('prepare_input_data 555, rmsd: ', rmsd)

            if args.consider_energy is True:
                EvdW = float(ligand.GetProp('FR_FITTED_vdW'))
                Eelec = float(ligand.GetProp('FR_FITTED_Elec'))
                Ehbond = float(ligand.GetProp('FR_FITTED_Elec_M')) + float(ligand.GetProp('FR_FITTED_Bond_M'))
                Ewater = float(ligand.GetProp('FR_FITTED_Water_vdW')) + float(ligand.GetProp('FR_FITTED_Water_Elec')) + \
                         float(ligand.GetProp('FR_FITTED_Water_HBonds'))
                MScore = float(ligand.GetProp('FR_FITTED_MScore'))
                RankScore = float(ligand.GetProp('FR_FITTED_Score'))

            # stereogenic Hs are not removed by removeHs=True
            try:
                ligand = Chem.RemoveAllHs(ligand)
            except:
                print('| Something is wrong with this molecule, it will be skipped                                                         |')
                continue

            if (Chem.SanitizeMol(ligand, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                         Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                         Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                         Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                         Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                                         Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
                                 catchErrors=True)) != 0:
                return False

            if args.forecaster_graphs:
                graphMol = get_Forecaster_graph(0, graphMolFileName, False)

            mol_charge = get_mol_charge(ligand)
            nrot = get_mol_nrot(ligand)
            n_atoms_lig = get_mol_natoms(ligand)

            count = count + 1

            interacting_atoms_lig, interacting_atoms_prot = get_interacting_atoms(ligand, protein, args)
            interaction_pairs = get_interacting_pairs(ligand, protein, interacting_atoms_lig, interacting_atoms_prot, args)

            if args.forecaster_graphs is False:
                complex_mol = (protein, ligand, interaction_pairs, interacting_atoms_lig, interacting_atoms_prot,
                               mol_charge, nrot, n_atoms_lig, rmsd, EvdW, Eelec, Ehbond, Ewater, MScore, RankScore)
            else:
                complex_mol = (protein, ligand, graphProtein, graphMol, interaction_pairs, interacting_atoms_lig, interacting_atoms_prot,
                               mol_charge, nrot, n_atoms_lig, rmsd, EvdW, Eelec, Ehbond, Ewater, MScore, RankScore)

            with open(args.data_dir + 'temp1_' + str(count) + '.mfi.tmp.pkl', 'wb') as complex_f:
                pickle.dump(complex_mol, complex_f)

            good_poses_keys.append('temp1_' + str(count) + '.mfi.tmp.pkl')

        open(args.keys_dir + 'first_poses_keys.mfi.tmp.pkl', "w").close()
        with open(args.keys_dir + 'first_poses_keys.mfi.tmp.pkl', 'wb') as keys_f:
            pickle.dump(good_poses_keys, keys_f)

        if args.first_same_as_second is True:
            open(args.keys_dir + 'second_poses_keys.mfi.tmp.pkl', "w").close()
            with open(args.keys_dir + 'second_poses_keys.mfi.tmp.pkl', 'wb') as keys_f:
                pickle.dump(good_poses_keys, keys_f)
        elif args.second_pose_input != 'none':
            # ligands = Chem.ForwardSDMolSupplier(args.raw_data_dir + args.second_pose_input + '.sdf', removeHs=False, sanitize=False)
            if os.path.isfile(args.raw_data_dir + args.second_pose_input + '.sdf'):
                ligands = Chem.ForwardSDMolSupplier(args.raw_data_dir + args.second_pose_input + '.sdf', removeHs=False, sanitize=False)
                print('| Ligand file loaded: %-93s |' % (args.raw_data_dir + args.second_pose_input + '.sdf'))
            elif os.path.isfile(args.raw_data_dir + args.second_pose_input):
                ligands = Chem.ForwardSDMolSupplier(args.raw_data_dir + args.second_pose_input, removeHs=False, sanitize=False)
                print('| Ligand file loaded: %-93s |' % (args.raw_data_dir + args.second_pose_input))
            else:
                print('| Ligand file is missing                                     %-54s |' % ' ')
                filename = args.raw_data_dir + args.second_pose_input + '.sdf'
                print('|   Searched for: %-85s             |' % filename)
                filename = args.raw_data_dir + args.second_pose_input
                print('|            and: %-85s             |' % filename)
                return False

            graphs_found = False
            if args.forecaster_graphs is False:
                graphs_found = True
            else:
                for molFile2 in os.listdir(args.raw_data_dir):
                    if args.forecaster_graphs is True and molFile2.find(args.second_pose_input + "-graph.txt") != -1:
                        graphs_found = True
                        graphMolFileName = args.raw_data_dir + molFile2
                        break
                    elif args.forecaster_graphs is True and molFile2.find(args.second_pose_input.removesuffix('.sdf') + "-graph.txt") != -1:
                        graphs_found = True
                        graphMolFileName = args.raw_data_dir + molFile2
                        break
            if graphs_found is False:
                print('| Forecaster generated graph file for the ligand is missing  %-54s |' % ' ')
                return False

            count = 0
            for ligand in ligands:
                if ligand is None or ligand.GetNumAtoms() == 0:
                    continue

                add_formal_charges(ligand)

                rmsd = float(ligand.GetProp('FR_FITTED_RMSD'))
                if rmsd == 0 or rmsd > 90:
                    rmsd = 99
                # print('prepare_input_data 616, rmsd: ', rmsd)

                if args.consider_energy is True:
                    EvdW = float(ligand.GetProp('FR_FITTED_vdW'))
                    Eelec = float(ligand.GetProp('FR_FITTED_Elec'))
                    Ehbond = float(ligand.GetProp('FR_FITTED_Elec_M')) + float(ligand.GetProp('FR_FITTED_Bond_M'))
                    Ewater = float(ligand.GetProp('FR_FITTED_Water_vdW')) + float(ligand.GetProp('FR_FITTED_Water_Elec')) + \
                             float(ligand.GetProp('FR_FITTED_Water_HBonds'))
                    MScore = float(ligand.GetProp('FR_FITTED_MScore'))
                    RankScore = float(ligand.GetProp('FR_FITTED_Score'))

                # stereogenic Hs are not removed by removeHs=True
                try:
                    ligand = Chem.RemoveAllHs(ligand)
                except:
                    print('| Something is wrong with this molecule, it will be skipped                                                         |')
                    continue
                if (Chem.SanitizeMol(ligand, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                             Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                             Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                             Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                             Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                                             Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
                                     catchErrors=True)) != 0:
                    return False

                if args.forecaster_graphs:
                    graphMol = get_Forecaster_graph(0, graphMolFileName, False)
                #print('graphmol 2: ', graphMol)

                mol_charge = get_mol_charge(ligand)
                nrot = get_mol_nrot(ligand)
                n_atoms_lig = get_mol_natoms(ligand)

                count = count + 1

                interacting_atoms_lig, interacting_atoms_prot = get_interacting_atoms(ligand, protein, args)
                interaction_pairs = get_interacting_pairs(ligand, protein, interacting_atoms_lig, interacting_atoms_prot, args)

                if args.forecaster_graphs is False:
                    complex_mol = (protein, ligand, interaction_pairs, interacting_atoms_lig, interacting_atoms_prot,
                                   mol_charge, nrot, n_atoms_lig, rmsd, EvdW, Eelec, Ehbond, Ewater, MScore, RankScore)
                else:
                    complex_mol = (protein, ligand, graphProtein, graphMol, interaction_pairs, interacting_atoms_lig, interacting_atoms_prot,
                                   mol_charge, nrot, n_atoms_lig, rmsd, EvdW, Eelec, Ehbond, Ewater, MScore, RankScore)

                with open(args.data_dir + 'temp2_' + str(count) + '.mfi.tmp.pkl', 'wb') as complex_f:
                    pickle.dump(complex_mol, complex_f)

                bad_poses_keys.append('temp2_' + str(count) + '.mfi.tmp.pkl')

            open(args.keys_dir + 'second_poses_keys.mfi.tmp.pkl', "w").close()
            with open(args.keys_dir + 'second_poses_keys.mfi.tmp.pkl', 'wb') as keys_f:
                pickle.dump(bad_poses_keys, keys_f)

        print('| Complexes prepared                                                                                                |')
        print('|-------------------------------------------------------------------------------------------------------------------|')
    except:
        print('| Problem preparing complexes                                                                                       |')
        print('|-------------------------------------------------------------------------------------------------------------------|')
        return False

    return True


def prepare_graphs(args):

    for thisSet in range(3):
        list_keys_graphs = []
        keys_filename = 'none'
        keys_graph_filename = 'none'
        if thisSet == 0:
           keys_filename = args.train_keys
           keys_graph_filename = args.train_keys_graphs
        elif thisSet == 1:
            keys_filename = args.test_keys
            keys_graph_filename = args.test_keys_graphs
        elif thisSet == 2:
            keys_filename = args.val_keys
            keys_graph_filename = args.val_keys_graphs

        if keys_filename == 'none':
            return

        count = 0
        if os.path.isfile(args.keys_dir + keys_filename) is True:
            with open(args.keys_dir + keys_filename, 'rb') as fp:
                list_keys = pickle.load(fp)

                for key in list_keys:
                    filename = key
                    # print('prepare_set 34, generate datasets ', filename)
                    count += 1
                    with open(args.data_dir + filename, 'rb') as key_f:
                        protein, ligand, interaction_pairs, interacting_atoms_lig, interacting_atoms_prot, mol_charge, nrot, n_atoms_lig, rmsd,\
                            EvdW, Eelec, Ehbond, Ewater, MScore, RankScore = pickle.load(key_f)

                        # Get node features
                        node_features = []

                        node_features = get_node_features(ligand, node_features, interacting_atoms_lig, 'ligand', args)
                        node_features = get_node_features(protein, node_features, interacting_atoms_prot, 'protein', args)

                        node_features = np.array(node_features)
                        node_features = torch.tensor(node_features, dtype=torch.float)

                        # Get edge features
                        edge_indices_pl, edge_features_pl = \
                            get_edge_features(ligand, protein, interaction_pairs, interacting_atoms_lig, interacting_atoms_prot, args)

                        # Get labels info
                        # if pose prediction, the value goes from 1 to 0 between ca. 1.5 and 2.5 using a sigmoid function
                        label = get_labels(1 if '_active' in filename else 0)

                        # provide the first atom number (will be used to identify the start of a complex in a batch)
                        node_index = torch.tensor(range(len(node_features)), dtype=torch.long)

                        data = (node_features, node_index, edge_indices_pl, edge_features_pl, mol_charge, nrot, EvdW, Eelec, Ehbond, Ewater, MScore, RankScore,
                                filename, label)
                        with open(args.graphs_dir + filename.replace('.pkl', '') + '_graph.pkl', 'wb') as data_f:
                            pickle.dump(data, data_f)

                        print('| graphs from: %-78s (%5s out of %5s) |' % (key, str(count), len(list_keys)), flush=True)

                        list_keys_graphs.append(filename.replace('.pkl', '') + '_graph.pkl')

        open(args.keys_dir + keys_graph_filename, "w").close()
        with open(args.keys_dir + keys_graph_filename, 'wb') as keys_f:
            pickle.dump(list_keys_graphs, keys_f)


def prepare_graphs_predict(args):

    predict_keys_graphs = []
    count = 0
    if os.path.isfile(args.keys_dir + 'first_poses_keys.mfi.tmp.pkl') is True:
        with open(args.keys_dir + 'first_poses_keys.mfi.tmp.pkl', 'rb') as fp:
            predict_keys = pickle.load(fp)

            for key in predict_keys:
                filename = key
                # print('prepare_set 34, generate datasets ', filename)
                count += 1
                with open(args.data_dir + filename, 'rb') as key_f:
                    protein, ligand, interaction_pairs, interacting_atoms_lig, interacting_atoms_prot, mol_charge, nrot, n_atoms_lig, rmsd,\
                        EvdW, Eelec, Ehbond, Ewater, MScore, RankScore = pickle.load(key_f)

                    # print('prepare_set 38')
                    # Get node features
                    node_features = []

                    # print('prepare_set 42: \n', list(interacting_atoms_lig), ligand.GetNumAtoms())
                    node_features = get_node_features(ligand, node_features, interacting_atoms_lig, 'ligand', args)
                    # print('prepare_set 45: \n', len(complex_node_features))

                    # print('prepare_set 47: \n', list(interacting_atoms_prot), protein.GetNumAtoms())
                    node_features = get_node_features(protein, node_features, interacting_atoms_prot, 'protein', args)

                    node_features = np.array(node_features)
                    node_features = torch.tensor(node_features, dtype=torch.float)

                    # Get edge features
                    # edge_indices_0, edge_features_0, \
                    edge_indices_pl, edge_features_pl = \
                        get_edge_features(ligand, protein, interaction_pairs, interacting_atoms_lig, interacting_atoms_prot, args)

                    label = 0

                    # provide the first atom number (will be used to identify the start of a complex in a batch)
                    node_index = torch.tensor(range(len(node_features)), dtype=torch.long)

                    # print("prepare-set 68: 16 node features:   ", complex_node_features.shape[1])  #, node_index.shape)
                    # print("prepare-set 69: 9 bond features:    ", complex_edge_features.shape[1])  #, complex_edge_indices.shape)
                    # print("prepare-set 70: charge, nrot, label: ", mol_charge.shape, nrot.shape, label.shape, '\n', flush=True)
                    # print("prepare-set 71: ligand name: ", filename, '\n', flush=True)
                    # print("prepare-set 70: \n", mol_charge, nrot, label)

                    # edge_indices_0, edge_features_0,
                    data = (node_features, node_index, edge_indices_pl, edge_features_pl, mol_charge, nrot, EvdW, Eelec, Ehbond, Ewater, MScore, RankScore,
                            filename, label)
                    with open(args.graphs_dir + filename.replace('.pkl', '') + '_graph.pkl', 'wb') as data_f:
                        pickle.dump(data, data_f)

                    print('| graphs from: %-78s (%5s out of %5s) |' % (key, str(count), len(predict_keys)), flush=True)

                    predict_keys_graphs.append(filename.replace('.pkl', '') + '_graph.pkl')

    open(args.keys_dir + 'poses_keys_graph.mfi.tmp.pkl', "w").close()
    with open(args.keys_dir + 'poses_keys_graph.mfi.tmp.pkl', 'wb') as keys_f:
        pickle.dump(predict_keys_graphs, keys_f)


def prepare_pairs_of_graphs(args):
    # scrambling for more diversity
    rng = np.random.default_rng(12345)
    for thisSet in range(3):
        list_keys_graphs = []
        keys_filename = 'none'
        keys_graph_filename = 'none'
        if thisSet == 0:
            keys_filename = args.train_keys
            keys_graph_filename = args.train_keys_graphs
        elif thisSet == 1:
            keys_filename = args.test_keys
            keys_graph_filename = args.test_keys_graphs
        elif thisSet == 2:
            keys_filename = args.val_keys
            keys_graph_filename = args.val_keys_graphs

        if keys_filename == 'none':
            return

        list_keys_graphs = []
        count = 0
        if os.path.isfile(args.keys_dir + keys_filename) is True:
            with open(args.keys_dir + keys_filename, 'rb') as fp:
                list_keys = pickle.load(fp)

                for key in list_keys:
                    rand_ints = rng.integers(low=1, high=args.max_number_of_poses, size=1)

                    # Naming convention: prefix: protein name + '_1' if key is active, '_2' if key is inactive + pose number
                    if '_active' in key:
                        filename_good = key
                        filename_bad = filename_good.replace('_active', '_inactive').replace('.pkl', '')
                        filename = filename_good.replace('_active', '_1').replace('.pkl', '').replace('_ligand', '')
                        if filename_bad[-2] == '_':  # in case one digit
                            filename_bad = filename_bad[:-1] + str(rand_ints[0]) + '.pkl'
                        else:
                            filename_bad = filename_bad[:-2] + str(rand_ints[0]) + '.pkl'
                    else:
                        filename_bad = key
                        filename_good = filename_bad.replace('_inactive', '_active').replace('.pkl', '')
                        filename = filename_good.replace('_active', '_2').replace('.pkl', '').replace('_ligand', '')
                        if filename_good[-2] == '_':  # in case one digit
                            filename_good = filename_good[:-1] + str(rand_ints[0]) + '.pkl'
                        else:
                            filename_good = filename_good[:-2] + str(rand_ints[0]) + '.pkl'

                    if os.path.isfile(args.data_dir + filename_good) is False or os.path.isfile(args.data_dir + filename_bad) is False:
                        continue

                    with open(args.data_dir + filename_good, 'rb') as key_f_good, open(args.data_dir + filename_bad, 'rb') as key_f_bad:
                        if args.forecaster_graphs is False:
                            protein_good, ligand_good, interaction_pairs_good, interacting_atoms_lig_good, interacting_atoms_prot_good, mol_charge_good, nrot_good, \
                                n_atoms_lig_good, rmsd_good, EvdW_good, Eelec_good, Ehbond_good, Ewater_good, MScore_good, RankScore_good = pickle.load(key_f_good)
                            protein_bad, ligand_bad, interaction_pairs_bad, interacting_atoms_lig_bad, interacting_atoms_prot_bad, mol_charge_bad, nrot_bad, \
                                n_atoms_lig_bad, rmsd_bad, EvdW_bad, Eelec_bad, Ehbond_bad, Ewater_bad, MScore_bad, RankScore_bad = pickle.load(key_f_bad)
                        else:
                            protein_good, ligand_good, graphProtein_good, graphMol_good, interaction_pairs_good, interacting_atoms_lig_good, \
                                interacting_atoms_prot_good, mol_charge_good, nrot_good, n_atoms_lig_good, rmsd_good, EvdW_good, Eelec_good, Ehbond_good, \
                                Ewater_good, MScore_good, RankScore_good = pickle.load(key_f_good)
                            protein_bad, ligand_bad, graphProtein_bad, graphMol_bad, interaction_pairs_bad, interacting_atoms_lig_bad, \
                                interacting_atoms_prot_bad, mol_charge_bad, nrot_bad, n_atoms_lig_bad, rmsd_bad, EvdW_bad, Eelec_bad, Ehbond_bad, \
                                Ewater_bad, MScore_bad, RankScore_bad = pickle.load(key_f_bad)
                        if ligand_good is None or ligand_good.GetNumAtoms() == 0:
                            continue
                        if ligand_bad is None or ligand_bad.GetNumAtoms() == 0:
                            continue
                        if args.forecaster_graphs is True:
                            if graphProtein_good is None or graphProtein_bad is None:
                                continue
                            if graphMol_good is None or graphMol_bad is None:
                                continue

                        if rmsd_good > args.max_rmsd_good or rmsd_bad < args.min_rmsd_bad:
                            continue

                        count += 1
                        if args.scrambling_graphs is True:
                            ligand_good, ligand_bad = scramble_atom_list_pairs(ligand_good, ligand_bad, rng)

                        # Get node features
                        node_features = []
                        all_interacting_atoms = []

                        #The node features (independent on the conformation) are the same for both good and bad poses
                        if args.forecaster_graphs is False:
                            node_features = get_node_features(ligand_good, node_features, all_interacting_atoms, 'ligand', args)
                            node_features = get_node_features(protein_good, node_features, all_interacting_atoms, 'protein', args)
                        else:
                            node_features = get_node_features_from_graphs(ligand_good, protein_good, graphProtein_good, graphMol_good, node_features,
                                                                          filename_bad, count, args)

                        if node_features is False:
                            continue

                        node_features = np.array(node_features)
                        node_features = torch.tensor(node_features, dtype=torch.float)

                        # provide the first atom number (will be used to identify the start of a complex in a batch)
                        node_index = torch.tensor(range(len(node_features)), dtype=torch.long)

                        # get edge features
                        if (count % 2) == 0:
                            label = 1.0
                            # Get edge features
                            if args.forecaster_graphs is False:
                                edge_indices_good, edge_features_good, edge_indices_bad, edge_features_bad = \
                                    get_edge_features_pairs(ligand_good, ligand_bad, protein_good, args)
                            else:
                                try:
                                    edge_indices_good, edge_features_good, edge_indices_bad, edge_features_bad = \
                                        get_edge_features_pairs_from_graphs(ligand_good, ligand_bad, protein_good, graphProtein_good,
                                                                            graphMol_good, graphMol_bad, count, args)
                                except:
                                    continue

                            data = (node_features, node_index, edge_indices_good, edge_features_good, edge_indices_bad, edge_features_bad, mol_charge_good,
                                    nrot_good, EvdW_good, Eelec_good, Ehbond_good, Ewater_good, MScore_good, RankScore_good,
                                    EvdW_bad, Eelec_bad, Ehbond_bad, Ewater_bad, MScore_bad, RankScore_bad, filename, label)

                            with open(args.graphs_dir + filename + '_1_graph.pkl', 'wb') as data_f:
                                pickle.dump(data, data_f)
                            list_keys_graphs.append(filename + '_1_graph.pkl')
                        else:
                            label = 0.0
                            # Get edge features
                            if args.forecaster_graphs is False:
                                edge_indices_bad, edge_features_bad, edge_indices_good, edge_features_good = \
                                    get_edge_features_pairs(ligand_bad, ligand_good, protein_bad, args)
                            else:
                                try:
                                    edge_indices_bad, edge_features_bad, edge_indices_good, edge_features_good = \
                                        get_edge_features_pairs_from_graphs(ligand_bad, ligand_good, protein_bad, graphProtein_bad,
                                                                            graphMol_bad, graphMol_good, count, args)
                                except:
                                    continue

                            data = (node_features, node_index, edge_indices_bad, edge_features_bad, edge_indices_good, edge_features_good, mol_charge_good,
                                    nrot_good, EvdW_bad, Eelec_bad, Ehbond_bad, Ewater_bad, MScore_bad, RankScore_bad,
                                    EvdW_good, Eelec_good, Ehbond_good, Ewater_good, MScore_good, RankScore_good, filename, label)

                            with open(args.graphs_dir + filename + '_2_graph.pkl', 'wb') as data_f:
                                pickle.dump(data, data_f)
                            list_keys_graphs.append(filename + '_2_graph.pkl')

                        filename_output = filename_good + " " + filename_bad
                        print('| graphs from: %-77s (%6s out of %6s) |' % (filename_output, str(count), len(list_keys)), flush=True)

        open(args.keys_dir + keys_graph_filename, "w").close()
        with open(args.keys_dir + keys_graph_filename, 'wb') as keys_f:
            pickle.dump(list_keys_graphs, keys_f)


def prepare_pairs_of_graphs_predict(args):
    predict_keys_graphs = []
    count = 0
    if os.path.isfile(args.keys_dir + 'first_poses_keys.mfi.tmp.pkl') is True and os.path.isfile(args.keys_dir + 'second_poses_keys.mfi.tmp.pkl') is True:
        with open(args.keys_dir + 'first_poses_keys.mfi.tmp.pkl', 'rb') as fp1, open(args.keys_dir + 'second_poses_keys.mfi.tmp.pkl', 'rb') as fp2:
            predict_keys_1 = pickle.load(fp1)
            predict_keys_2 = pickle.load(fp2)
            count_pose_1 = 0
            count_pose_2 = 0

            for key_1 in predict_keys_1:
                count_pose_1 += 1
                count_pose_2 = 0
                filename_1 = key_1

                for key_2 in predict_keys_2:
                    # If we compare poses from the same file, no need to compare 1 to 1 and no need to compare 1 to 2 and then 2 to 1:
                    count_pose_2 += 1
                    if count_pose_2 == count_pose_1 and args.first_same_as_second is True:
                        continue

                    filename_2 = key_2

                    if os.path.isfile(args.data_dir + filename_1) is False or os.path.isfile(args.data_dir + filename_2) is False:
                        continue

                    # print('prepare_input_data 1396, generate datasets ', filename_1)
                    count += 1
                    # print("prepare_input_data 1398: ", count, flush=True)
                    with open(args.data_dir + filename_1, 'rb') as key_f_1, open(args.data_dir + filename_2, 'rb') as key_f_2:
                        if args.forecaster_graphs is False:
                            protein_1, ligand_1, interaction_pairs_1, interacting_atoms_lig_1, interacting_atoms_prot_1, mol_charge_1, nrot_1, \
                                n_atoms_lig_1, rmsd_1, EvdW_1, Eelec_1, Ehbond_1, Ewater_1, MScore_1, RankScore_1 = pickle.load(key_f_1)
                            protein_2, ligand_2, interaction_pairs_2, interacting_atoms_lig_2, interacting_atoms_prot_2, mol_charge_2, nrot_2, \
                                n_atoms_lig_2, rmsd_2, EvdW_2, Eelec_2, Ehbond_2, Ewater_2, MScore_2, RankScore_2 = pickle.load(key_f_2)
                        else:
                            protein_1, ligand_1, graphProtein_1, graphMol_1, interaction_pairs_1, interacting_atoms_lig_1, \
                                interacting_atoms_prot_1, mol_charge_1, nrot_1, n_atoms_lig_1, rmsd_1, EvdW_1, Eelec_1, Ehbond_1, \
                                Ewater_1, MScore_1, RankScore_1 = pickle.load(key_f_1)
                            protein_2, ligand_2, graphProtein_2, graphMol_2, interaction_pairs_2, interacting_atoms_lig_2, \
                                interacting_atoms_prot_bad, mol_charge_2, nrot_2, n_atoms_lig_2, rmsd_2, EvdW_2, Eelec_2, Ehbond_2, \
                                Ewater_2, MScore_2, RankScore_2 = pickle.load(key_f_2)

                        # Get node features
                        node_features = []
                        all_interacting_atoms = []

                        if args.forecaster_graphs is False:
                            node_features = get_node_features(ligand_1, node_features, all_interacting_atoms, 'ligand', args)
                            node_features = get_node_features(protein_1, node_features, all_interacting_atoms, 'protein', args)
                        else:
                            node_features = get_node_features_from_graphs(ligand_1, protein_1, graphProtein_1, graphMol_1, node_features,
                                                                          filename_1, count, args)

                        node_features = np.array(node_features)
                        node_features = torch.tensor(node_features, dtype=torch.float)

                        # provide the first atom number (will be used to identify the start of a complex in a batch)
                        node_index = torch.tensor(range(len(node_features)), dtype=torch.long)

                        # Get edge features
                        if args.forecaster_graphs is False:
                            edge_indices_1, edge_features_1, edge_indices_2, edge_features_2 = \
                                get_edge_features_pairs(ligand_1, ligand_2, protein_1, args)
                        else:
                            try:
                                edge_indices_1, edge_features_1, edge_indices_2, edge_features_2 = \
                                    get_edge_features_pairs_from_graphs(ligand_1, ligand_2, protein_1, graphProtein_1,
                                                                        graphMol_1, graphMol_2, count, args)
                            except:
                                continue

                        #edge_indices_1, edge_features_1, edge_indices_2, edge_features_2 = \
                        #    get_edge_features_pairs(ligand_1, ligand_2, protein_1, args)

                        #print("prepare_input_data 1455: ", edge_indices_1.shape, edge_indices_2.shape, flush=True)
                        # print("prepare-set 71: ligand name: ", filename, '\n', flush=True)
                        # print("prepare-set 70: \n", mol_charge, nrot, label)

                        data_1 = (node_features, node_index, edge_indices_1, edge_features_1, edge_indices_2, edge_features_2, mol_charge_1,
                                  nrot_1, EvdW_1, Eelec_1, Ehbond_1, Ewater_1, MScore_1, RankScore_1,
                                  EvdW_2, Eelec_2, Ehbond_2, Ewater_2, MScore_2, RankScore_2, filename_1, 1)

                        # Get edge features
                        if args.forecaster_graphs is False:
                            edge_indices_2, edge_features_2, edge_indices_1, edge_features_1 = \
                                get_edge_features_pairs(ligand_2, ligand_1, protein_2, args)
                        else:
                            try:
                                edge_indices_2, edge_features_2, edge_indices_1, edge_features_1 = \
                                    get_edge_features_pairs_from_graphs(ligand_2, ligand_1, protein_2, graphProtein_2,
                                                                        graphMol_2, graphMol_1, count, args)
                            except:
                                continue

                        #edge_indices_2, edge_features_2, edge_indices_1, edge_features_1 = \
                        #    get_edge_features_pairs(ligand_2, ligand_1, protein_2, args)

                        data_2 = (node_features, node_index, edge_indices_2, edge_features_2, edge_indices_1, edge_features_1, mol_charge_2,
                                  nrot_2, EvdW_2, Eelec_2, Ehbond_2, Ewater_2, MScore_2, RankScore_2,
                                  EvdW_1, Eelec_1, Ehbond_1, Ewater_1, MScore_1, RankScore_1, filename_2, 0)

                        graph_fileName_1 = 'pair_' + str(count) + '_1_graph.pkl'
                        graph_fileName_2 = 'pair_' + str(count) + '_2_graph.pkl'

                        if args.first_same_as_second is True:
                            graph_fileName_1 = 'pair_' + str(rmsd_1) + '_' + str(rmsd_2) + '_' + str(count_pose_1) + '_' + str(count_pose_2) + '_graph.pkl'
                            with open(args.graphs_dir + graph_fileName_1, 'wb') as data_f:
                                pickle.dump(data_1, data_f)

                        else:
                            if rmsd_1 < 99 and rmsd_2 < 99:
                                graph_fileName_1 = 'pair_' + str(count) + '_' + str(rmsd_1) + '_' + str(rmsd_2) + '_graph.pkl'
                                graph_fileName_2 = 'pair_' + str(count) + '_' + str(rmsd_2) + '_' + str(rmsd_1) + '_graph.pkl'

                            with open(args.graphs_dir + graph_fileName_1, 'wb') as data_f:
                                pickle.dump(data_1, data_f)
                            with open(args.graphs_dir + graph_fileName_2, 'wb') as data_f:
                                pickle.dump(data_2, data_f)

                    print('| graphs # %5s out of %5s %85s |' % (str(count), len(predict_keys_1) * len(predict_keys_2), ' '), flush=True)

                    predict_keys_graphs.append(graph_fileName_1)
                    if args.first_same_as_second is False:
                        predict_keys_graphs.append(graph_fileName_2)
    else:
        print('| File not found: %86s |' % (args.keys_dir + 'first_poses_keys.mfi.tmp.pkl'))
        print('|             or: %86s |' % (args.keys_dir + 'second_poses_keys.mfi.tmp.pkl'))

    open(args.keys_dir + 'poses_keys_graph.mfi.tmp.pkl', "w").close()
    with open(args.keys_dir + 'poses_keys_graph.mfi.tmp.pkl', 'wb') as keys_f:
        pickle.dump(predict_keys_graphs, keys_f)


# Splits the poses into correct (RMSD <= 2Angs) and incorrect (RMSD > 2 Angs)
def split_good_bad_poses(args):
    # construct a random number generator - rng
    rng = np.random.default_rng(12345)
    protein_list = []
    # read the list of pdb id's

    if os.path.isfile(args.raw_data_dir + args.list_trainset) is True and os.path.isfile(args.raw_data_dir + args.list_testset) is True:
        print('| Training set selected from %-86s |' % args.list_trainset)
        with open(args.raw_data_dir + args.list_trainset, 'rb') as list_f:
            for line in list_f:
                protein_list.append(line.strip().decode('UTF-8'))
        print('| Testing set selected from %-86s  |' % args.list_testset)
        with open(args.raw_data_dir + args.list_testset, 'rb') as list_f:
            for line in list_f:
                protein_list.append(line.strip().decode('UTF-8'))
        print('| Validation set selected from %-83s  |' % args.list_valset)
        with open(args.raw_data_dir + args.list_valset, 'rb') as list_f:
            for line in list_f:
                protein_list.append(line.strip().decode('UTF-8'))
    else:
        print('| Complexes built from files in data file                                                                           |')
        protein_list = os.listdir(args.raw_data_dir)
        protein_list = [x for x in protein_list if os.path.getsize(args.raw_data_dir + '/' + x) > 0]
        protein_list = [x.replace('_output_initial_pose.sdf', '') for x in protein_list
                        if '_output_initial_pose.sdf' in x]

    # print(protein_list)

    protein_number = 0
    total_count_good = 0
    total_count_bad = 0
    for protein_id in protein_list:
        if protein_number == 0 and args.verbose > 2:
            print('|-------------------------------------------------------------------------------------------------------------------|')
            print('|     # | Protein name                      | # of good poses | cumulative      | # of bad poses  | cumulative      |')
        protein_number += 1
        print('| %5s | %-33s | ' % (protein_number, protein_id), end='')
        initial_conf = True
        reference_energy = 150
        sanitization = True
        # time.sleep(1)
        energies = []
        rmsd = []
        # Combine the files into a single file starting with the minimized Xray pose (name: $pdb_all_poses.sdf).
        # Remove those too high in energy and update the lowest energy observed so far
        # if os.path.exists(args.raw_data_dir + protein_id + '_all_poses.sdf') is False:
        #     print('1123: file ', args.raw_data_dir + protein_id + '_all_poses.sdf', ' does not exist')
        with Chem.SDWriter(args.raw_data_dir + protein_id + '_all_poses.sdf') as w:
            try:
                ligands = Chem.ForwardSDMolSupplier(args.raw_data_dir + protein_id + '_output_initial_pose.sdf',
                                                    removeHs=False, sanitize=False)
            except:
                print('problem reading the output_initial_pose.sdf file                      |')
                continue

            for ligand in ligands:
                if ligand is None or ligand.GetNumAtoms() == 0:
                    continue
                if initial_conf is True:
                    reference_energy = float(ligand.GetProp('FR_FITTED_Energy'))
                    initial_conf = False

                try:
                    if float(ligand.GetProp('FR_FITTED_Energy')) > (reference_energy + args.energy_threshold):
                        continue
                except:
                    print('problem reading the output_initial_pose.sdf file                      |')
                    continue

                if float(ligand.GetProp('FR_FITTED_Energy')) < reference_energy:
                    reference_energy = float(ligand.GetProp('FR_FITTED_Energy'))

                if Chem.SanitizeMol(ligand, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                    Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                    Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                    Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                                    Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                    Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                                    Chem.SanitizeFlags.SANITIZE_ADJUSTHS, catchErrors=True) == 0:
                    add_formal_charges(ligand)

                    try:
                        w.write(ligand)
                        energies.append(float(ligand.GetProp('FR_FITTED_Energy')))
                        rmsd.append(float(ligand.GetProp('FR_FITTED_RMSD')))
                    except:
                        sanitization = False
                        break

            if sanitization is False:
                continue

            if args.debug == 0:
                try:
                    ligands = Chem.ForwardSDMolSupplier(args.raw_data_dir + protein_id + '_output.sdf',
                                                        removeHs=False, sanitize=False)
                except:
                    print('problem reading the output.sdf file                                   |')
                    print('|       |                                   | ', end='')

                for ligand in ligands:
                    if initial_conf is True:
                        reference_energy = float(ligand.GetProp('FR_FITTED_Energy'))
                        initial_conf = False

                    try:
                        if float(ligand.GetProp('FR_FITTED_Energy')) > (reference_energy + args.energy_threshold):
                            continue
                    except:
                        continue

                    if float(ligand.GetProp('FR_FITTED_Energy')) < reference_energy:
                        reference_energy = float(ligand.GetProp('FR_FITTED_Energy'))

                    if Chem.SanitizeMol(ligand, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                             Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                             Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                             Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                                             Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                             Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                                             Chem.SanitizeFlags.SANITIZE_ADJUSTHS, catchErrors=True) == 0:
                        add_formal_charges(ligand)
                        w.write(ligand)
                        energies.append(float(ligand.GetProp('FR_FITTED_Energy')))
                        rmsd.append(float(ligand.GetProp('FR_FITTED_RMSD')))

                try:
                    ligands = Chem.ForwardSDMolSupplier(args.raw_data_dir + protein_id + '_output_evolving_population.sdf',
                                                        removeHs=False, sanitize=False)
                    for ligand in ligands:
                        if ligand is None or ligand.GetNumAtoms() == 0:
                            continue
                        if float(ligand.GetProp('FR_FITTED_Energy')) > (reference_energy + args.energy_threshold):
                            continue

                        if float(ligand.GetProp('FR_FITTED_Energy')) < reference_energy:
                            reference_energy = float(ligand.GetProp('FR_FITTED_Energy'))

                        if Chem.SanitizeMol(ligand, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                            Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                            Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                            Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                                            Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                            Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                                            Chem.SanitizeFlags.SANITIZE_ADJUSTHS, catchErrors=True) == 0:
                            add_formal_charges(ligand)
                            w.write(ligand)
                            energies.append(float(ligand.GetProp('FR_FITTED_Energy')))
                            rmsd.append(float(ligand.GetProp('FR_FITTED_RMSD')))
                except:
                    print('problem reading the output_evolving_population.sdf file               |')
                    print('|       |                                   | ', end='')

                try:
                    ligands = Chem.ForwardSDMolSupplier(args.raw_data_dir + protein_id + '_output_initial_population.sdf',
                                                        removeHs=False, sanitize=False)
                    for ligand in ligands:
                        if ligand is None or ligand.GetNumAtoms() == 0:
                            continue
                        if float(ligand.GetProp('FR_FITTED_Energy')) > (reference_energy + args.energy_threshold):
                            continue

                        if float(ligand.GetProp('FR_FITTED_Energy')) < reference_energy:
                            reference_energy = float(ligand.GetProp('FR_FITTED_Energy'))

                        if Chem.SanitizeMol(ligand, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                            Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                            Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                            Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                                            Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                            Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                                            Chem.SanitizeFlags.SANITIZE_ADJUSTHS, catchErrors=True) == 0:
                            add_formal_charges(ligand)
                            w.write(ligand)
                            energies.append(float(ligand.GetProp('FR_FITTED_Energy')))
                            rmsd.append(float(ligand.GetProp('FR_FITTED_RMSD')))
                except:
                    print('problem reading the output_initial_population.sdf file                |')
                    print('|       |                                   | ', end='')

        # check if some poses have been found
        file_size = os.path.getsize(args.raw_data_dir + protein_id + '_all_poses.sdf')
        if file_size == 0 or len(energies) == 0:
            print('No pose found - rdkit sanitization problem                            |')
            continue

        # else:
        #    ligands = Chem.ForwardSDMolSupplier(args.raw_data_dir + protein_id + '_all_poses.sdf',
        #                                        removeHs=False, sanitize=False)
        #    for ligand in ligands:
        #        energies.append(float(ligand.GetProp('FR_FITTED_Energy')))
        #        rmsd.append(float(ligand.GetProp('FR_FITTED_RMSD')))

        # sort both lists by energy value.
        # print('1245: ', energies, rmsd)
        energies_rmsd = sorted(zip(energies, rmsd))
        # print('1247: ', energies_rmsd)
        # remove duplicates (same energy, same rmsd):
        energies_rmsd = list(dict.fromkeys(energies_rmsd))
        # print('1265: ', energies_rmsd)

        # extract correct and incorrect poses
        energies, rmsd = zip(*energies_rmsd)
        energies_good = []
        energies_bad = []
        rmsd_good = []
        rmsd_bad = []
        for i in range(len(energies)):
            if rmsd[i] <= args.max_rmsd_good:
                energies_good.append(energies[i])
                rmsd_good.append(rmsd[i])
            if rmsd[i] >= args.min_rmsd_bad:
                energies_bad.append(energies[i])
                rmsd_bad.append(rmsd[i])
        # print('1278: ', energies_good, rmsd_good)
        # print('1279: ', energies_bad, rmsd_bad)
        # Take only the best energies
        if len(energies_good) > args.max_number_of_poses:
            energies_good = energies_good[0:args.max_number_of_poses]
            rmsd_good = rmsd_good[0:args.max_number_of_poses]
        if len(energies_bad) > args.max_number_of_poses:
            energies_bad = energies_bad[0:args.max_number_of_poses]
            rmsd_bad = rmsd_bad[0:args.max_number_of_poses]

        # print('1288: ', energies_good, rmsd_good)
        # print('1289: ', energies_bad, rmsd_bad)

        if len(energies_good) == 0:
            print('No correct pose found                                                 |')
            continue
        if len(energies_bad) == 0 and args.debug == 0:
            print('No incorrect pose found                                               |')
            continue

        energies_good_final = []
        rmsd_good_final = []
        energies_bad_final = []
        rmsd_bad_final = []

        # We also split by RMSD.
        # lowest_E = 1000000
        # ligand_lowest_E = Chem.MolFromMolFile(args.raw_data_dir + protein_id + '_output_initial_pose.sdf',
        #                                      removeHs=True, sanitize=False)

        count_good = 0
        count_bad = 0
        initial_conf = True
        if sanitization is False:
            continue

        # As the lowest energy has gone down in the process above, some more may now be removed.
        # Now we split the all_poses.sdf file into correct (RMSD <= 2) and incorrect poses
        with Chem.SDWriter(args.raw_data_dir + protein_id + '_good_poses.sdf') as w_good:
            with Chem.SDWriter(args.raw_data_dir + protein_id + '_bad_poses.sdf') as w_bad:

                ligands = Chem.ForwardSDMolSupplier(args.raw_data_dir + protein_id + '_all_poses.sdf',
                                                    removeHs=False, sanitize=False)
                for ligand in ligands:
                    # in case the Xray minimized pose is not within threshold, we write it now to ensure we have
                    # at least one "good"
                    if initial_conf is True:
                        initial_conf = False
                        w_good.write(ligand)
                        count_good += 1
                        ligand_lowest_E = ligand
                        continue

                    rmsd = float(ligand.GetProp('FR_FITTED_RMSD'))
                    E = float(ligand.GetProp('FR_FITTED_Energy'))
                    # print('1333: ', E, rmsd, count_good, count_bad, energies_good_final, rmsd_good_final, energies_bad_final, rmsd_bad_final)
                    if rmsd in rmsd_good and E in energies_good and count_good < args.max_number_of_poses and \
                            (rmsd not in rmsd_good_final or E not in energies_good_final):
                        w_good.write(ligand)
                        energies_good_final.append(E)
                        rmsd_good_final.append(rmsd)
                        # print('1340 (good): ', rmsd_good_final, energies_good_final)
                        count_good += 1
                    elif rmsd in rmsd_bad and E in energies_bad and count_bad < args.max_number_of_poses and \
                            (rmsd not in rmsd_bad_final or E not in energies_bad_final):
                        w_bad.write(ligand)
                        energies_bad_final.append(E)
                        rmsd_bad_final.append(rmsd)
                        # print('1346 (bad): ', rmsd_bad_final, energies_bad_final)
                        count_bad += 1
                    # else:
                    #    print('1349: ', rmsd, E, ' already seen')

                    if count_good >= args.max_number_of_poses and count_bad >= args.max_number_of_poses:
                        break

        # remove the file containing all the poses.
        os.remove(args.raw_data_dir + protein_id + '_all_poses.sdf')

        # Data augmentation (scrambling atom list) for the subsets that are smaller than the expected number of poses
        # Scrambling cannot be done if we want pairs (same atom order - same graph needed)
        tempWritten = False
        if count_good < args.max_number_of_poses:
            with Chem.SDWriter(args.raw_data_dir + protein_id + '_temp.sdf') as w_temp_good:
                while count_good < args.max_number_of_poses and count_good > 0:
                    # we make a temporary copy of the file (temp.sdf) and append some additional systems.
                    if tempWritten is False:
                        tempWritten = True
                        ligands = Chem.ForwardSDMolSupplier(args.raw_data_dir + protein_id + '_good_poses.sdf',
                                                            removeHs=False, sanitize=False)
                        for ligand in ligands:
                            w_temp_good.write(ligand)

                    # Now append new structures
                    ligands = Chem.ForwardSDMolSupplier(args.raw_data_dir + protein_id + '_good_poses.sdf',
                                                        removeHs=False, sanitize=False)
                    for ligand in ligands:
                        if args.good_and_bad_pairs is False and args.scrambling_graphs is True:
                            ligand = scramble_atom_list(ligand, rng)
                        w_temp_good.write(ligand)
                        count_good += 1
                        if count_good >= args.max_number_of_poses:
                            break

            # Now we copy the temporary file into the _good_poses.sdf file
            shutil.copyfile(args.raw_data_dir + protein_id + '_temp.sdf', args.raw_data_dir + protein_id + '_good_poses.sdf')

        # Now we do the same with incorrect poses
        tempWritten = False
        if count_bad < args.max_number_of_poses:
            with Chem.SDWriter(args.raw_data_dir + protein_id + '_temp.sdf') as w_temp_bad:

                while count_bad < args.max_number_of_poses and count_bad > 0:
                    # we make a temporary copy of the file and append some additional systems.
                    if tempWritten is False:
                        tempWritten = True
                        ligands = Chem.ForwardSDMolSupplier(args.raw_data_dir + protein_id + '_bad_poses.sdf',
                                                            removeHs=False, sanitize=False)
                        for ligand in ligands:
                            w_temp_bad.write(ligand)

                    # Now append new structures
                    ligands = Chem.ForwardSDMolSupplier(args.raw_data_dir + protein_id + '_bad_poses.sdf',
                                                        removeHs=False, sanitize=False)
                    for ligand in ligands:
                        if args.good_and_bad_pairs is False and args.scrambling_graphs is True:
                            ligand = scramble_atom_list(ligand, rng)
                        w_temp_bad.write(ligand)
                        count_bad += 1
                        if count_bad >= args.max_number_of_poses:
                            break

            shutil.copyfile(args.raw_data_dir + protein_id + '_temp.sdf', args.raw_data_dir + protein_id + '_bad_poses.sdf')

        total_count_good += count_good
        total_count_bad += count_bad
        if args.verbose > 2:
            print('%-15.0f | %-15.0f | %-15.0f | %-15.0f |' % (count_good, total_count_good, count_bad, total_count_bad), flush=True)
    print('|-------------------------------------------------------------------------------------------------------------------|')
    print('| %5s pdb files for %8.0f correct poses and %8.0f wrong poses.                                              |'
          % (protein_number, total_count_good, total_count_bad))
    print('|-------------------------------------------------------------------------------------------------------------------|')


def get_interacting_atoms(ligand, protein, args):

    interacting_atoms_lig = []
    interacting_atoms_prot = []
    neighboring_atoms = []

    if args.good_and_bad_pairs is True:
        return interacting_atoms_lig, interacting_atoms_prot

    ligand_coordinates = np.array(ligand.GetConformers()[0].GetPositions())
    protein_coordinates = np.array(protein.GetConformers()[0].GetPositions())

    cutoff_water = 1.5  # For displaced water molecules
    cutoff = args.max_interaction_distance

    # get the distance matrix (with only distance < threshold) then convert into 1/r^2
    cartesian_distance_matrix_interactions = distance_matrix(ligand_coordinates, protein_coordinates)
    cartesian_distance_matrix_interactions = (cutoff > cartesian_distance_matrix_interactions) * cartesian_distance_matrix_interactions
    cartesian_distance_matrix_interactions = (cartesian_distance_matrix_interactions > cutoff_water) * cartesian_distance_matrix_interactions

    count = 0
    for i in range(cartesian_distance_matrix_interactions.shape[0]):
        if args.entire_ligand is True:
            interacting_atoms_lig.append(i)

        for j in range(cartesian_distance_matrix_interactions.shape[1]):
            if cartesian_distance_matrix_interactions[i][j] > 0.01:
                interacting_atoms_prot.append(j)
                if args.entire_ligand is False:
                    interacting_atoms_lig.append(i)

    # removing duplicates
    interacting_atoms_lig = list(dict.fromkeys(interacting_atoms_lig))
    interacting_atoms_prot = list(dict.fromkeys(interacting_atoms_prot))

    if args.entire_ligand is False:
        adj_matrix_lig = rdmolops.GetAdjacencyMatrix(ligand)
        for num_of_layers in range(args.num_of_bonds):
            for i in range(adj_matrix_lig.shape[0]):
                for j in range(adj_matrix_lig.shape[1]):
                    if adj_matrix_lig[i][j] == 1:
                        if i in interacting_atoms_lig:
                            neighboring_atoms.append(j)
                        elif j in interacting_atoms_lig:
                            neighboring_atoms.append(i)
            interacting_atoms_lig = interacting_atoms_lig + neighboring_atoms
            neighboring_atoms.clear()

        # removing duplicates
        interacting_atoms_lig = list(dict.fromkeys(interacting_atoms_lig))

    adj_matrix_prot = rdmolops.GetAdjacencyMatrix(protein)
    for num_of_layers in range(args.num_of_bonds):
        for i in range(adj_matrix_prot.shape[0]):
            for j in range(adj_matrix_prot.shape[1]):
                if adj_matrix_prot[i][j] == 1:
                    if i in interacting_atoms_prot:
                        neighboring_atoms.append(j)
                    elif j in interacting_atoms_prot:
                        neighboring_atoms.append(i)
        interacting_atoms_prot = interacting_atoms_prot + neighboring_atoms
        neighboring_atoms.clear()

    # removing duplicates
    interacting_atoms_prot = list(dict.fromkeys(interacting_atoms_prot))

    return interacting_atoms_lig, interacting_atoms_prot


def get_interacting_pairs(ligand, protein, interacting_atoms_lig, interacting_atoms_prot, args):

    pairs = []
    if args.good_and_bad_pairs is True:
        return pairs

    ligand_coordinates = np.array(ligand.GetConformers()[0].GetPositions())
    protein_coordinates = np.array(protein.GetConformers()[0].GetPositions())

    cutoff_water = 1.5  # For displaced water molecules
    cutoff = args.max_interaction_distance

    cutoff_squared = cutoff * cutoff
    # get the distance matrix (with only distance < threshold) then convert into 1/r^2
    distance_matrix_interactions = distance_matrix(ligand_coordinates, protein_coordinates)
    distance_matrix_interactions = (cutoff > distance_matrix_interactions) * distance_matrix_interactions
    distance_matrix_interactions = (distance_matrix_interactions > cutoff_water) * distance_matrix_interactions

    # TODO: we should identify if it is a water. Otherwise we do not count the clashes with the regular protein atoms...
    for i in range(distance_matrix_interactions.shape[0]):
        if i in interacting_atoms_lig:
            for j in range(distance_matrix_interactions.shape[1]):
                if j in interacting_atoms_prot:
                    pair = []
                    if distance_matrix_interactions[i][j] > cutoff_water:
                        pair.append(i)
                        pair.append(interacting_atoms_lig.index(i))
                        pair.append(j)
                        pair.append(interacting_atoms_prot.index(j))

                        # distance_matrix_interactions[i][j] = (1.0 / distance_matrix_interactions[i][j])
                        pair.append(distance_matrix_interactions[i][j])

                        # bond distance (#7-9) 1/r^2, 1/r^6, 1/r^12
                        # d2 = distance_matrix_interactions[i][j] * distance_matrix_interactions[i][j]
                        # d6 = distance_matrix_interactions[i][j] * distance_matrix_interactions[i][j] * distance_matrix_interactions[i][j]
                        # d12 = d6 * d6

                        # pair.append(d2)
                        # pair.append(d6)
                        # pair.append(d12)

                        pairs.append(pair)

    return pairs
