# Author: Nic Moitessier
# McGill University, Montreal, QC, Canada

import argparse


def argsParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", help="learning rate", type=float, default=0.00001)
    parser.add_argument("--scheduler_gamma", help="scheduler gamma", type=float, default=0.995)
    parser.add_argument("--weight_decay", help="weight decay (L2 regularization)", type=float, default=1e-03)
    parser.add_argument("--epoch", help="epoch", type=int, default=1000)
    parser.add_argument("--ngpu", help="number of gpu", type=int, default=1)
    parser.add_argument("--batch_size", help="batch_size", type=int, default=32)
    parser.add_argument("--atom_feature_element", help="atom element", type=str2bool, default=True)
    parser.add_argument("--atom_feature_metal", help="atom element", type=str2bool, default=True)
    parser.add_argument("--atom_feature_water", help="atom from a water molecule", type=str2bool, default=False)
    parser.add_argument("--atom_feature_atom_size", help="atom size", type=str2bool, default=True)
    parser.add_argument("--atom_feature_electronegativity", help="atom electronegativity", type=str2bool, default=True)
    parser.add_argument("--atom_feature_hybridization", help="atom feature hybridization", type=str2bool, default=True)
    parser.add_argument("--atom_feature_partial_charge", help="atom charge", type=str2bool, default=True)
    parser.add_argument("--atom_feature_logP", help="atom logP", type=str2bool, default=True)
    parser.add_argument("--atom_feature_MR", help="atom MR", type=str2bool, default=True)
    parser.add_argument("--atom_feature_TPSA", help="atom TPSA", type=str2bool, default=True)
    parser.add_argument("--atom_feature_HBA_HBD", help="atom HBA/HBD", type=str2bool, default=True)
    parser.add_argument("--atom_feature_aromaticity", help="atom aromaticity", type=str2bool, default=True)
    parser.add_argument("--atom_feature_number_of_Hs", help="number of hydrogen(s) bound", type=str2bool, default=True)
    parser.add_argument("--atom_feature_formal_charge", help="number of hydrogen(s) bound", type=str2bool, default=True)
    parser.add_argument("--bond_feature_bond_order", help="bond order", type=str2bool, default=True)
    parser.add_argument("--bond_feature_conjugation", help="bond conjugation", type=str2bool, default=True)
    parser.add_argument("--bond_feature_charge_conjugation", help="bond conjugation with charge", type=str2bool, default=True)
    parser.add_argument("--interaction_exponent_1", help="exponent 1 of the LJ term", type=int, default=2)
    parser.add_argument("--interaction_coefficient_1", help="coefficient 1 of the LJ term", type=int, default=1)
    parser.add_argument("--interaction_exponent_2", help="exponent 2 of the LJ term", type=int, default=1)
    parser.add_argument("--interaction_coefficient_2", help="coefficient 2 of the LJ term", type=int, default=-1)
    parser.add_argument("--good_and_bad_pairs", help="the training will be on pairs of graphs", type=str2bool, default=True)
    parser.add_argument("--scrambling_graphs", help="data augmentation through scrambling of graph", type=str2bool, default=False)
    parser.add_argument("--LJ_or_one_hot", help="interaction descriptors", type=str2bool, default=True)
    parser.add_argument("--consider_energy", help="bond conjugation", type=str2bool, default=True)
    parser.add_argument("--max_interaction_distance", help="bond conjugation", type=float, default=8.0)
    parser.add_argument("--entire_ligand", help="entire ligand or not", type=str2bool, default=True)
    parser.add_argument("--energy_threshold", help="threshold for considering poses", type=float, default=50)
    parser.add_argument("--max_number_of_poses", help="threshold for considering poses", type=int, default=50)
    parser.add_argument("--num_workers", help="number of workers", type=int, default=6)
    parser.add_argument("--n_graph_layers", help="number of GNN layers", type=int, default=4)
    parser.add_argument("--embedding_size", help="dimension of GNN layers", type=int, default=512)
    parser.add_argument("--num_of_bonds", help="number of bonds to interacting atoms", type=int, default=4)
    parser.add_argument("--n_FC_layers", help="number of FC layers", type=int, default=3)
    parser.add_argument("--model_dense_neurons", help="dimension of FC layers", type=int, default=256)
    parser.add_argument("--model_attention_heads", help="model attention heads", type=int, default=4)
    parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default=0.1)
    parser.add_argument("--loss_function", help="loss function (MSE or BCE)", type=str, default='BCE')
    parser.add_argument("--initializer_gain", help="gain for Xavier initialization", type=float, default=1.0)
    parser.add_argument("--mode", help="split_good_bad_poses, prepare_complexes, prepare_graphs, train, usage, predict", type=str, default="usage")
    parser.add_argument("--train_mode", help="docking, scoring", type=str, default="scoring")
    parser.add_argument("--first_pose_input", help="file with first poses", type=str, default='none')
    parser.add_argument("--second_pose_input", help="file with second poses", type=str, default='none')
    parser.add_argument("--first_same_as_second", help="file with second poses", type=str2bool, default=False)
    parser.add_argument("--protein_input", help="file with protein", type=str, default="none")
    parser.add_argument("--max_rmsd_good", help="max rmsd for good poses", type=float, default=1.75)
    parser.add_argument("--min_rmsd_bad", help="min rmsd for bad poses", type=float, default=3.5)
    parser.add_argument("--raw_data_dir", help="path to data folder", type=str, default='raw_data/')
    parser.add_argument("--list_trainset", help="list of systems for train set", type=str, default='list_trainset.txt')
    parser.add_argument("--list_testset", help="list of systems for test set", type=str, default='list_testset.txt')
    parser.add_argument("--list_valset", help="list of systems for validation set", type=str, default='none')
    parser.add_argument("--graph_as_input", help="graphs or complexes as input", type=str2bool, default=False)
    parser.add_argument("--data_dir", help="path to data folder", type=str, default='data/')
    parser.add_argument("--graphs_dir", help="path to data folder", type=str, default='data_graphs/')
    parser.add_argument("--keys_dir", help="path to data folder", type=str, default='keys/')
    parser.add_argument("--save_dir", help="save directory of model parameter", type=str, default='./saved_models/')
    parser.add_argument("--model_dir", help="save directory of model parameter", type=str, default='./saved_models/')
    parser.add_argument("--train_keys", help="train keys", type=str, default='train_keys.pkl')
    parser.add_argument("--test_keys", help="test keys", type=str, default='test_keys.pkl')
    parser.add_argument("--val_keys", help="val keys", type=str, default='val_keys.pkl')
    parser.add_argument("--train_keys_graphs", help="train keys", type=str, default='train_keys_graphs.pkl')
    parser.add_argument("--test_keys_graphs", help="test keys", type=str, default='test_keys_graphs.pkl')
    parser.add_argument("--val_keys_graphs", help="val keys", type=str, default='val_keys_graphs.pkl')
    parser.add_argument("--forecaster_graphs", help="graphs from forecaster", type=str2bool, default=False)
    parser.add_argument("--output", help="output file prefix", type=str, default='output/')
    parser.add_argument("--verbose", help="amount of data output", type=int, default=0)
    parser.add_argument("--seed", help="seed for random", type=int, default=42)
    parser.add_argument("--restart", help="restart", type=str, default='none')
    parser.add_argument("--load_model", help="load model for predicting", type=str, default='none')
    parser.add_argument("--model_txt_file", help="write model as text", type=str, default='none')
    parser.add_argument("--split_train_test", help="seed for random", type=float, default=0.8)
    parser.add_argument("--max_num_of_systems", help="converting only limited number", type=int, default=0)
    parser.add_argument("--good", help="label for better poses", type=float, default=1.0)
    parser.add_argument("--bad", help="label for worse poses", type=float, default=0.0)
    parser.add_argument("--graph_file", help="file name for the graphs", type=str, default="graphs.mfi.txt")
    parser.add_argument("--debug", help="add value for debug", type=int, default=0)
    parser.add_argument("--node_feature_size", help="number of features for nodes", type=int, default=0)
    parser.add_argument("--edge_feature_size", help="number of features for edges", type=int, default=0)
    parser.add_argument("--edge_feature2_size", help="number of features for edges (complex #2)", type=int, default=0)
    parser.add_argument("--best_pose_of", help="number of features for edges (complex #2)", type=int, default=0)
    args = parser.parse_args()

    print('|-------------------------------------------------------------------------------------------------------------------|')
    print('| GNNic - Nic Moitessier                                                                                            |')
    print('| Dept of Chemistry, McGill University                                                                              |')
    print('| Montreal, QC, Canada                                                                                              |')
    print('|-------------------------------------------------------------------------------------------------------------------|')
    print('| Parameters used:                                                                                                  |')
    print('| --mode (train, prepare_complexes, usage,...):                      %-46s |' % args.mode)

    if args.mode == 'train' or args.mode == 'usage':
        print('| --train_mode (docking or scoring):                                 %-46s |' % args.train_mode)
        print('| --lr (learning rate):                                              %-46.8f |' % args.lr)
        print('| --weight_decay (weight decay, L2 regularization):                  %-46.6f |' % args.weight_decay)
        print('| --scheduler_gamma (gamma for scheduler):                           %-46.6f |' % args.scheduler_gamma)
        print('| --epoch (number of epochs):                                        %-46.0f |' % args.epoch)
        print('| --ngpu (number of gpu):                                            %-46.0f |' % args.ngpu)
        print('| --batch_size (batch_size):                                         %-46.0f |' % args.batch_size)
        print('| --graph_as_input (complexes given as pre-computed graphs):         %-46s |' % args.graph_as_input)
        print('| --num_workers (number of workers for multiple CPU usage):          %-46.0f |' % args.num_workers)
        print('| --n_graph_layer (number of GNN layers):                            %-46.0f |' % args.n_graph_layers)
        print('| --embedding_size (dimension of embedding after GNN layers):        %-46.0f |' % args.embedding_size)
        print('| --n_FC_layer (number of fully connected layers):                   %-46.0f |' % args.n_FC_layers)
        print('| --model_dense_neurons (dimension of the fully connected layers):   %-46.0f |' % args.model_dense_neurons)
        print('| --model_attention_heads (multi head attention)                     %-46.0f |' % args.model_attention_heads)
        print('| --dropout_rate (dropout_rate):                                     %-46.2f |' % args.dropout_rate)
        print('| --loss_function (MSE or BCE)                                       %-46s |' % args.loss_function)
        print('| --initializer_gain (Xavier initialization)                         %-46.2f |' % args.initializer_gain)
        print('| --output (output file name):                                       %-46s |' % args.output)
        print('| --restart (model from which to restart training):                  %-46s |' % args.restart)
        print('| --model_dir (path to the model from which to restart training):    %-46s |' % args.model_dir)
        print('| --max_num_of_systems (limit number of systems):                    %-46.0f |' % args.max_num_of_systems)

    if args.mode == 'train' or args.mode == 'usage' or args.mode == 'prepare_graphs':
        print('| --atom_feature_element (atom element):                             %-46s |' % args.atom_feature_element)
        print('| --atom_feature_atom_size (atom size):                              %-46s |' % args.atom_feature_atom_size)
        print('| --atom_feature_electronegativity (electronegativity):              %-46s |' % args.atom_feature_electronegativity)
        print('| --atom_feature_hybridization (hybridization):                      %-46s |' % args.atom_feature_hybridization)
        print('| --atom_feature_metal (metal or not):                               %-46s |' % args.atom_feature_metal)
        print('| --atom_feature_water (water or not):                               %-46s |' % args.atom_feature_water)
        print('| --atom_feature_partial_charge (atom partial charge):               %-46s |' % args.atom_feature_partial_charge)
        print('| --atom_feature_logP (atom contribution to logP):                   %-46s |' % args.atom_feature_logP)
        print('| --atom_feature_MR (atom contribution to MR):                       %-46s |' % args.atom_feature_MR)
        print('| --atom_feature_TPSA (atom contribution to TPSA):                   %-46s |' % args.atom_feature_TPSA)
        print('| --atom_feature_HBA_HBD (atom HBD and/or HBA):                      %-46s |' % args.atom_feature_HBA_HBD)
        print('| --atom_feature_aromaticity (aromaticity):                          %-46s |' % args.atom_feature_aromaticity)
        print('| --atom_feature_number_of_Hs (number of Hs bound):                  %-46s |' % args.atom_feature_number_of_Hs)
        print('| --atom_feature_formal_charge (formal charge):                      %-46s |' % args.atom_feature_formal_charge)
        print('| --bond_feature_bond_order (bond order):                            %-46s |' % args.bond_feature_bond_order)
        print('| --bond_feature_conjugation (bond conjugation):                     %-46s |' % args.bond_feature_conjugation)
        print('| --LJ_or_one_hot (Lennard Jones-like or 0/1):                       %-46s |' % args.LJ_or_one_hot)
        print('| --num_of_bonds (including atoms close to interacting atoms):       %-46.0f |' % args.num_of_bonds)
        print('| --consider_energy (includes energy terms):                         %-46.0f |' % args.consider_energy)
        print('| --interaction_exponent_1 (exponent of the first LJ term)           %-46.1f |' % args.interaction_exponent_1)
        print('| --interaction_coefficient_1 (coefficient of the first LJ term)     %-46.0f |' % args.interaction_coefficient_1)
        print('| --interaction_exponent_2 (exponent of the second LJ term)          %-46.1f |' % args.interaction_exponent_2)
        print('| --interaction_coefficient_2 (coefficient of the second LJ term)    %-46.0f |' % args.interaction_coefficient_2)

    if args.mode == 'prepare_complexes' or args.mode == 'prepare_graphs' or args.mode == 'usage':
        print('| --num_of_bonds (number of bonds to interacting atoms):             %-46.0f |' % args.num_of_bonds)
        print('| --split_train_test (split into train and test sets):               %-46.2f |' % args.split_train_test)
        print('| --max_num_of_systems (limit number of systems):                    %-46.0f |' % args.max_num_of_systems)
        print('| --consider_energy (includes energy terms):                         %-46.0f |' % args.consider_energy)
        print('| --max_interaction_distance (distance for interaction):             %-46.0f |' % args.max_interaction_distance)
        print('| --entire_ligand (entire ligand or only interacting atoms):         %-46s |' % args.entire_ligand)
        print('| --scrambling_graphs (scrambling the atom order):                   %-46s |' % args.scrambling_graphs)
        print('| --list_trainset (list of proteins for train set):                  %-46s |' % args.list_trainset)
        print('| --list_testset (list of proteins for test set):                    %-46s |' % args.list_testset)
        print('| --list_valset (list of proteins for validation set):               %-46s |' % args.list_valset)
        print('| --forecaster_graphs (graphs from Forecaster provided):             %-46s |' % args.forecaster_graphs)

    if args.mode == 'split_good_bad_poses' or args.mode == 'train' or args.mode == 'prepare_complexes' or args.mode == 'prepare_graphs' \
            or args.mode == 'usage' or args.mode == 'predict':
        print('| --good_and_bad_pairs (pairs of good and bad poses as input)        %-46.0f |' % args.good_and_bad_pairs)

    if args.mode == 'split_good_bad_poses' or args.mode == 'usage':
        print('| --energy_threshold (poses above this threshold from min Xray):     %-46.0f |' % args.energy_threshold)
        print('| --max_number_of_poses (number of poses per protein):               %-46.0f |' % args.max_number_of_poses)
        print('| --max_rmsd_good (threshold for correct poses):                     %-46.2f |' % args.max_rmsd_good)
        print('| --min_rmsd_bad (threshold for incorrect poses):                    %-46.2f |' % args.min_rmsd_bad)

    if args.mode == 'split_good_bad_poses':
        print('| --list_trainset (list of proteins for train set):                  %-46s |' % args.list_trainset)
        print('| --list_testset (list of proteins for test set):                    %-46s |' % args.list_testset)
        print('| --list_valset (list of proteins for validation set):               %-46s |' % args.list_valset)
        print('| --debug (debug mode):                                              %-46s |' % args.debug)
        print('| --max_rmsd_good (threshold for correct poses):                     %-46.2f |' % args.max_rmsd_good)
        print('| --min_rmsd_bad (threshold for incorrect poses):                    %-46.2f |' % args.min_rmsd_bad)

    if args.mode == 'prepare_graphs' or args.mode == 'predict':
        print('| --graphs_dir (path to the graph folder):                           %-46s |' % args.graphs_dir)

    if args.mode == 'predict':
        print('| --first_pose_input (file containing the first pose):               %-46s |' % args.first_pose_input)
        print('| --second_pose_input (file containing the second pose):             %-46s |' % args.second_pose_input)
        print('| --protein_input (file containing the protein):                     %-46s |' % args.protein_input)
        print('| --load_model (name of the model to load):                          %-46s |' % args.load_model)
        print('| --model_dir (path to the model):                                   %-46s |' % args.model_dir)
        print('| --forecaster_graphs (graphs from Forecaster provided):             %-46s |' % args.forecaster_graphs)
        print('| --best_pose_of (identify the best pose from a set)                 %-46s |' % args.best_pose_of)

    if args.mode == 'prepare_complexes_and_graphs':
        print('| --first_pose_input (file containing the first pose):               %-46s |' % args.first_pose_input)
        print('| --second_pose_input (file containing the second pose):             %-46s |' % args.second_pose_input)
        print('| --protein_input (file containing the protein):                     %-46s |' % args.protein_input)
        print('| --graph_file (name of the model to load):                          %-46s |' % args.graph_file)
        print('| --output (output file name):                                       %-46s |' % args.output)
        print('| --load_model (name of the model to load):                          %-46s |' % args.load_model)
        print('| --model_dir (path to the model):                                   %-46s |' % args.model_dir)
        print('| --model_txt_file (model in txt format file name):                  %-46s |' % args.output)
        print('| --forecaster_graphs (graphs from Forecaster provided):             %-46s |' % args.forecaster_graphs)
        print('| --node_feature_size                                                %-46.0f |' % args.node_feature_size)
        print('| --edge_feature_size                                                %-46.0f |' % args.edge_feature_size)
        print('| --edge_feature2_size                                               %-46.0f |' % args.edge_feature2_size)

    if args.mode == 'usage':
        print('| --debug (a non zero value will output additional information):     %-46.0f |' % args.debug)

    print('| --data_dir (path to the data folder - must be pickled):            %-46s |' % args.data_dir)
    print('| --raw_data_dir (path to the raw data folder):                      %-46s |' % args.raw_data_dir)
    print('| --keys_dir (path to the data folder - must be pickled):            %-46s |' % args.keys_dir)

    if args.mode == 'train':
        print('| --save_dir (directory where model parameters will be saved):       %-46s |' % args.save_dir)

    if args.mode != 'split_good_bad_poses':
        if args.graph_as_input is False and args.mode != 'prepare_graphs':
            print('| --train_keys (file name for the keys of the training set):         %-46s |' % args.train_keys)
            print('| --test_keys (file name for the keys of the testing set or none):   %-46s |' % args.test_keys)
            print('| --val_keys (file name for the keys of the validation set or none): %-46s |' % args.val_keys)
        else:
            print('| --train_keys_graphs:                                               %-46s |' % args.train_keys_graphs)
            print('| --test_keys_graphs:                                                %-46s |' % args.test_keys_graphs)
            print('| --val_keys_graphs:                                                 %-46s |' % args.val_keys_graphs)

    print('| --verbose (amount of output and intermediate testing):             %-46s |' % args.verbose)
    print('| --seed (seed value for random):                                    %-46.0f |' % args.seed)
    print('|----------------------------------------------------------------------------------------------------------'
          '---------|', flush=True)
    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
