# Author: Nic Moitessier
# McGill University, Montreal, QC, Canada

import pickle
import torch
import numpy as np
import random
import os
import time

from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler, RandomSampler

from prepare_input_data import generate_complexes, prepare_graphs, prepare_pairs_of_graphs, split_good_bad_poses, generate_complexes_predict, \
    prepare_graphs_predict, prepare_pairs_of_graphs_predict
from prepare_dataset import generate_datasets, generate_datasets_pairs
from argParser import argsParser
from print_and_plot import print_graphs, print_model_txt, print_model_only_txt
from utils import set_cuda_visible_device, get_weights, restrict_set
from train_GNNic import training, testing, predict
from usage import usage


if __name__ == '__main__':

    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    print('|-------------------------------------------------------------------------------------------------------------------|')
    print('| %s                                                                                               |' % s)
    
    args = argsParser()

    # set the seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cpu")

    if args.mode == 'train':
        print('| Checking if GPU is available                                                                                      |')
        gpu = torch.cuda.is_available()
        if gpu:
            gpu_name = torch.cuda.get_device_name(0)
            print('| %20s is available                                                                                 |' % gpu_name)
        else:
            print('| No GPU available                                                                                                  |')
        print('|-------------------------------------------------------------------------------------------------------------------|', flush=True)
    
        set_cuda_visible_device(args.ngpu)
    
        if gpu:
            cmd = set_cuda_visible_device(0)
            os.environ['CUDA_VISIBLE_DEVICES'] = cmd[:-1]
            torch.cuda.empty_cache()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Reading and preparing the sets
    if args.mode == 'usage':
        usage()

    elif args.mode == 'prepare_complexes':
        print('|-------------------------------------------------------------------------------------------------------------------|')
        print('| Generating protein/ligand complexes and assembling into training and testing sets.                                |', flush=True)
        generate_complexes(args)

    elif args.mode == 'prepare_graphs':
        if args.good_and_bad_pairs is False:
            train_dataset = prepare_graphs(args)
        else:
            train_dataset = prepare_pairs_of_graphs(args)

    elif args.mode == 'split_good_bad_poses':
        split_good_bad_poses(args)

    elif args.mode == 'train':
        # load training set
        # read data. data is stored in format of dictionary. Each key has information about a protein-ligand complex.
        print('|-------------------------------------------------------------------------------------------------------------------|')
        print('| Generating datasets...                                                                                            |', flush=True)

        if args.graph_as_input is False:
            with open(args.keys_dir + args.train_keys, 'rb') as fp:
                train_keys = pickle.load(fp)
            print('| Training set keys loaded                                                                                          |')
        else:
            with open(args.keys_dir + args.train_keys_graphs, 'rb') as fp:
                train_keys = pickle.load(fp)
            print('| Training set keys loaded (graphs)                                                                                 |')

        if args.max_num_of_systems > 0:
            train_keys = restrict_set(train_keys, args)

        # if args.graph_as_input is False:
        if args.good_and_bad_pairs is False:
            train_dataset = generate_datasets(train_keys, args)
        else:
            train_dataset = generate_datasets_pairs(train_keys, args)
        print('| Training dataset generated                                                                                        |', flush=True)

        if args.test_keys != 'none' and args.graph_as_input is False:
            with open(args.keys_dir + args.test_keys, 'rb') as fp:
                test_keys = pickle.load(fp)
            print('| Testing set keys loaded                                                                                           |', flush=True)

        if args.test_keys != 'none' and args.graph_as_input is True:
            with open(args.keys_dir + args.test_keys_graphs, 'rb') as fp:
                test_keys = pickle.load(fp)
            print('| Testing set keys loaded (graphs)                                                                                  |', flush=True)

        if args.max_num_of_systems > 0:
            test_keys = restrict_set(test_keys, args)

        if args.good_and_bad_pairs is False:
            test_dataset = generate_datasets(test_keys, args)
        else:
            test_dataset = generate_datasets_pairs(test_keys, args)
        print('| Testing dataset generated                                                                                         |', flush=True)

        if args.val_keys != 'none':
            if args.graph_as_input is False:
                with open(args.keys_dir + args.val_keys, 'rb') as fp:
                    val_keys = pickle.load(fp)
                print('| Validation set keys loaded                                                                                        |', flush=True)

            if args.val_keys != 'none' and args.graph_as_input is True:
                with open(args.keys_dir + args.val_keys_graphs, 'rb') as fp:
                    val_keys = pickle.load(fp)
                print('| Validation set keys loaded (graphs)                                                                               |', flush=True)
            if args.max_num_of_systems > 0:
                val_keys = restrict_set(val_keys, args)

            if args.good_and_bad_pairs is False:
                val_dataset = generate_datasets(val_keys, args)
            else:
                val_dataset = generate_datasets_pairs(val_keys, args)
            print('| Validation dataset generated                                                                                      |', flush=True)
        else:
            val_dataset = test_dataset

        hypers = {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'scheduler_gamma': args.scheduler_gamma,
            'model_embedding_size': args.embedding_size,
            'model_gnn_layers': args.n_graph_layers,
            'model_fc_layers': args.n_FC_layers,
            'model_dropout_rate': args.dropout_rate,
            'model_dense_neurons': args.model_dense_neurons,
            'model_attention_heads': args.model_attention_heads,
        }

        train_weights, train_set_size, val_set_size, test_set_size = get_weights(train_keys, val_keys, test_keys, args)
        if args.train_mode == "docking":
            print('| Dataset analyzed                                                                                                  |', flush=True)
        else:
            print('| Weights computed                                                                                                  |', flush=True)

        # In pose prediction, we want to use all the poses which have similar weights (replacement=False)
        if args.train_mode == "docking":
            train_sampler = RandomSampler(data_source=train_dataset, num_samples=len(train_weights), replacement=False)
        else:
            train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=hypers['batch_size'],
                                  num_workers=args.num_workers, sampler=train_sampler)
        print('| Training set sampler and data loader ready                                                                        |')

        if args.test_keys != 'none':
            test_loader = DataLoader(test_dataset, batch_size=hypers['batch_size'],
                                     num_workers=args.num_workers, shuffle=False)
        else:
            test_loader = DataLoader(train_dataset, batch_size=hypers['batch_size'],
                                     num_workers=args.num_workers, shuffle=False)
        print('| Testing set sampler and data loader ready                                                                         |', flush=True)

        if args.val_keys != 'none':
            val_loader = DataLoader(val_dataset, batch_size=hypers['batch_size'],
                                     num_workers=args.num_workers, shuffle=False)
        else:
            val_loader = DataLoader(test_dataset, batch_size=hypers['batch_size'],
                                     num_workers=args.num_workers, shuffle=False)
        print('| Validation set sampler and data loader ready                                                                      |', flush=True)

        # training the model
        trained_model = training(train_dataset, hypers, train_loader, val_loader, test_loader, train_set_size, val_set_size, test_set_size, args)

        # saving the final model
        torch.save(trained_model.state_dict(), args.save_dir + args.output + '.pth')

        # printing the parameters
        # for name, param in best_trained_model.named_parameters():
        #    if param.requires_grad:
        #        print(name, param.data)

        # testing the final model
        testing(trained_model, train_loader, val_loader, test_loader, train_set_size, val_set_size, test_set_size, args)

    elif args.mode == 'predict':
        if generate_complexes_predict(args) is False:
            print('| Preparation of the ligand/protein complexes is incomplete. The program must exit                                  |')
        else:
            if args.good_and_bad_pairs is False:
                train_dataset = prepare_graphs_predict(args)
                print('| Graphs prepared                                                                                                   |')
            else:
                train_dataset = prepare_pairs_of_graphs_predict(args)
                print('| Graphs loaded                                                                                                     |')
            print('|-------------------------------------------------------------------------------------------------------------------|')

            with open(args.keys_dir + 'poses_keys_graph.mfi.tmp.pkl', 'rb') as fp:
                test_keys = pickle.load(fp)
            print('| Pose set keys loaded (graphs)                                                                                     |', flush=True)

            args.graph_as_input = True
            args.batch_size = 1

            if args.good_and_bad_pairs is False:
                test_dataset = generate_datasets(test_keys, args)
            else:
                test_dataset = generate_datasets_pairs(test_keys, args)
            print('| Pose dataset generated                                                                                            |', flush=True)
            print('|-------------------------------------------------------------------------------------------------------------------|')

            test_set_size = len(test_keys)
            hypers = {
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'weight_decay': args.weight_decay,
                'scheduler_gamma': args.scheduler_gamma,
                'model_embedding_size': args.embedding_size,
                'model_gnn_layers': args.n_graph_layers,
                'model_fc_layers': args.n_FC_layers,
                'model_dropout_rate': args.dropout_rate,
                'model_dense_neurons': args.model_dense_neurons,
                'model_attention_heads': args.model_attention_heads,
            }

            test_loader = DataLoader(test_dataset, batch_size=hypers['batch_size'], num_workers=args.num_workers, shuffle=False)
            print('| Testing set sampler and data loader ready                                                                         |', flush=True)
            print('|-------------------------------------------------------------------------------------------------------------------|')

            # applying the model
            trained_model = predict(test_dataset, hypers, test_loader, test_set_size, args)

    # The following mode is to write graphs and models in text format
    elif args.mode == 'prepare_complexes_and_graphs':
        generate_complexes_predict(args)

        if args.good_and_bad_pairs is False:
            train_dataset = prepare_graphs_predict(args)
        else:
            train_dataset = prepare_pairs_of_graphs_predict(args)

        args.graph_as_input = True
        args.batch_size = 1

        print('| Pose set keys loaded (graphs)                                                                                     |', flush=True)
        with open(args.keys_dir + 'poses_keys_graph.mfi.tmp.pkl', 'rb') as fp:
            test_keys = pickle.load(fp)

        if args.good_and_bad_pairs is False:
            test_dataset = generate_datasets(test_keys, args)
        else:
            test_dataset = generate_datasets_pairs(test_keys, args)
        print('| Pose dataset generated                                                                                            |', flush=True)
        hypers = {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'scheduler_gamma': args.scheduler_gamma,
            'model_embedding_size': args.embedding_size,
            'model_gnn_layers': args.n_graph_layers,
            'model_fc_layers': args.n_FC_layers,
            'model_dropout_rate': args.dropout_rate,
            'model_dense_neurons': args.model_dense_neurons,
            'model_attention_heads': args.model_attention_heads,
        }
        print_graphs(test_dataset, args)
        print_model_txt(test_dataset, hypers, args)

    elif args.mode == 'write_model':
        hypers = {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'scheduler_gamma': args.scheduler_gamma,
            'model_embedding_size': args.embedding_size,
            'model_gnn_layers': args.n_graph_layers,
            'model_fc_layers': args.n_FC_layers,
            'model_dropout_rate': args.dropout_rate,
            'model_dense_neurons': args.model_dense_neurons,
            'model_attention_heads': args.model_attention_heads,
            'model_node_feature_size': args.node_feature_size,
            'model_edge_feature_size': args.edge_feature_size,
            'model_edge_feature2_size': args.edge_feature2_size,
        }
        print_model_only_txt(hypers, args)

    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    print('|-------------------------------------------------------------------------------------------------------------------|')
    print('| Job finished at %s                                                                               |' % s)
    print('|-------------------------------------------------------------------------------------------------------------------|')
