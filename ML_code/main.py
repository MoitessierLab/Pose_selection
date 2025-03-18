# Author: Anne Labarre
# McGill University, Montreal, QC, Canada

import pickle
import torch
import numpy as np
import random
import os
import time

from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler, RandomSampler

from argParser import argsParser
from utils import set_cuda_visible_device, get_weights, restrict_set
from prepare_input_data import filter_set, split_set, select_mismatch_pairs, pair_good_bad_poses_substract, make_pairs_from_top_10
from models import train_LR, train_LRc, train_XGBr, train_XGBc
from plot_models import plot_docking_accuracies_with_scaling_4x4, plot_docking_accuracies_with_scaling_2x1, plot_GNN_prediction, plot_prediction_accuracies_4x4, plot_prediction_accuracies_2x1, plot_GNN_accuracies_2x1, plot_features, plot_RESI_SHAP, plot_stuff_for_ML1, plot_set_overlap, plot_violin_plot, plot_PDBBind_accuracy_hardcoded, predict_accuracy, plot_accuracy, plot_GNN_accuracies, plot_RESI_features, plot_accuracy_graph, plot_accuracy_classifier, plot_RMSDvsEnergy, plot_datasetRMSDdistribution, plot_datasetEnergyRMSDdistribution, plot_scatterplot_RMSDvsEnergy

#### REMOVE
#from print_and_plot import print_graphs, print_model_txt, print_model_only_txt
#from train_GNNic import training, testing, predict
#from usage import usage

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

#### REMOVE
    # if args.mode == 'train':
    #     print('| Checking if GPU is available                                                                                      |')
    #     gpu = torch.cuda.is_available()
    #     if gpu:
    #         gpu_name = torch.cuda.get_device_name(0)
    #         print('| %20s is available                                                                                 |' % gpu_name)
    #     else:
    #         print('| No GPU available                                                                                                  |')
    #     print('|-------------------------------------------------------------------------------------------------------------------|', flush=True)
    # 
    #     set_cuda_visible_device(args.ngpu)
    # 
    #     if gpu:
    #         cmd = set_cuda_visible_device(0)
    #         os.environ['CUDA_VISIBLE_DEVICES'] = cmd[:-1]
    #         torch.cuda.empty_cache()
    # 
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Reading and preparing the sets
    if args.mode == 'usage':
        usage()

#### REMOVE
    # elif args.mode == 'prepare_complexes':
    #     print('|-------------------------------------------------------------------------------------------------------------------|')
    #     print('| Generating protein/ligand complexes and assembling into training and testing sets.                                |', flush=True)
    #     generate_complexes(args)
    # 
    # elif args.mode == 'prepare_graphs':
    #     if args.good_and_bad_pairs is False:
    #         train_dataset = prepare_graphs(args)
    #     else:
    #         train_dataset = prepare_pairs_of_graphs(args)

# ANNE
    elif args.mode == 'filter_set':
        filter_set(args)
        #select_mismatch_pairs(args)
        #pair_good_bad_poses_substract(args)
        
    elif args.mode == 'subtract_set':
        pair_good_bad_poses_substract(args)

# ANNE
    elif args.mode == 'train_model':

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

        if args.model == 'none':
            print('| Model type must be selected. Available models are:                                                                |')
            print('|   LR      -- Linear Regression                                                                                    |')
            print('|   LRC     -- Logistic Regression                                                                                  |')
            print('|   XGBr    -- eXtreme Gradient Boosting - regression                                                               |')
            print('|   XGBc    -- eXtreme Gradient Boosting - classifier                                                               |')
        elif args.model == 'LR':
            train_LR(args)
        elif args.model == 'LRc':
            train_LRc(args)
        elif args.model == 'XGBr':
            train_XGBr(args)
        elif args.model == 'XGBc':
            train_XGBc(args)
        else:
            print('| Invalid model selection. Available models are:                                                                    |')
            print('|   LR      -- Linear Regression                                                                                    |')
            print('|   LRc      -- Logistic Regression                                                                                 |')
            print('|   XGBr    -- eXtreme Gradient Boosting - regressor                                                                |')
            print('|   XGBc    -- eXtreme Gradient Boosting - classifier                                                               |')

# ANNE
    elif args.mode == 'test_model':
        if args.model == 'none':
            print('| Model type must be selected. Available models are:                                                                |')
            print('|   LR      -- Linear Regression                                                                                    |')
            print('|   XGBr    -- eXtreme Gradient Boosting - regressor                                                                |')
            print('|   XGBr    -- eXtreme Gradient Boosting - classifier                                                               |')
        elif args.model == 'LR':
            plot_accuracy(args)
        elif args.model == 'LRc':
            #plot_accuracy(args)
            make_pairs_from_top_10(args)
        elif args.model == 'XGBr':
            plot_accuracy(args)
        elif args.model == 'XGBc':
            #plot_accuracy(args)
            #plot_accuracy_classifier(args)
            make_pairs_from_top_10(args)
        else:
            print('| Invalid model selection. Available models are:                                                                    |')
            print('|   LR      -- Linear Regression                                                                                    |')
            print('|   XGBr    -- eXtreme Gradient Boosting - regressor                                                                |')
            print('|   XGBr    -- eXtreme Gradient Boosting - classifier                                                               |')

# ANNE
    elif args.mode == 'plotstuff':

        #plot_RMSDvsEnergy(args)
        #plot_datasetRMSDdistribution(args)
        #plot_datasetEnergyRMSDdistribution(args)
        #plot_scatterplot_RMSDvsEnergy(args)
        #predict_accuracy(args)
        #plot_accuracy_graph(args)
        #plot_GNN_accuracies()
        #plot_GNN_accuracies_2x1()
        #plot_PDBBind_accuracy_hardcoded()
        #plot_violin_plot(args)
        #plot_set_overlap()
        #plot_stuff_for_ML1()
        plot_prediction_accuracies_4x4()
        #plot_prediction_accuracies_2x1()
        #plot_GNN_prediction(args)
        #plot_docking_accuracies_with_scaling_2x1()
        #plot_docking_accuracies_with_scaling_4x4()
        print("im here not to make the code crash rip")

    elif args.mode == 'plot_accuracy':
        plot_accuracy_graph(args)

    elif args.mode == 'plot_ALL':
        plot_features(args)

    elif args.mode == 'plot_RESI':
        plot_RESI_features(args)
        #plot_RESI_SHAP(args)

##### REMOVE
    # elif args.mode == 'train':
    #     # load training set
    #     # read data. data is stored in format of dictionary. Each key has information about a protein-ligand complex.
    #     print('|-------------------------------------------------------------------------------------------------------------------|')
    #     print('| Generating datasets...                                                                                            |', flush=True)
    # 
    #     if args.graph_as_input is False:
    #         with open(args.keys_dir + args.train_keys, 'rb') as fp:
    #             train_keys = pickle.load(fp)
    #         print('| Training set keys loaded                                                                                          |')
    #     else:
    #         with open(args.keys_dir + args.train_keys_graphs, 'rb') as fp:
    #             train_keys = pickle.load(fp)
    #         print('| Training set keys loaded (graphs)                                                                                 |')
    # 
    #     if args.max_num_of_systems > 0:
    #         train_keys = restrict_set(train_keys, args)
    # 
    #     # if args.graph_as_input is False:
    #     if args.good_and_bad_pairs is False:
    #         train_dataset = generate_datasets(train_keys, args)
    #     else:
    #         train_dataset = generate_datasets_pairs(train_keys, args)
    #     print('| Training dataset generated                                                                                        |', flush=True)
    # 
    #     if args.test_keys != 'none' and args.graph_as_input is False:
    #         with open(args.keys_dir + args.test_keys, 'rb') as fp:
    #             test_keys = pickle.load(fp)
    #         print('| Testing set keys loaded                                                                                           |', flush=True)
    # 
    #     if args.test_keys != 'none' and args.graph_as_input is True:
    #         with open(args.keys_dir + args.test_keys_graphs, 'rb') as fp:
    #             test_keys = pickle.load(fp)
    #         print('| Testing set keys loaded (graphs)                                                                                  |', flush=True)
    # 
    #     if args.max_num_of_systems > 0:
    #         test_keys = restrict_set(test_keys, args)
    # 
    #     if args.good_and_bad_pairs is False:
    #         test_dataset = generate_datasets(test_keys, args)
    #     else:
    #         test_dataset = generate_datasets_pairs(test_keys, args)
    #     print('| Testing dataset generated                                                                                         |', flush=True)
    # 
    #     hypers = {
    #         'batch_size': args.batch_size,
    #         'learning_rate': args.lr,
    #         'weight_decay': args.weight_decay,
    #         'scheduler_gamma': args.scheduler_gamma,
    #         'model_embedding_size': args.embedding_size,
    #         'model_gnn_layers': args.n_graph_layers,
    #         'model_fc_layers': args.n_FC_layers,
    #         'model_dropout_rate': args.dropout_rate,
    #         'model_dense_neurons': args.model_dense_neurons,
    #         'model_attention_heads': args.model_attention_heads,
    #     }
    # 
    #     train_weights, train_set_size, test_set_size = get_weights(train_keys, test_keys, args)
    #     if args.train_mode == "docking":
    #         print('| Dataset analyzed                                                                                                  |', flush=True)
    #     else:
    #         print('| Weights computed                                                                                                  |', flush=True)
    # 
    #     # In pose prediction, we want to use all the poses which have similar weights (replacement=False)
    #     if args.train_mode == "docking":
    #         train_sampler = RandomSampler(data_source=train_dataset, num_samples=len(train_weights), replacement=False)
    #     else:
    #         train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)
    # 
    #     train_loader = DataLoader(train_dataset, batch_size=hypers['batch_size'],
    #                               num_workers=args.num_workers, sampler=train_sampler)
    #     print('| Training set sampler and data loader ready                                                                        |')
    # 
    #     if args.test_keys != 'none':
    #         test_loader = DataLoader(test_dataset, batch_size=hypers['batch_size'],
    #                                  num_workers=args.num_workers, shuffle=False)
    #     else:
    #         test_loader = DataLoader(train_dataset, batch_size=hypers['batch_size'],
    #                                  num_workers=args.num_workers, shuffle=False)
    #     print('| Testing set sampler and data loader ready                                                                         |', flush=True)
    # 
    #     # training the model
    #     trained_model = training(train_dataset, hypers, train_loader, test_loader, train_set_size, test_set_size, args)
    # 
    #     # saving the final model
    #     torch.save(trained_model.state_dict(), args.save_dir + args.output + '.pth')
    # 
    #     # printing the parameters
    #     # for name, param in best_trained_model.named_parameters():
    #     #    if param.requires_grad:
    #     #        print(name, param.data)
    # 
    #     # testing the final model
    #     testing(trained_model, train_loader, test_loader, train_set_size, test_set_size, args)
    # 
    # elif args.mode == 'predict':
    #     generate_complexes_predict(args)
    #     if args.good_and_bad_pairs is False:
    #         train_dataset = prepare_graphs_predict(args)
    #     else:
    #         train_dataset = prepare_pairs_of_graphs_predict(args)
    #     print('| Graphs prepared                                                                                                   |')
    #     print('|-------------------------------------------------------------------------------------------------------------------|')
    # 
    #     with open(args.keys_dir + 'poses_keys_graph.mfi.tmp.pkl', 'rb') as fp:
    #         test_keys = pickle.load(fp)
    # 
    #     args.graph_as_input = True
    #     args.batch_size = 1
    # 
    #     print('| Pose set keys loaded (graphs)                                                                                     |', flush=True)
    # 
    #     if args.good_and_bad_pairs is False:
    #         test_dataset = generate_datasets(test_keys, args)
    #     else:
    #         test_dataset = generate_datasets_pairs(test_keys, args)
    #     print('| Pose dataset generated                                                                                            |', flush=True)
    # 
    #     test_set_size = len(test_keys)
    #     hypers = {
    #         'batch_size': args.batch_size,
    #         'learning_rate': args.lr,
    #         'weight_decay': args.weight_decay,
    #         'scheduler_gamma': args.scheduler_gamma,
    #         'model_embedding_size': args.embedding_size,
    #         'model_gnn_layers': args.n_graph_layers,
    #         'model_fc_layers': args.n_FC_layers,
    #         'model_dropout_rate': args.dropout_rate,
    #         'model_dense_neurons': args.model_dense_neurons,
    #         'model_attention_heads': args.model_attention_heads,
    #     }
    # 
    #     test_loader = DataLoader(test_dataset, batch_size=hypers['batch_size'], num_workers=args.num_workers, shuffle=False)
    #     print('| Testing set sampler and data loader ready                                                                         |', flush=True)
    # 
    #     # training the model
    #     trained_model = predict(test_dataset, hypers, test_loader, test_set_size, args)
    # 
    # # The following mode is to write graphs and models in text format
    # elif args.mode == 'prepare_complexes_and_graphs':
    #     generate_complexes_predict(args)
    # 
    #     if args.good_and_bad_pairs is False:
    #         train_dataset = prepare_graphs_predict(args)
    #     else:
    #         train_dataset = prepare_pairs_of_graphs_predict(args)
    # 
    #     args.graph_as_input = True
    #     args.batch_size = 1
    # 
    #     print('| Pose set keys loaded (graphs)                                                                                     |', flush=True)
    #     with open(args.keys_dir + 'poses_keys_graph.mfi.tmp.pkl', 'rb') as fp:
    #         test_keys = pickle.load(fp)
    # 
    #     if args.good_and_bad_pairs is False:
    #         test_dataset = generate_datasets(test_keys, args)
    #     else:
    #         test_dataset = generate_datasets_pairs(test_keys, args)
    #     print('| Pose dataset generated                                                                                            |', flush=True)
    #     hypers = {
    #         'batch_size': args.batch_size,
    #         'learning_rate': args.lr,
    #         'weight_decay': args.weight_decay,
    #         'scheduler_gamma': args.scheduler_gamma,
    #         'model_embedding_size': args.embedding_size,
    #         'model_gnn_layers': args.n_graph_layers,
    #         'model_fc_layers': args.n_FC_layers,
    #         'model_dropout_rate': args.dropout_rate,
    #         'model_dense_neurons': args.model_dense_neurons,
    #         'model_attention_heads': args.model_attention_heads,
    #     }
    #     print_graphs(test_dataset, args)
    #     print_model_txt(test_dataset, hypers, args)
    # 
    # elif args.mode == 'write_model':
    #     hypers = {
    #         'batch_size': args.batch_size,
    #         'learning_rate': args.lr,
    #         'weight_decay': args.weight_decay,
    #         'scheduler_gamma': args.scheduler_gamma,
    #         'model_embedding_size': args.embedding_size,
    #         'model_gnn_layers': args.n_graph_layers,
    #         'model_fc_layers': args.n_FC_layers,
    #         'model_dropout_rate': args.dropout_rate,
    #         'model_dense_neurons': args.model_dense_neurons,
    #         'model_attention_heads': args.model_attention_heads,
    #         'model_node_feature_size': args.node_feature_size,
    #         'model_edge_feature_size': args.edge_feature_size,
    #         'model_edge_feature2_size': args.edge_feature2_size,
    #     }
    #     print_model_only_txt(hypers, args)

    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    print('|-------------------------------------------------------------------------------------------------------------------|')
    print('| Job finished at %s                                                                               |' % s)
    print('|-------------------------------------------------------------------------------------------------------------------|')
