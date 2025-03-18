# Author: Nic Moitessier
# McGill University, Montreal, QC, Canada

# import random
import torch
import numpy as np
# import os
# import seaborn as sns
import matplotlib.pyplot as plt
from GNN import GNN, GNN_pairs
# from utils import search


def plot_figure1(train_loss_all, test_loss_all, args):
    plt.plot(train_loss_all, label="training")
    plt.plot(test_loss_all, label="validation")
    plt.legend()
    plt.ylim(0.0, 2.0)
    plt.grid(axis='y')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(args.output + 'epochs_focused.pdf')


def print_prediction(preds, labels, smiles, mol_num, centers, proposed_centers, args):

    if args.mode == 'test':
        for i in range(len(preds)):
            print('| %6.0f | %-81s |  %3s  |  %3s  | %5.2f | %5.2f |' % (mol_num[i], smiles[i], proposed_centers[i],
                                                                         centers[i], labels[i], preds[i]))
    else:
        for i in range(len(preds)):
            print('| %6.0f | %-88s | %6s | %13.2f |' % (mol_num[i], smiles[i], centers[i], preds[i]))

    print('|----------------------------------------------------------------------------------------------------------------------------|')


def print_graphs(loader, args):
    device = torch.device("cpu")
    print('|-------------------------------------------------------------------------------------------------------------------|')
    print('| Writing the graphs                                                                                                |')
    print('|-------------------------------------------------------------------------------------------------------------------|')

    with open(args.output + args.graph_file, 'w') as f:
        for batch in loader:
            torch.set_printoptions(threshold=100000)
            batch.to(device)
            #print('| Writing batch')
            if args.good_and_bad_pairs is False:
                f.write('#node_features: \n')
                np.savetxt(f, batch.x.shape, fmt='%10.0f')
                np.savetxt(f, batch.x, fmt='%10.5f')
                f.write('#edge_indexes: \n')
                np.savetxt(f, batch.edge_index_pl.shape, fmt='%10.0f')
                np.savetxt(f, batch.edge_index_pl, fmt='%2.0f')
                f.write('#edge_features: \n')
                np.savetxt(f, batch.edge_attr_pl.shape, fmt='%10.0f')
                np.savetxt(f, batch.edge_attr_pl, fmt='%10.5f')
            else:
                f.write('node_features: \n')
                np.savetxt(f, batch.x.shape, fmt='%10.0f')
                np.savetxt(f, batch.x, fmt='%8.5f')
                f.write('edge_indexes_complex_#1: \n')
                np.savetxt(f, batch.edge_index_1.shape, fmt='%10.0f')
                np.savetxt(f, batch.edge_index_1, fmt='%2.0f')
                f.write('edge_features_complex_#1: \n')
                np.savetxt(f, batch.edge_attr_1.shape, fmt='%10.0f')
                np.savetxt(f, batch.edge_attr_1, fmt='%10.5f')
                f.write('edge_indexes_complex_#2: \n')
                np.savetxt(f, batch.edge_index_2.shape, fmt='%10.0f')
                np.savetxt(f, batch.edge_index_2, fmt='%2.0f')
                f.write('edge_features_complex_#2: \n')
                np.savetxt(f, batch.edge_attr_2.shape, fmt='%10.0f')
                np.savetxt(f, batch.edge_attr_2, fmt='%10.5f')
                f.write('-----------------------------------------------------\n')


def print_model_txt(test_dataset, hypers, args):
    if args.load_model != 'none':
        print('| Writing the model                                                                                                 |')
        print('|-------------------------------------------------------------------------------------------------------------------|')
        model_params = {k: v for k, v in hypers.items() if k.startswith("model_")}

        if args.good_and_bad_pairs is False:
            trained_model = GNN(feature_size=test_dataset[0].x.shape[1],
                                edge_dim_pl=test_dataset[0].edge_attr_pl.shape[1],
                                model_params=model_params,
                                args=args)
        else:
            trained_model = GNN_pairs(feature_size=test_dataset[0].x.shape[1],
                                      edge_dim_1=test_dataset[0].edge_attr_1.shape[1],
                                      edge_dim_2=test_dataset[0].edge_attr_2.shape[1],
                                      model_params=model_params,
                                      args=args)

        checkpoint = torch.load(args.model_dir + args.load_model)
        trained_model.load_state_dict(checkpoint['model_state_dict'])
        with open(args.output + args.model_txt_file, 'w') as f:
            for name, param in trained_model.named_parameters():
                #if param.requires_grad:
                f.write(name)
                f.write('\n')
                # print('param.data shape: ', param.data.shape)
                tensor_shape = list(map(int, param.data.shape))
                tensor_list = list(tensor_shape)
                tensor_dim = len(list(tensor_shape))
                f.write(str(tensor_dim))
                f.write('\n')
                for dim in range(tensor_dim):
                    f.write(str(tensor_list[dim]))
                    f.write('\n')

                # savetxt cannot take tensors with more than 2 dimensions
                if len(list(tensor_shape)) > 2:
                    for slice_2d in param.data:
                        np.savetxt(f, slice_2d, fmt='%12.8f')
                else:
                    np.savetxt(f, param.data, fmt='%12.8f')

            # below if for running_mean and running_var
            for name, param in trained_model.named_buffers():
                #if param.requires_grad:
                f.write(name)
                f.write('\n')
                # print('param.data shape: ', param.data.shape)
                tensor_shape = list(map(int, param.data.shape))
                tensor_list = list(tensor_shape)
                tensor_dim = len(list(tensor_shape))
                f.write(str(tensor_dim))
                f.write('\n')
                for dim in range(tensor_dim):
                    f.write(str(tensor_list[dim]))
                    f.write('\n')

                # savetxt cannot take tensors with more than 2 dimensions
                if len(list(tensor_shape)) > 2:
                    for slice_2d in param.data:
                        np.savetxt(f, slice_2d, fmt='%12.8f')
                elif len(list(tensor_shape)) > 0:
                    np.savetxt(f, param.data, fmt='%12.8f')


def print_model_only_txt(hypers, args):
    if args.load_model != 'none':
        print('| Writing the model                                                                                                 |')
        print('|-------------------------------------------------------------------------------------------------------------------|')
        model_params = {k: v for k, v in hypers.items() if k.startswith("model_")}

        if args.good_and_bad_pairs is False:
            trained_model = GNN(feature_size=hypers['model_node_feature_size'],
                                edge_dim_pl=hypers['model_edge_feature_size'],
                                model_params=model_params,
                                args=args)
        else:
            trained_model = GNN_pairs(feature_size=hypers['model_node_feature_size'],
                                      edge_dim_1=hypers['model_edge_feature_size'],
                                      edge_dim_2=hypers['model_edge_feature2_size'],
                                      model_params=model_params,
                                      args=args)

        checkpoint = torch.load(args.model_dir + args.load_model)
        print("checkpoint:\n", checkpoint)
        trained_model.load_state_dict(checkpoint['model_state_dict'])
        print("trained_model:\n", trained_model)
        with open(args.output + args.model_txt_file, 'w') as f:
            for name, param in trained_model.named_parameters():
                #if param.requires_grad:
                f.write(name)
                f.write('\n')
                # print('param.data shape: ', param.data.shape)
                tensor_shape = list(map(int, param.data.shape))
                tensor_list = list(tensor_shape)
                tensor_dim = len(list(tensor_shape))
                f.write(str(tensor_dim))
                f.write('\n')
                for dim in range(tensor_dim):
                    f.write(str(tensor_list[dim]))
                    f.write('\n')

                # savetxt cannot take tensors with more than 2 dimensions
                if len(list(tensor_shape)) > 2:
                    for slice_2d in param.data:
                        np.savetxt(f, slice_2d, fmt='%12.8f')
                else:
                    np.savetxt(f, param.data, fmt='%12.8f')

            # below if for running_mean and running_var
            for name, param in trained_model.named_buffers():
                #if param.requires_grad:
                f.write(name)
                f.write('\n')
                # print('param.data shape: ', param.data.shape)
                tensor_shape = list(map(int, param.data.shape))
                tensor_list = list(tensor_shape)
                tensor_dim = len(list(tensor_shape))
                f.write(str(tensor_dim))
                f.write('\n')
                for dim in range(tensor_dim):
                    f.write(str(tensor_list[dim]))
                    f.write('\n')

                # savetxt cannot take tensors with more than 2 dimensions
                if len(list(tensor_shape)) > 2:
                    for slice_2d in param.data:
                        np.savetxt(f, slice_2d, fmt='%12.8f')
                elif len(list(tensor_shape)) > 0:
                    np.savetxt(f, param.data, fmt='%12.8f')
