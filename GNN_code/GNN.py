# Author: Nic Moitessier
# McGill University, Montreal, QC, Canada

# import copy
import time

import torch
import torch.nn.functional as F

from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import AttentionalAggregation, GATv2Conv


# In GATv2Conv, the fc weights W(l) are initialized using Glorot uniform initialization.
# The attention weights are using xavier initialization method.
# code here: https://docs.dgl.ai/en/1.0.x/_modules/dgl/nn/pytorch/conv/gatv2conv.html#GATv2Conv
class GNN(torch.nn.Module):
    def __init__(self, feature_size,
                 edge_dim_pl, model_params, args):
        super(GNN, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        self.gnn_layers = model_params["model_gnn_layers"]
        self.dense_layers = model_params["model_fc_layers"]
        self.p = model_params["model_dropout_rate"]
        dense_neurons = model_params["model_dense_neurons"]
        n_heads = model_params["model_attention_heads"]

        # self.conv_layers_0 = ModuleList([])
        self.conv_layers_pl = ModuleList([])

        self.transf_layers = ModuleList([])
        self.bn_layers = ModuleList([])
        self.fc_layers = ModuleList([])

        # GNN Layers
        # self.conv1_0 = GATv2Conv(feature_size,
        #                         embedding_size,
        #                         heads=n_heads,
        #                         edge_dim=edge_dim_0,
        #                         dropout=self.p,
        #                         concat=True)
        self.conv1_pl = GATv2Conv(feature_size,
                                  embedding_size,
                                  heads=n_heads,
                                  edge_dim=edge_dim_pl,
                                  dropout=self.p,
                                  concat=True)

        self.transf1 = Linear(embedding_size * n_heads, embedding_size)

        self.bn1 = BatchNorm1d(embedding_size)

        for i in range(self.gnn_layers-1):
            # self.conv_layers_0.append(GATv2Conv(embedding_size,
            #                                    embedding_size,
            #                                    heads=n_heads,
            #                                    edge_dim=edge_dim_0,
            #                                    dropout=self.p,
            #                                    concat=True))

            self.conv_layers_pl.append(GATv2Conv(embedding_size,
                                                 embedding_size,
                                                 heads=n_heads,
                                                 edge_dim=edge_dim_pl,
                                                 dropout=self.p,
                                                 concat=True))

            self.transf_layers.append(Linear(embedding_size * n_heads, embedding_size))
            
            self.bn_layers.append(BatchNorm1d(embedding_size))
            
        self.att = AttentionalAggregation(Linear(embedding_size, 1))

        # Linear layers: the formal charge and NROT will be added at this stage
        self.linear1 = Linear(embedding_size + 2, dense_neurons)
        for i in range(self.dense_layers - 1):
            self.fc_layers.append(Linear(dense_neurons, int(dense_neurons/2)))
            dense_neurons = int(dense_neurons/2)
        
        self.out_layer = Linear(dense_neurons, 1)

        self.exponent_1 = torch.nn.Parameter(torch.Tensor([args.interaction_exponent_1]).float())
        self.coefficient_1 = torch.nn.Parameter(torch.Tensor([args.interaction_coefficient_1]).float())
        self.exponent_2 = torch.nn.Parameter(torch.Tensor([args.interaction_exponent_2]).float())
        self.coefficient_2 = torch.nn.Parameter(torch.Tensor([args.interaction_coefficient_2]).float())

    def forward(self, x,
                # edge_index_0, edge_attr_0,
                edge_index_pl, edge_attr_pl, mol_formal_charge, mol_nrot, batch_index):
        # At this stage, x is a single tensor containing all the atoms of all the batch molecules. The references to
        # the corresponding molecules are given in batch_index

        # x_pl = copy.deepcopy(x)
        
        # Initial GATv2Conv transformation
        # x = self.conv1_0(x, edge_index_0, edge_attr_0)
        # x = torch.relu(self.transf1(x))
        # x = self.bn1(x)

        # x_pl = self.conv1_pl(x_pl, edge_index_pl, edge_attr_pl)
        # x_pl = torch.relu(self.transf1(x_pl))
        # x_pl = self.bn1(x_pl)

        # add the parameters for the interation distances
        #edge_attr_pl[5] = torch.pow(edge_attr_pl[5], self.exponent_1)
        #edge_attr_pl[5] = torch.matmul(edge_attr_pl[5], self.coefficient_1)
        #edge_attr_pl[6] = torch.pow(edge_attr_pl[5], self.exponent_2)
        #edge_attr_pl[6] = torch.matmul(edge_attr_pl[5], self.coefficient_2)

        x = self.conv1_pl(x, edge_index_pl, edge_attr_pl)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)
        # now layers of GATv2Conv
        for i in range(self.gnn_layers-1):
            # x = self.conv_layers_0[i](x, edge_index_0, edge_attr_0)
            # x = torch.relu(self.transf_layers[i](x))
            # x = self.bn_layers[i](x)
 
            # x_pl = self.conv_layers_pl[i](x_pl, edge_index_pl, edge_attr_pl)
            # x_pl = torch.relu(self.transf_layers[i](x_pl))
            # x_pl = self.bn_layers[i](x_pl)
            x = self.conv_layers_pl[i](x, edge_index_pl, edge_attr_pl)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)

        # Attention
        x = self.att(x, batch_index)
        # x_pl = self.att(x_pl, batch_index)
        
        # Now we keep only interactions
        # x =  torch.subtract(x_pl, x)
        # print("GNN 123: ", x.shape)

        # We append formal charge to the embedding
        # formal_charge shape is [batch_size*2], we want it [batch_size, 2]
        # We first add a dimension then transpose
        mol_formal_charge = mol_formal_charge[:, None]
        torch.transpose(mol_formal_charge, 0, 1)
        x = torch.cat([x, mol_formal_charge], axis=1)

        # We do the same with NROT
        mol_nrot = mol_nrot[:, None]
        torch.transpose(mol_nrot, 0, 1)
        x = torch.cat([x, mol_nrot], axis=1)

        x = torch.relu(self.linear1(x))
        # print("GNN 110: ", x.shape)

        x = F.dropout(x, p=self.p)
        # print("GNN 113: ", x.shape)

        for i in range(self.dense_layers-1):
            x = torch.relu(self.fc_layers[i](x))

            x = F.dropout(x, p=self.p)

        x = self.out_layer(x)
        # print("GNN 121: ", x.shape)

        return x


class GNN_pairs(torch.nn.Module):
    def __init__(self, feature_size, edge_dim_1, edge_dim_2, model_params, args):
        super(GNN_pairs, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        self.gnn_layers = model_params["model_gnn_layers"]
        self.dense_layers = model_params["model_fc_layers"]
        self.p = model_params["model_dropout_rate"]
        dense_neurons = model_params["model_dense_neurons"]
        n_heads = model_params["model_attention_heads"]

        # self.embed = Linear(2*feature_size, embedding_size, bias=False)
        self.conv_layers_1 = ModuleList([])
        self.conv_layers_2 = ModuleList([])

        self.transf_layers1 = ModuleList([])
        self.bn_layers1 = ModuleList([])
        self.transf_layers2 = ModuleList([])
        self.bn_layers2 = ModuleList([])
        self.fc_layers = ModuleList([])

        # TODO: we may want to embed before entering the GAT, GNN-DTI uses:
        # in the __init__: self.embede = nn.Linear(2*N_atom_features, d_graph_layer, bias = False)
        # in forward: c_hs = self.embede(c_hs)
        # The use 140 as size for 28 atom features (size is equivalent to features for 5 atoms)
        # GNN Layers
        self.conv1_1 = GATv2Conv(feature_size,
                                 embedding_size,
                                 heads=n_heads,
                                 edge_dim=edge_dim_1,
                                 dropout=self.p,
                                 concat=True)
        self.conv1_2 = GATv2Conv(feature_size,
                                 embedding_size,
                                 heads=n_heads,
                                 edge_dim=edge_dim_2,
                                 dropout=self.p,
                                 concat=True)

        self.transf1 = Linear(embedding_size * n_heads, embedding_size)

        self.bn1 = BatchNorm1d(embedding_size)
        self.transf2 = Linear(embedding_size * n_heads, embedding_size)

        self.bn2 = BatchNorm1d(embedding_size)

        for i in range(self.gnn_layers - 1):
            self.conv_layers_1.append(GATv2Conv(embedding_size,
                                                embedding_size,
                                                heads=n_heads,
                                                edge_dim=edge_dim_1,
                                                dropout=self.p,
                                                concat=True))

            self.conv_layers_2.append(GATv2Conv(embedding_size,
                                                embedding_size,
                                                heads=n_heads,
                                                edge_dim=edge_dim_2,
                                                dropout=self.p,
                                                concat=True))

            self.transf_layers1.append(Linear(embedding_size * n_heads, embedding_size))
            self.transf_layers2.append(Linear(embedding_size * n_heads, embedding_size))

            self.bn_layers1.append(BatchNorm1d(embedding_size))
            self.bn_layers2.append(BatchNorm1d(embedding_size))

        self.att1 = AttentionalAggregation(Linear(embedding_size, 1))
        self.att2 = AttentionalAggregation(Linear(embedding_size, 1))

        # Linear layers: the formal charge and NROT will be added at this stage
        # self.linear1 = Linear(embedding_size + 2, dense_neurons)
        self.linear1 = Linear(embedding_size, dense_neurons)
        for i in range(self.dense_layers - 1):
            self.fc_layers.append(Linear(dense_neurons, int(dense_neurons / 2)))
            dense_neurons = int(dense_neurons / 2)

        self.out_layer = Linear(dense_neurons, 1)

        #self.exponent_1 = torch.nn.Parameter(torch.Tensor([args.interaction_exponent_1]).float())
        #self.coefficient_1 = torch.nn.Parameter(torch.Tensor([args.interaction_coefficient_1]).float())
        #self.exponent_2 = torch.nn.Parameter(torch.Tensor([args.interaction_exponent_2]).float())
        #self.coefficient_2 = torch.nn.Parameter(torch.Tensor([args.interaction_coefficient_2]).float())

    def forward(self, x, edge_index_1, edge_attr_1, edge_index_2, edge_attr_2, mol_formal_charge, mol_nrot, batch_index, args):

        #edge_attr_1b = edge_attr_1.clone()
        #edge_attr_2b = edge_attr_2.clone()

        # add the parameters for the interation distances
        #edge_attr_1b[5] = torch.pow(edge_attr_1[5], self.exponent_1)
        #edge_attr_1b[5] = torch.mul(edge_attr_1[5], self.coefficient_1)
        #edge_attr_1b[6] = torch.pow(edge_attr_1[6], self.exponent_2)
        #edge_attr_1b[6] = torch.mul(edge_attr_1[6], self.coefficient_2)
        #edge_attr_2b[5] = torch.pow(edge_attr_2[5], self.exponent_1)
        #edge_attr_2b[5] = torch.mul(edge_attr_2[5], self.coefficient_1)
        #edge_attr_2b[6] = torch.pow(edge_attr_2[6], self.exponent_2)
        #edge_attr_2b[6] = torch.mul(edge_attr_2[6], self.coefficient_2)
        #if args.debug == 3:
        #    torch.set_printoptions(threshold=10000)
        #    print('| GNN 262 (x): ', x)
        x_1 = self.conv1_1(x, edge_index_1, edge_attr_1)
        #if args.debug == 3:
        #    torch.set_printoptions(threshold=10000)
        #    print('| GNN 265 (x_1): ', x_1)
        #    time.sleep(5)
        x_1 = torch.relu(self.transf1(x_1))
        #if args.debug == 3:
        #    print('| GNN 268 (x_1): ', x_1)
        x_1 = self.bn1(x_1)
        #if args.debug == 3:
        #    print('| GNN 274 (x_1): ', x_1)
        x_2 = self.conv1_2(x, edge_index_2, edge_attr_2)
        x_2 = torch.relu(self.transf2(x_2))
        x_2 = self.bn2(x_2)
        #if args.debug == 3:
        #    torch.set_printoptions(threshold=10000)
        #    print('| GNN 259 (x_2): ', x_2)
        #del edge_attr_1b
        #del edge_attr_2b

        # now layers of GATv2Conv
        for i in range(self.gnn_layers - 1):
            x_1 = self.conv_layers_1[i](x_1, edge_index_1, edge_attr_1)
            x_1 = torch.relu(self.transf_layers1[i](x_1))
            x_1 = self.bn_layers1[i](x_1)
            x_2 = self.conv_layers_2[i](x_2, edge_index_2, edge_attr_2)
            x_2 = torch.relu(self.transf_layers2[i](x_2))
            x_2 = self.bn_layers2[i](x_2)

        #if args.debug == 3:
        #    torch.set_printoptions(threshold=10000)
        #    print('| GNN 272 (x_1): ', x_1)
        #    print('| GNN 273 (x_2): ', x_2)

        # Attention
        x_1 = self.att1(x_1, batch_index)
        x_2 = self.att2(x_2, batch_index)

        #if args.debug == 3:
        #torch.set_printoptions(threshold=10000, sci_mode=False)
        #print('| GNN 297 (x_1): ', x_1)
        #print('| GNN 298 (x_2): ', x_2)
        # Now we keep only interactions
        x = torch.subtract(x_2, x_1)
        #if args.debug == 3:
        #    torch.set_printoptions(threshold=10000)
        #print('| GNN 303 (x): ', x)
        #    sys.exit()
        # print("GNN 123: ", x.shape)

        #TODO: do we need this for pose prediction?
        # We append formal charge to the embedding
        # formal_charge shape is [batch_size*2], we want it [batch_size, 2]
        # We first add a dimension then transpose
        #mol_formal_charge = mol_formal_charge[:, None]
        #torch.transpose(mol_formal_charge, 0, 1)
        #x = torch.cat([x, mol_formal_charge], axis=1)

        # We do the same with NROT
        #mol_nrot = mol_nrot[:, None]
        #torch.transpose(mol_nrot, 0, 1)
        #x = torch.cat([x, mol_nrot], axis=1)

        x = torch.relu(self.linear1(x))
        # print("GNN 110: ", x.shape)

        x = F.dropout(x, p=self.p)
        # print("GNN 113: ", x.shape)

        for i in range(self.dense_layers - 1):
            x = torch.relu(self.fc_layers[i](x))
            x = F.dropout(x, p=self.p)

        x = self.out_layer(x)

        # this is needed for BCEloss as it only accepts values between 0 and 1
        if args.loss_function == 'BCE':
            x = F.sigmoid(x)
        # print("GNN 121: ", x.shape)

        return x
