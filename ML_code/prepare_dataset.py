# Author: Nic Moitessier
# McGill University, Montreal, QC, Canada

import torch
import numpy as np
import time
import pickle

from torch_geometric.data import Data
from featurizer import get_node_features, get_edge_features, get_labels


# TODO: what about tautomers (same name would likely overwrite...)
class generate_datasets:

    datasets = []
    # construct a random number generator - rng
    rng = np.random.default_rng(12345)

    def __init__(self, keys, args):
        self.keys = keys
        self.args = args

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # print('prepare_set 32')
        filename = self.keys[idx]
        # print('prepare_set 34, generate datasets ', filename, self.args.graph_as_input)
        node_features = []
        if self.args.graph_as_input is False:
            with open(self.args.data_dir + filename, 'rb') as key_f:
                protein, ligand, interaction_pairs, interacting_atoms_lig, interacting_atoms_prot, mol_charge, nrot, n_atoms_lig, rmsd, \
                    EvdW, Eelec, Ehbond, Ewater, MScore, RankScore = pickle.load(key_f)

            # Get node features
            # print('prepare_set 42: \n', list(interacting_atoms_lig), ligand.GetNumAtoms())
            node_features = get_node_features(ligand, node_features, interacting_atoms_lig, 'ligand', self.args)
            # print('node_features (ligand): \n', node_features)
            # print('prepare_set 45: \n', len(complex_node_features))

            # print('prepare_set 47: \n', list(interacting_atoms_prot), protein.GetNumAtoms())
            node_features = get_node_features(protein, node_features, interacting_atoms_prot, 'protein', self.args)
            # print('node_features (ligand, protein): \n', node_features)
            # print('prepare_set 50: \n', len(complex_node_features))

            node_features = np.array(node_features)
            node_features = torch.tensor(node_features, dtype=torch.float)
            # print('node_features (ligand, protein): \n', node_features)

            # Get edge features
            # edge_indices_0, edge_features_0, \
            edge_indices_pl, edge_features_pl = \
                get_edge_features(ligand, protein, interaction_pairs, interacting_atoms_lig, interacting_atoms_prot, self.args)
            # print('edge_indices_pl: \n', edge_indices_pl)

            # Get labels info
            # if pose prediction, the value goes from 1 to 0 between ca. 1.5 and 2.5 using a sigmoid function
            # if self.args.train_mode == "docking":
            # print(list(ligand.GetPropNames()))
            # label = get_sigmoid_value(rmsd)
            # else:
            label = get_labels(1 if '_active' in filename else 0)

            # provide the first atom number (will be used to identify the start of a complex in a batch)
            node_index = torch.tensor(range(len(node_features)), dtype=torch.long)

            # print("prepare-set 68: 16 node features:   ", complex_node_features.shape[1])  #, node_index.shape)
            # print("prepare-set 69: 9 bond features:    ", complex_edge_features.shape[1])  #,
            # complex_edge_indices.shape)
            # print("prepare-set 70: charge, nrot, label: ", mol_charge.shape, nrot.shape, label.shape, '\n',
            # flush=True)
            # print("prepare-set 71: ligand name: ", filename, '\n', flush=True)
            # print("prepare-set 70: \n", mol_charge, nrot, label)
            data = Data(x=node_features,
                        # edge_index_0=edge_indices_0,
                        # edge_attr_0=edge_features_0,
                        node_index=node_index,
                        edge_index_pl=edge_indices_pl,
                        edge_attr_pl=edge_features_pl,
                        mol_charge=mol_charge,
                        nrot=nrot,
                        EvdW=EvdW,
                        Eelec=Eelec,
                        Ehbond=Ehbond,
                        Ewater=Ewater,
                        MScore=MScore,
                        RankScore=RankScore,
                        key=filename,
                        y=label)
        else:
            with open(self.args.data_dir + filename, 'rb') as key_f:
                # edge_indices_0, edge_features_0, \
                node_features, node_index, edge_indices_pl, edge_features_pl, mol_charge, nrot,\
                    EvdW, Eelec, Ehbond, Ewater, MScore, RankScore, filename, label = pickle.load(key_f)

            # print('prepare_set 40')
            # print(filename)
            # print('molcharge: ', mol_charge)
            # print('nrot: ', nrot)
            # print('EvdW: ', EvdW)
            # print('Eelec: ', Eelec)
            # print('Ehbond: ', Ehbond)
            # print('Ewater: ', Ewater)
            # print('MScore: ', MScore)
            # print('RankScore: ', RankScore)
            # torch.set_printoptions(threshold=10000)
            # print('node_features (ligand, protein): \n', node_features)
            # torch.set_printoptions(threshold=10000)
            # print('edge_features (ligand, protein): \n', edge_features_pl)

            label = get_labels(1 if '_active' in filename else 0)

            # print('label: ', label)

            # print(node_features.shape, node_features.shape[1], self.args.atom_feature_formal_charge)
            if node_features.shape[1] > 27:
                # Features 12: HBA/HBD (#26-27)
                if self.args.atom_feature_HBA_HBD is False:
                    node_features = torch.cat([node_features[:, 0:25], node_features[:, 27:]], dim=1)

                # Features 11: Atom formal charge (#23-25)
                if self.args.atom_feature_formal_charge is False:
                    node_features = torch.cat([node_features[:, 0:22], node_features[:, 25:]], dim=1)

                # print("prepare set 102: ", node_features.shape)
                # print("\n", node_features)
                # Features 10: Number of hydrogen (#20-22)
                if self.args.atom_feature_number_of_Hs is False:
                    node_features = torch.cat([node_features[:, 0:19], node_features[:, 22:]], dim=1)
                # print("prepare set 107: ", node_features.shape)
                # print("\n", node_features)

                # Features 9: Aromaticity (#19)
                if self.args.atom_feature_aromaticity is False:
                    node_features = torch.cat([node_features[:, 0:18], node_features[:, 19:]], dim=1)
                # print("prepare set 113: ", node_features.shape)
                # print("\n", node_features)

                # Feature 8: contribution to TPSA (#18)
                if self.args.atom_feature_TPSA is False:
                    # test = node_features[:, :]
                    # print("prepare set 119: ", test.shape)
                    # test = node_features[:, 0:17]
                    # print("prepare set 121: ", test.shape)
                    node_features = torch.cat([node_features[:, 0:17], node_features[:, 18:]], dim=1)
                # print("prepare set 119: ", node_features.shape)
                # print("\n", node_features)

                # Feature 7: molecular refractivity (#17)
                if self.args.atom_feature_MR is False:
                    node_features = torch.cat([node_features[:, 0:16], node_features[:, 17:]], dim=1)

                # Feature 6: contribution to logP (#16)
                if self.args.atom_feature_logP is False:
                    node_features = torch.cat([node_features[:, 0:15], node_features[:, 16:]], dim=1)

                # Feature 5: atom size (#15)
                if self.args.atom_feature_atom_size is False:
                    node_features = torch.cat([node_features[:, 0:14], node_features[:, 15:]], dim=1)

                # Feature 4: partial charge (#14)
                if self.args.atom_feature_partial_charge is False:
                    node_features = torch.cat([node_features[:, 0:13], node_features[:, 14:]], dim=1)

                # Feature 3: metal (#13)
                if self.args.atom_feature_metal is False:
                    node_features = torch.cat([node_features[:, 0:12], node_features[:, 13:]], dim=1)

                # Feature 2: electronegativity (#12)
                if self.args.atom_feature_electronegativity is False:
                    node_features = torch.cat([node_features[:, 0:11], node_features[:, 12:]], dim=1)

                # Feature 1: element (#1 -11)
                if self.args.atom_feature_element is False:
                    node_features = node_features[:, 11:]

                # bond distance (#7-10) - 1/r, 1/r2, 1/r6, 1/r12
                # or 1 if dist < 5, 1 if dist < 4, 1 if dist < 3
                if self.args.LJ_or_one_hot is False:
                    for i in range(edge_features_pl.shape[0]):
                        dist = 100
                        if edge_features_pl[i, 6] > 0:
                            dist = edge_features_pl[i, 6]

                            edge_features_pl[i, 6] = (dist < 7.0 and dist >= 4.5)
                            edge_features_pl[i, 7] = (dist < 4.5 and dist >= 3.5)
                            edge_features_pl[i, 8] = (dist < 3.5 and dist > 3.0)
                            edge_features_pl[i, 9] = dist < 3.0

            # torch.set_printoptions(threshold=10000)
            # print('node_features final (ligand, protein): \n', node_features)
            # torch.set_printoptions(threshold=10000)
            # print('edge_features final (ligand, protein): \n', edge_features_pl, flush=True)
            # print("prepare set 162: ", filename, node_features, edge_features_pl)
            # print("prepare set 174: ", node_features.shape, edge_features_pl.shape)
            # print("\n", node_features, "\n", edge_features_pl)
            data = Data(x=node_features,
                        # edge_index_0=edge_indices_0,
                        # edge_attr_0=edge_features_0,
                        node_index=node_index,
                        edge_index_pl=edge_indices_pl,
                        edge_attr_pl=edge_features_pl,
                        mol_charge=mol_charge,
                        nrot=nrot,
                        EvdW=EvdW,
                        Eelec=Eelec,
                        Ehbond=Ehbond,
                        Ewater=Ewater,
                        MScore=MScore,
                        RankScore=RankScore,
                        key=filename,
                        y=label)

        return data


class generate_datasets_pairs:

    datasets = []
    # construct a random number generator - rng
    rng = np.random.default_rng(12345)

    def __init__(self, keys, args):
        self.keys = keys
        self.args = args

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # print('prepare_set 32')
        filename = self.keys[idx]

        # print('prepare_set 34, generate datasets ', filename, self.args.graph_as_input)
        node_features = []
        if self.args.graph_as_input is False:
            #TODO: this section must be updated to include pairs.
            with open(self.args.data_dir + filename, 'rb') as key_f:
                protein, ligand, interaction_pairs, interacting_atoms_lig, interacting_atoms_prot, mol_charge, nrot, n_atoms_lig, rmsd, \
                    EvdW, Eelec, Ehbond, Ewater, MScore, RankScore = pickle.load(key_f)

            # Get node features
            # print('prepare_set 42: \n', list(interacting_atoms_lig), ligand.GetNumAtoms())
            node_features = get_node_features(ligand, node_features, interacting_atoms_lig, 'ligand', self.args)
            # print('node_features (ligand): \n', node_features)
            # print('prepare_set 45: \n', len(complex_node_features))

            # print('prepare_set 47: \n', list(interacting_atoms_prot), protein.GetNumAtoms())
            node_features = get_node_features(protein, node_features, interacting_atoms_prot, 'protein', self.args)
            # print('node_features (ligand, protein): \n', node_features)
            # print('prepare_set 50: \n', len(complex_node_features))

            node_features = np.array(node_features)
            node_features = torch.tensor(node_features, dtype=torch.float)
            # print('node_features (ligand, protein): \n', node_features)

            # Get edge features
            # edge_indices_0, edge_features_0, \
            edge_indices_pl, edge_features_pl = \
                get_edge_features(ligand, protein, interaction_pairs, interacting_atoms_lig, interacting_atoms_prot, self.args)
            # print('edge_indices_pl: \n', edge_indices_pl)

            # Get labels info
            # if pose prediction, the value goes from 1 to 0 between ca. 1.5 and 2.5 using a sigmoid function
            # if self.args.train_mode == "docking":
            # print(list(ligand.GetPropNames()))
            # label = get_sigmoid_value(rmsd)
            # else:
            label = get_labels(1 if '_active' in filename else 0)

            # provide the first atom number (will be used to identify the start of a complex in a batch)
            node_index = torch.tensor(range(len(node_features)), dtype=torch.long)

            # print("prepare-set 68: 16 node features:   ", complex_node_features.shape[1])  #, node_index.shape)
            # print("prepare-set 69: 9 bond features:    ", complex_edge_features.shape[1])  #,
            # complex_edge_indices.shape)
            # print("prepare-set 70: charge, nrot, label: ", mol_charge.shape, nrot.shape, label.shape, '\n',
            # flush=True)
            # print("prepare-set 71: ligand name: ", filename, '\n', flush=True)
            # print("prepare-set 70: \n", mol_charge, nrot, label)
            data = Data(x=node_features,
                        # edge_index_0=edge_indices_0,
                        # edge_attr_0=edge_features_0,
                        node_index=node_index,
                        edge_index_pl=edge_indices_pl,
                        edge_attr_pl=edge_features_pl,
                        mol_charge=mol_charge,
                        nrot=nrot,
                        EvdW=EvdW,
                        Eelec=Eelec,
                        Ehbond=Ehbond,
                        Ewater=Ewater,
                        MScore=MScore,
                        RankScore=RankScore,
                        key=filename,
                        y=label)
        else:
            with open(self.args.graphs_dir + filename, 'rb') as key_f:
                # edge_indices_0, edge_features_0, \
                node_features, node_index, edge_indices_1, edge_features_1, edge_indices_2, edge_features_2, mol_charge_1, nrot_1, \
                    EvdW_1, Eelec_1, Ehbond_1, Ewater_1, MScore_1, RankScore_1, \
                    EvdW_2, Eelec_2, Ehbond_2, Ewater_2, MScore_2, RankScore_2, \
                    filename_1, label = pickle.load(key_f)
            # print('prepare dataset 312: ', edge_features_1.shape[1])
            # if self.args.debug == 1:
            #    print('prepare_dataset 310')
            #    print(filename)
            #    print('molcharge: ', mol_charge_1)
            #    print('nrot: ', nrot_1)
            #    print('EvdW: ', EvdW_1, EvdW_2)
            #    print('Eelec: ', Eelec_1, Eelec_2)
            #    print('Ehbond: ', Ehbond_1, Ehbond_2)
            #    print('Ewater: ', Ewater_1, Ewater_2)
            #    print('MScore: ', MScore_1, MScore_2)
            #    print('RankScore: ', RankScore_1, RankScore_2)
            #    torch.set_printoptions(threshold=10000)
            #    print('node_features (ligand, protein): \n', node_features)
            #    print('node_features (ligand, protein): \n', node_features[:, 0:27])
            #    torch.set_printoptions(threshold=10000)
            #    print('edge_features #1 (ligand, protein): \n', edge_features_1)
            #    print('edge_features #2 (ligand, protein): \n', edge_features_2)
            #    print('edge_indices #1 (ligand, protein): \n', edge_indices_1)
            #    print('edge_indices #2 (ligand, protein): \n', edge_indices_2)
            #    sys.exit()
            #label1 = label
            #label2 = get_labels(1 if '_1_graph.pkl' in filename else 0)

            #print('prepare_dataset 332: ', label1, label2, filename, filename_1)
            # print(node_features.shape, node_features.shape[1], self.args.atom_feature_formal_charge)
            if node_features.shape[1] >= 27:
                #TODO: this should be removed after I reprepare the set without these last 2 features (ligand or protein)
                # I remove the last two
                node_features = node_features[:, 0:27]
                # Features 12: HBA/HBD (#26-27)
                if self.args.atom_feature_HBA_HBD is False:
                    node_features = node_features[:, 0:25]

                # Features 11: Atom formal charge (#23-25)
                if self.args.atom_feature_formal_charge is False:
                    node_features = torch.cat([node_features[:, 0:22], node_features[:, 25:]], dim=1)

                # Features 10: Number of hydrogen (#20-22)
                if self.args.atom_feature_number_of_Hs is False:
                    node_features = torch.cat([node_features[:, 0:19], node_features[:, 22:]], dim=1)

                # Features 9: Aromaticity (#19)
                if self.args.atom_feature_aromaticity is False:
                    node_features = torch.cat([node_features[:, 0:18], node_features[:, 19:]], dim=1)

                # Feature 8: contribution to TPSA (#18)
                if self.args.atom_feature_TPSA is False:
                    node_features = torch.cat([node_features[:, 0:17], node_features[:, 18:]], dim=1)

                # Feature 7: molecular refractivity (#17)
                if self.args.atom_feature_MR is False:
                    node_features = torch.cat([node_features[:, 0:16], node_features[:, 17:]], dim=1)

                # Feature 6: contribution to logP (#16)
                if self.args.atom_feature_logP is False:
                    node_features = torch.cat([node_features[:, 0:15], node_features[:, 16:]], dim=1)

                # Feature 5: atom size (#15)
                if self.args.atom_feature_atom_size is False:
                    node_features = torch.cat([node_features[:, 0:14], node_features[:, 15:]], dim=1)

                # Feature 4: partial charge (#14)
                if self.args.atom_feature_partial_charge is False:
                    node_features = torch.cat([node_features[:, 0:13], node_features[:, 14:]], dim=1)

                # Feature 3: metal (#13)
                if self.args.atom_feature_metal is False:
                    node_features = torch.cat([node_features[:, 0:12], node_features[:, 13:]], dim=1)

                # Feature 2: electronegativity (#12)
                if self.args.atom_feature_electronegativity is False:
                    node_features = torch.cat([node_features[:, 0:11], node_features[:, 12:]], dim=1)

                # Feature 1: element (#1 -11)
                if self.args.atom_feature_element is False:
                    node_features = node_features[:, 11:]
            # TODO remove this section when done with this testing
            #  I added this as the one we are testing often is the following (even if not all have been computed). The 11th is the metal so we keep it
            # and remove only the first 10
            elif node_features.shape[1] >= 20:
                # Feature 1: element (#1 -11)
                if self.args.atom_feature_element is False:
                    node_features = node_features[:, 10:]

            # if self.args.debug == 2:
            #    print('prepare_dataset 310')
            #    print(filename)
            #    print('molcharge: ', mol_charge_1)
            #    print('nrot: ', nrot_1)
            #    print('EvdW: ', EvdW_1, EvdW_2)
            #    print('Eelec: ', Eelec_1, Eelec_2)
            #    print('Ehbond: ', Ehbond_1, Ehbond_2)
            #    print('Ewater: ', Ewater_1, Ewater_2)
            #    print('MScore: ', MScore_1, MScore_2)
            #    print('RankScore: ', RankScore_1, RankScore_2)
            #    torch.set_printoptions(threshold=10000)
            #    print('node_features (ligand, protein): \n', node_features)
            #    torch.set_printoptions(threshold=10000)
            #    print('edge_features #1 (ligand, protein): \n', edge_features_1)
            #    print('edge_features #2 (ligand, protein): \n', edge_features_2)
            #    print('edge_indices #1 (ligand, protein): \n', edge_indices_1.shape, edge_indices_1)
            #    print('edge_indices #2 (ligand, protein): \n', edge_indices_2.shape, edge_indices_2)
            #    sys.exit()
            #torch.set_printoptions(threshold=100000)
            #print(filename)
            #print('node_features final (ligand, protein): \n', node_features.shape, '\n', node_features)
            # torch.set_printoptions(threshold=10000)
            #print('edge_features_1 final (ligand, protein): \n', edge_features_1.shape, '\n', edge_features_1, flush=True)
            #print('edge_indices_1 final (ligand, protein): \n', edge_indices_1.shape, '\n', edge_indices_1, flush=True)
            #time.sleep(25)
            data = Data(x=node_features,
                        node_index=node_index,
                        edge_index_1=edge_indices_1,
                        edge_attr_1=edge_features_1,
                        edge_index_2=edge_indices_2,
                        edge_attr_2=edge_features_2,
                        mol_charge=mol_charge_1,
                        nrot=nrot_1,
                        EvdW_1=EvdW_1,
                        Eelec_1=Eelec_1,
                        Ehbond_1=Ehbond_1,
                        Ewater_1=Ewater_1,
                        MScore_1=MScore_1,
                        RankScore_1=RankScore_1,
                        EvdW_2=EvdW_2,
                        Eelec_2=Eelec_2,
                        Ehbond_2=Ehbond_2,
                        Ewater_2=Ewater_2,
                        MScore_2=MScore_2,
                        RankScore_2=RankScore_2,
                        key=filename,
                        y=label)

        return data
