# Author: Nic Moitessier
# McGill University, Montreal, QC, Canada

import copy
import torch
import numpy as np
import math
import time

from rdkit import Chem
from rdkit.Chem import rdmolops, rdMolDescriptors, AllChem
from scipy.spatial import distance_matrix

# electronegativity and hardness from Pearson: https://pubs.acs.org/doi/10.1021/ic00277a030
# from: https://chem.libretexts.org/Bookshelves/Inorganic_Chemistry/
# Inorganic_Chemistry_(LibreTexts)/03%3A_Simple_Bonding_Theory/3.02%3A_Valence_Shell_Electron-Pair_Repulsion/
# 3.2.03%3A_Electronegativity_and_Atomic_Size_Effects

# logP and MR from: https://pubs.acs.org/doi/10.1021/ci990307l
# TPSA from: https://pubs.acs.org/doi/abs/10.1021/jm000942e

electronegativity = {
    'C': 6.27,
    'N': 7.30,
    'O': 7.54,
    'F': 10.41,
    'H': 7.18,
    'Cl': 8.30,
    'S': 6.22,
    'Br': 7.59,
    'I': 6.76,
    'P': 5.62,
    'B': 4.29,
    'Si': 4.77,
    'Se': 5.89,
    'As': 5.30,
    'Zn': 4.45,
    'Fe': 4.06,
    'Mg': 3.75,
    'Mn': 3.72,
    'Al': 3.23,
    'Cu': 4.48
}

electronegativity_norm = {
    'min': 3.23,
    'max': 10.41,
    'variance': 7.18  # 10.41 - 3.23
}

# Hardness from: https://link.springer.com/article/10.1007/s00894-013-1778-z
hardness = {
    'C': 5.00,
    'N': 7.23,
    'O': 6.08,
    'F': 7.01,
    'H': 6.43,
    'S': 4.14,
    'Cl': 4.68,
    'Br': 4.22,
    'I': 3.69,
    'P': 4.88,
    'B': 4.01,
    'Si': 3.38,
    'Se': 3.87,
    'As': 4.50,
    'Zn': 4.94,
    'Fe': 3.81,
    'Mg': 3.90,
    'Mn': 3.72,
    'Al': 2.77,
    'Cu': 3.25
}

hardness_norm = {
    'min': 2.77,
    'max': 7.23,
    'variance': 4.46  # 7.23 - 2.77
}

# from https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
atom_diameter = {
    'C': 75.0,
    'N': 71.0,
    'O': 63.0,
    'F': 64.0,
    'H': 32.0,
    'S': 103.0,
    'Cl': 99.0,
    'Br': 114.0,
    'I': 133.0,
    'P': 111.0,
    'B': 85.0,
    'Si': 116.0,
    'Se': 116.0,
    'As': 121.0
}

atom_diameter_norm = {
    'min': 63.0,
    'max': 133.0,
    'variance': 70.0  # 133 - 63
}

# logP and MR from: https://pubs.acs.org/doi/10.1021/ci990307l
logP_norm = {
    'min': -1.950,
    'max': 0.8857,
    'variance': 2.8357  # 0.8857 + 1.950
}

MR_norm = {
    'min': 0.000,
    'max': 14.02,
    'variance': 14.02
}

# TPSA from: https://pubs.acs.org/doi/abs/10.1021/jm000942e
tpsa_norm = {
    'min': 0.0,  # if no contribution to PSA
    'max': 38.80,
    'variance': 38.80  # 38.80 - 0.0
}


def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_node_features(mol, mol_nodes_features, interacting_atoms, ligOrProt, args):

    # Compute atomic properties (partial charges, contribution to logP, MR and TPSA
    AllChem.ComputeGasteigerCharges(mol)
    logP_MR = rdMolDescriptors._CalcCrippenContribs(mol)  # [logP, MR]
    tpsa = rdMolDescriptors._CalcTPSAContribs(mol)  # [TPSA]

    for atom in mol.GetAtoms():
        if atom.GetIdx() in interacting_atoms or len(interacting_atoms) == 0:

            node_features = []
            # Features 1: Atomic number (#1-11)
            if args.atom_feature_element is True:
                node_features += one_hot(atom.GetSymbol(),
                                         ['B', 'C', 'N', 'O', 'F', 'Si', 'P', ('S' or 'Se'),
                                          'Cl', ('Br' or 'I'), ('Zn' or 'Mn' or 'Mg' or 'Fe')])

            # Feature 2: electronegativity (#12)
            if args.atom_feature_electronegativity is True:
                electroneg = 0.5
                if atom.GetSymbol() in electronegativity:
                    electroneg = (electronegativity[atom.GetSymbol()] - electronegativity_norm['min'])
                    electroneg /= electronegativity_norm['variance']
                node_features += [electroneg]

            # Feature 3: metal (#13)
            if args.atom_feature_metal is True:
                if atom.GetSymbol() == 'Zn' or atom.GetSymbol() == 'Mn' or atom.GetSymbol() == 'Mg' or \
                   atom.GetSymbol() == 'Fe':
                    node_features += [True]
                else:
                    node_features += [False]

            # Feature 4: partial charge (#14)
            if args.atom_feature_partial_charge is True:
                charge = atom.GetDoubleProp('_GasteigerCharge')
                if np.isnan(charge) or np.isinf(charge):
                    charge = 0.0
                charge = (charge + 1) / 2.0  # normalization assuming charges from -1 to +1
                node_features += [charge]

            # Feature 5: atom size (#15)
            if args.atom_feature_atom_size is True:
                atom_d = 0.5
                if atom.GetSymbol() in atom_diameter:
                    atom_d = (atom_diameter[atom.GetSymbol()] - atom_diameter_norm['min']) / atom_diameter_norm['variance']
                node_features += [atom_d]

            # Feature 5: Hybridization (#16-18)
            if args.atom_feature_hybridization is True:
                node_features += one_hot(atom.GetHybridization(),
                                         [Chem.rdchem.HybridizationType.SP,
                                          Chem.rdchem.HybridizationType.SP2,
                                          Chem.rdchem.HybridizationType.SP3])

            # Feature 6: contribution to logP (#19)
            if args.atom_feature_logP is True:
                logP_a = (logP_MR[atom.GetIdx()][0] - logP_norm['min']) / logP_norm['variance']
                if np.isnan(logP_a):
                    logP_a = 0.5
                node_features += [logP_a]

            # Feature 7: molecular refractivity (#20)
            if args.atom_feature_MR is True:
                MR_a = (logP_MR[atom.GetIdx()][1] - MR_norm['min']) / MR_norm['variance']
                if np.isnan(MR_a):
                    MR_a = 0.5
                node_features += [MR_a]

            # Feature 8: contribution to TPSA (#21)
            if args.atom_feature_TPSA is True:
                TPSA_a = tpsa[atom.GetIdx()] / tpsa_norm['variance']
                if np.isnan(TPSA_a):
                    TPSA_a = 0.5
                node_features += [TPSA_a]

            # Features 9: Aromaticity (#22)
            if args.atom_feature_aromaticity is True:
                node_features += [atom.GetIsAromatic()]

            # Features 6: Number of hydrogen (#23-25)
            if args.atom_feature_number_of_Hs is True:
                node_features += one_hot(atom.GetTotalNumHs(),
                                         [0, 1, (2 or 3 or 4)])
                #print('featurizer 208 (# of Hs: ', atom.GetTotalNumHs(), node_features)

            # Features 7: Atom formal charge (#26-28)
            if args.atom_feature_formal_charge is True:
                node_features += one_hot(atom.GetFormalCharge(),
                                         [-1, 0, 1])
                #print('featurizer 214 (formal charge: ', atom.GetFormalCharge(), node_features)

            # Features 8: HB donor and acceptor (#29-30)
            if args.atom_feature_HBA_HBD is True:
                node_features += is_HBA_HBD(atom)

            # Features 9: Water. We do not have this directly from the protein sdf (we recommend to generate the graph from Process
            if args.atom_feature_water is True:
                node_features += [0]

            # Append node features to matrix (total 34 features)
            mol_nodes_features.append(node_features)

    return mol_nodes_features


def get_node_features_from_graphs(mol, protein, graphProtein, graphMol, mol_nodes_features, filename, mol_number, args):

    # Atomic properties (partial charges, contribution to logP, MR and TPSA,...) are computed in Forecaster
    count = 0
    protein_atom_name = []
    ligand_atom_name = []
    # protein_group_name = []
    # print(graphMol_good)
    for line in graphMol:
        count += 1
        # print('line ', count, ': ', line, '\n')
        # print(graphMol_good[line].split('##'), '\n')
        lines = graphMol[line].split('##')
        if lines[0].startswith('atom_features') is False and lines[0].startswith('node_features') is False:
            print('| Problem with the graphs from Forecaster %73s |' % ' ')
            return False

        num_of_atoms = int(lines[1])
        num_of_features = int(lines[2])
        if mol_number == 1:
            print('| The first molecule has %3s atoms each with %2s features %58s |' % (num_of_atoms, num_of_features, ' '))

        for i in range(num_of_atoms):
            node_features = []
            # Remove the multiple white spaces
            str = lines[i + 3].replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').split(' ')
            ligand_atom_name.append(str[2])
            # print(lines[i+3])
            # print(str)
            # The first two items are old atom number (with hydrogens) and new atom number (after hydrogens are removed)
            for j in range(num_of_features):
                # print('| reading features: %s %s )' % (j, str))
                node_features += [float(str[j+3])]

            # Append node features to matrix (total 34 features)
            mol_nodes_features.append(node_features)
    # print("node_features ligand: \n", mol_nodes_features)

    # check the validity of the graph compared to the ligand after rdkit
    count_atom = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            continue
        # print('Mol ', atom.GetSymbol())#, ligand_atom_name[count_atom])
        if count_atom >= len(ligand_atom_name):
            print('| ERROR: the ligand atoms in file and graph do not match (266): ', filename, count_atom, len(ligand_atom_name))
            return False
        if atom.GetSymbol() == '':
            print('| ERROR: the ligand atoms in file and graph do not match (269): ', filename)
            return False

        if atom.GetSymbol()[0] != ligand_atom_name[count_atom][0]:
            print('| ERROR: the ligand atoms in file and graph do not match (273): ', atom.GetSymbol(), ligand_atom_name[count_atom])
            return False
        count_atom += 1

    # extract the features from the protein graph prepared with Forecaster
    for line in graphProtein:
        count += 1
        # print('line ', count, ': ', line, '\n')
        # print(graphProtein[line].split('##'), '\n')
        lines = graphProtein[line].split('##')
        if lines[0].startswith('atom_features') is False and lines[0].startswith('node_features') is False:
            print('| Problem with the graphs from Forecaster %73s |' % ' ')
            return False

        num_of_atoms = int(lines[1])
        num_of_features = int(lines[2])
        if mol_number == 1:
            print('| The protein cavity has %3s atoms each with %2s features %58s |' % (num_of_atoms, num_of_features, ' '))

        for i in range(num_of_atoms):
            node_features = []
            str = lines[i + 3].replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').split(' ')
            # print(str, '\n', lines)
            # print(len(str), str)
            protein_atom_name.append(str[2])
            # protein_group_name.append(str[3])
            # The first two items are old atom number (with hydrogens) and new atom number (after hydrogens are removed)
            # the next two are protein atom name and residue name
            for j in range(num_of_features):
                # print(j, str[j+4])
                node_features += [float(str[j+4])]

            # Append node features to matrix (total 34 features)
            mol_nodes_features.append(node_features)

    # check the validity of the graph compared to the protein after rdkit
    count_atom = 0
    for atom in protein.GetAtoms():
        # print('Protein ', atom.GetSymbol(), protein_atom_name[count_atom])
        if atom.GetSymbol() == 'H':
            continue

        # print('Mol ', atom.GetSymbol())#, ligand_atom_name[count_atom])
        if count_atom >= len(protein_atom_name):
            print('| ERROR: the ligand atoms in file and graph do not match (316): ', filename, count_atom, len(protein_atom_name))
            return False
        if atom.GetSymbol() == '':
            print('| ERROR: the ligand atoms in file and graph do not match (319): ', filename)
            return False
        if atom.GetSymbol()[0] != protein_atom_name[count_atom][0]:
            print('| ERROR: the protein atoms in file and graph do not match: ', atom.GetSymbol(), protein_atom_name[count_atom])
            return False

        count_atom += 1

    # print("node_features complex: \n", mol_nodes_features)
    # time.sleep(5)
    #for k, v in data.items():
    #    print(f'{k} --> {v}')
    return mol_nodes_features


def get_edge_features(ligand, protein, interaction_pairs, interacting_atoms_lig, interacting_atoms_prot, args):
    edges_features_0 = []
    edge_indices_0 = []
    # edges_features_pl = []
    # edge_indices_pl = []

    num_of_lig_atoms = len(interacting_atoms_lig)
    num_of_prot_atoms = len(interacting_atoms_prot)
    # print(len(interacting_atoms_lig), ligand.GetNumAtoms())

    # edge features for the ligand (atoms 1 to num_of_lig_atoms)
    for bond in ligand.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # with the if statement below only interacting atoms and atoms within num_of_bonds are included.
        if i in interacting_atoms_lig and j in interacting_atoms_lig:
            edge_features = []
            # Feature 1: Bond type (#1-4) 0 is for interactions (see below)
            if args.bond_feature_bond_order is True:
                edge_features += one_hot(bond.GetBondTypeAsDouble(),
                                         [0, 1, 1.5, (2 or 3)])

            # Feature 2: Conjugation (#5)
            if args.bond_feature_conjugation is True and args.bond_feature_charge_conjugation is False:
                edge_features.append(bond.GetIsConjugated())

            #OR
            # Feature 2: Conjugation (#5)
            if args.bond_feature_charge_conjugation is True:
                atom1 = bond.GetBeginAtomIdx()
                atom2 = bond.GetEndAtomIdx()
                central_atom = -1
                peripheral_atom = -1
                strongConjugation = 0
                weakConjugation = 0
                # if not conjugated, but one atom with a heteroatom and the other one in a double bond
                if bond.GetBondTypeAsDouble() == 1:
                    if ligand.GetAtomWithIdx(atom1).GetSymbol() == "O" or ligand.GetAtomWithIdx(atom1).GetSymbol() == "N":
                        central_atom = atom2
                        peripheral_atom = atom1
                    elif ligand.GetAtomWithIdx(atom2).GetSymbol() == "O" or ligand.GetAtomWithIdx(atom2).GetSymbol() == "N":
                        central_atom = atom1
                        peripheral_atom = atom2

                    if central_atom != -1 and ligand.GetAtomWithIdx(peripheral_atom).GetFormalCharge() == -1 and ligand.GetAtomWithIdx(
                            central_atom).GetSymbol() == "C":
                        for bond2 in ligand.GetBonds():
                            if bond2.GetBondTypeAsDouble() == 1:
                                continue
                            if bond2.GetBeginAtomIdx() == central_atom and (ligand.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol() == "O" or
                                                                            ligand.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol() == "N"):
                                strongConjugation = 1
                                break
                            elif bond2.GetEndAtomIdx() == central_atom and (ligand.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol() == "O" or
                                                                            ligand.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol() == "N"):
                                strongConjugation = 1
                                break
                    if central_atom != -1 and ligand.GetAtomWithIdx(peripheral_atom).GetFormalCharge() == 0 and ligand.GetAtomWithIdx(
                            central_atom).GetSymbol() == "C":
                        for bond2 in ligand.GetBonds():
                            if bond2.GetBondTypeAsDouble() == 1:
                                continue
                            if bond2.GetBeginAtomIdx() == central_atom and ligand.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetFormalCharge() == 1:
                                strongConjugation = 1
                                break
                            elif bond2.GetEndAtomIdx() == central_atom and ligand.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetFormalCharge() == 1:
                                strongConjugation = 1
                                break
                            if bond2.GetBeginAtomIdx() == central_atom and ligand.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetFormalCharge() == 0:
                                weakConjugation = 1
                                break
                            elif bond2.GetEndAtomIdx() == central_atom and ligand.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetFormalCharge() == 0:
                                weakConjugation = 1
                                break

                elif bond.GetBondTypeAsDouble() == 2:
                    if ligand.GetAtomWithIdx(atom1).GetSymbol() == "O" or ligand.GetAtomWithIdx(atom1).GetSymbol() == "N":
                        central_atom = atom2
                        peripheral_atom = atom1
                    elif ligand.GetAtomWithIdx(atom2).GetSymbol() == "O" or ligand.GetAtomWithIdx(atom2).GetSymbol() == "N":
                        central_atom = atom1
                        peripheral_atom = atom2

                    if central_atom != -1 and ligand.GetAtomWithIdx(peripheral_atom).GetFormalCharge() == 1 and ligand.GetAtomWithIdx(
                            central_atom).GetSymbol() == "C":
                        for bond2 in ligand.GetBonds():
                            if bond2.GetBondTypeAsDouble() != 1:
                                continue
                            # print(bond2.GetBeginAtomIdx(), bond2.GetEndAtomIdx(), mol.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol())
                            if bond2.GetBeginAtomIdx() == central_atom and (ligand.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol() == "O" or
                                                                            ligand.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol() == "N"):
                                strongConjugation = 1
                                break
                            elif bond2.GetEndAtomIdx() == central_atom and (ligand.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol() == "O" or
                                                                            ligand.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol() == "N"):
                                strongConjugation = 1
                                break
                    elif central_atom != -1 and ligand.GetAtomWithIdx(peripheral_atom).GetFormalCharge() == 0 and ligand.GetAtomWithIdx(
                            central_atom).GetSymbol() == "C":
                        for bond2 in ligand.GetBonds():
                            if bond2.GetBondTypeAsDouble() != 1:
                                continue
                            # print(central_atom, bond2.GetBeginAtomIdx(), bond2.GetEndAtomIdx(), mol.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol(),
                            #      mol.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol(), \
                            #      mol.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetFormalCharge(), mol.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetFormalCharge())
                            if bond2.GetBeginAtomIdx() == central_atom and ligand.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetFormalCharge() == -1:
                                weakConjugation = 0
                                strongConjugation = 1
                                break
                            elif bond2.GetEndAtomIdx() == central_atom and ligand.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetFormalCharge() == -1:
                                weakConjugation = 0
                                strongConjugation = 1
                                break
                            if bond2.GetBeginAtomIdx() == central_atom and ligand.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetFormalCharge() == 0 \
                                    and (ligand.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol() == "O" or
                                         ligand.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol() == "N"):
                                weakConjugation = 1
                            elif bond2.GetEndAtomIdx() == central_atom and ligand.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetFormalCharge() == 0 \
                                    and (ligand.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol() == "O" or
                                         ligand.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol() == "N"):
                                weakConjugation = 1

                if args.bond_feature_conjugation is True:
                    if args.bond_feature_charge_conjugation is False and strongConjugation == 1:
                        weakConjugation = 1
                    # if args.bond_feature_focused is False:
                    edge_features += [weakConjugation]

                if args.bond_feature_charge_conjugation is True:
                    edge_features += [strongConjugation]

            if args.bond_feature_bond_order is False and args.bond_feature_conjugation is False and args.bond_feature_charge_conjugation is False:
                edge_features.append(1)
            # bond or interaction (#6)
            # edge_features.append(0)

            # bond distance (#7)
            edge_features.append(0)
            edge_features.append(0)
            if args.LJ_or_one_hot is False:
                edge_features.append(0)
                edge_features.append(0)

            # Append edge features to matrix (twice, per direction)
            edges_features_0 += [edge_features, edge_features]

            # edge indices
            # new indices for atoms i and j are position in list of interacting atoms
            edge_indices_0 += [[interacting_atoms_lig.index(i), interacting_atoms_lig.index(j)],
                               [interacting_atoms_lig.index(j), interacting_atoms_lig.index(i)]]

    # edge features of the protein (num_of_lig_atoms+1 to num_of_lig_atoms+num_of_lig_prot)
    for bond in protein.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
    
        # with the if statement below only interacting atoms and atoms within num_of_bonds are included.
        if i in interacting_atoms_prot and j in interacting_atoms_prot:
            edge_features = []
            # Feature 1: Bond type (#1-4) 0 is for interactions (see below)
            if args.bond_feature_bond_order is True:
                edge_features += one_hot(bond.GetBondTypeAsDouble(),
                                         [0, 1, 1.5, (2 or 3)])
    
            # Feature 2: Conjugation (#5)
            if args.bond_feature_conjugation is True and args.bond_feature_charge_conjugation is False:
                edge_features.append(bond.GetIsConjugated())

            if args.bond_feature_charge_conjugation is True:
                atom1 = bond.GetBeginAtomIdx()
                atom2 = bond.GetEndAtomIdx()
                central_atom = -1
                peripheral_atom = -1
                strongConjugation = 0
                weakConjugation = 0
                # if not conjugated, but one atom with a heteroatom and the other one in a double bond
                if bond.GetBondTypeAsDouble() == 1:
                    if protein.GetAtomWithIdx(atom1).GetSymbol() == "O" or protein.GetAtomWithIdx(atom1).GetSymbol() == "N":
                        central_atom = atom2
                        peripheral_atom = atom1
                    elif protein.GetAtomWithIdx(atom2).GetSymbol() == "O" or protein.GetAtomWithIdx(atom2).GetSymbol() == "N":
                        central_atom = atom1
                        peripheral_atom = atom2

                    if central_atom != -1 and protein.GetAtomWithIdx(peripheral_atom).GetFormalCharge() == -1 and protein.GetAtomWithIdx(
                            central_atom).GetSymbol() == "C":
                        for bond2 in protein.GetBonds():
                            if bond2.GetBondTypeAsDouble() == 1:
                                continue
                            if bond2.GetBeginAtomIdx() == central_atom and (protein.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol() == "O" or
                                                                            protein.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol() == "N"):
                                strongConjugation = 1
                                break
                            elif bond2.GetEndAtomIdx() == central_atom and (protein.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol() == "O" or
                                                                            protein.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol() == "N"):
                                strongConjugation = 1
                                break
                    if central_atom != -1 and protein.GetAtomWithIdx(peripheral_atom).GetFormalCharge() == 0 and protein.GetAtomWithIdx(
                            central_atom).GetSymbol() == "C":
                        for bond2 in protein.GetBonds():
                            if bond2.GetBondTypeAsDouble() == 1:
                                continue
                            if bond2.GetBeginAtomIdx() == central_atom and protein.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetFormalCharge() == 1:
                                strongConjugation = 1
                                break
                            elif bond2.GetEndAtomIdx() == central_atom and protein.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetFormalCharge() == 1:
                                strongConjugation = 1
                                break
                            if bond2.GetBeginAtomIdx() == central_atom and protein.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetFormalCharge() == 0:
                                weakConjugation = 1
                                break
                            elif bond2.GetEndAtomIdx() == central_atom and protein.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetFormalCharge() == 0:
                                weakConjugation = 1
                                break

                elif bond.GetBondTypeAsDouble() == 2:
                    if protein.GetAtomWithIdx(atom1).GetSymbol() == "O" or protein.GetAtomWithIdx(atom1).GetSymbol() == "N":
                        central_atom = atom2
                        peripheral_atom = atom1
                    elif protein.GetAtomWithIdx(atom2).GetSymbol() == "O" or protein.GetAtomWithIdx(atom2).GetSymbol() == "N":
                        central_atom = atom1
                        peripheral_atom = atom2

                    if central_atom != -1 and protein.GetAtomWithIdx(peripheral_atom).GetFormalCharge() == 1 and protein.GetAtomWithIdx(
                            central_atom).GetSymbol() == "C":
                        for bond2 in protein.GetBonds():
                            if bond2.GetBondTypeAsDouble() != 1:
                                continue
                            # print(bond2.GetBeginAtomIdx(), bond2.GetEndAtomIdx(), mol.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol())
                            if bond2.GetBeginAtomIdx() == central_atom and (protein.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol() == "O" or
                                                                            protein.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol() == "N"):
                                strongConjugation = 1
                                break
                            elif bond2.GetEndAtomIdx() == central_atom and (protein.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol() == "O" or
                                                                            protein.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol() == "N"):
                                strongConjugation = 1
                                break
                    elif central_atom != -1 and protein.GetAtomWithIdx(peripheral_atom).GetFormalCharge() == 0 and protein.GetAtomWithIdx(
                            central_atom).GetSymbol() == "C":
                        for bond2 in protein.GetBonds():
                            if bond2.GetBondTypeAsDouble() != 1:
                                continue
                            # print(central_atom, bond2.GetBeginAtomIdx(), bond2.GetEndAtomIdx(), mol.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol(),
                            #      mol.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol(), \
                            #      mol.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetFormalCharge(), mol.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetFormalCharge())
                            if bond2.GetBeginAtomIdx() == central_atom and protein.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetFormalCharge() == -1:
                                weakConjugation = 0
                                strongConjugation = 1
                                break
                            elif bond2.GetEndAtomIdx() == central_atom and protein.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetFormalCharge() == -1:
                                weakConjugation = 0
                                strongConjugation = 1
                                break
                            if bond2.GetBeginAtomIdx() == central_atom and protein.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetFormalCharge() == 0 \
                                    and (protein.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol() == "O" or
                                         protein.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol() == "N"):
                                weakConjugation = 1
                            elif bond2.GetEndAtomIdx() == central_atom and protein.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetFormalCharge() == 0 \
                                    and (protein.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol() == "O" or
                                         protein.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol() == "N"):
                                weakConjugation = 1

                if args.bond_feature_conjugation is True:
                    if args.bond_feature_charge_conjugation is False and strongConjugation == 1:
                        weakConjugation = 1
                    # if args.bond_feature_focused is False:
                    edge_features += [weakConjugation]

                if args.bond_feature_charge_conjugation is True:
                    edge_features += [strongConjugation]

            if args.bond_feature_bond_order is False and args.bond_feature_conjugation is False and args.bond_feature_charge_conjugation is False:
                edge_features.append(1)
            # bond (0) or interaction (1) (#6)
            # edge_features.append(0)

            # bond distance (#6-7)
            edge_features.append(0)
            edge_features.append(0)
            if args.LJ_or_one_hot is False:
                edge_features.append(0)
                edge_features.append(0)
    
            # Append edge features to matrix (twice, per direction)
            edges_features_0 += [edge_features, edge_features]
    
            # edge indices
            # new indices for atoms i and j are position in list of interacting atoms
            edge_indices_0 += [[interacting_atoms_prot.index(i) + num_of_lig_atoms,
                                interacting_atoms_prot.index(j) + num_of_lig_atoms],
                               [interacting_atoms_prot.index(j) + num_of_lig_atoms,
                                interacting_atoms_prot.index(i) + num_of_lig_atoms]]

    edges_features_pl = copy.deepcopy(edges_features_0)
    edge_indices_pl = copy.deepcopy(edge_indices_0)

    # atoms in the ligand (atoms 1 to num_of_lig_atoms)
    # interacting with atoms in the protein (num_of_lig_atoms+1 to num_of_lig_atoms+num_of_lig_prot)
    for pair in interaction_pairs:
        if pair[1] in interacting_atoms_lig and pair[3] in interacting_atoms_prot:
            edge_features = []
            # print('featurizer 319\n', pair)
    
            # Features 1: bond order (#1-4)
            if args.bond_feature_bond_order is True:
                edge_features += one_hot(0,
                                         [0, 1, 1.5, (2 or 3)])
    
            # Feature 2: Conjugation (#5)
            if args.bond_feature_conjugation is True:
                edge_features.append(0)

            if args.bond_feature_bond_order is False and args.bond_feature_conjugation is False:
                edge_features.append(0)
            # bond (0) or interaction (1) (#6)
            # edge_features.append(1)
    
            if args.LJ_or_one_hot is True:
                # bond distances (#6-7)
                edge_features.append(float(pair[4]))
                edge_features.append(float(pair[4]))
            else:
                # bond distance (#6-9) < 3.0, < 4.0, <5.5, < 8.0
                dist = float(pair[4])
                if dist < 3.0:
                    edge_features.append(1)
                    edge_features.append(0)
                    edge_features.append(0)
                    edge_features.append(0)
                elif dist < 4.0:
                    edge_features.append(0)
                    edge_features.append(1)
                    edge_features.append(0)
                    edge_features.append(0)
                elif dist < 5.5:
                    edge_features.append(0)
                    edge_features.append(0)
                    edge_features.append(1)
                    edge_features.append(0)
                elif dist < 8.0:
                    edge_features.append(0)
                    edge_features.append(0)
                    edge_features.append(0)
                    edge_features.append(1)
    
            # edge indices (protein atoms are after ligand atoms in the complex)
            # The non-bonded interactions are in the top right and bottom left quadrants.
            edge_indices_pl += [[interacting_atoms_lig.index(pair[1]), interacting_atoms_prot.index(pair[3]) + num_of_lig_atoms],
                                [interacting_atoms_prot.index(pair[3]) + num_of_lig_atoms, interacting_atoms_lig.index(pair[1])]]
    
            # Append edge features to matrix (twice, per direction)
            edges_features_pl += [edge_features, edge_features]

    # edges_features_0 = np.array(edges_features_0)
    # edges_features_0 = torch.tensor(edges_features_0, dtype=torch.float32)
    # edge_indices_0 = torch.tensor(edge_indices_0)
    # edge_indices_0 = edge_indices_0.t().to(torch.long).view(2, -1)

    edges_features_pl = np.array(edges_features_pl)
    edges_features_pl = torch.tensor(edges_features_pl, dtype=torch.float32)
    edge_indices_pl = torch.tensor(edge_indices_pl)
    edge_indices_pl = edge_indices_pl.t().to(torch.long).view(2, -1)

    # edge_indices_0, edges_features_0,
    return edge_indices_pl, edges_features_pl


def get_edge_features_pairs(ligand_good, ligand_bad, protein, args):
    edges_features_good = []
    edge_indices_good = []
    edges_features_bad = []
    edge_indices_bad = []

    num_of_lig_atoms = ligand_good.GetNumAtoms()
    num_of_prot_atoms = protein.GetNumAtoms()
    # print(len(interacting_atoms_lig), ligand.GetNumAtoms())
    #TODO: remove the bond order 0 et the second distance
    # edge features for the ligand (atoms 1 to num_of_lig_atoms)
    for bond in ligand_good.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # with the if statement below only interacting atoms and atoms within num_of_bonds are included.
        edge_features = []
        # Feature 1: Bond type (#1-4) 0 is for interactions (see below)
        if args.bond_feature_bond_order is True:
            edge_features += one_hot(bond.GetBondTypeAsDouble(),
                                     [0, 1, 1.5, (2 or 3)])

        # Feature 2: Conjugation (#5)
        if args.bond_feature_conjugation is True:
            edge_features.append(bond.GetIsConjugated())

        if args.bond_feature_bond_order is False and args.bond_feature_conjugation is False:
            edge_features.append(1)

        # bond or interaction (#6)
        # edge_features.append(0)

        # bond distance (#7)
        edge_features.append(0)
        edge_features.append(0)
        if args.LJ_or_one_hot is False:
            edge_features.append(0)
            edge_features.append(0)

        # Append edge features to matrix (twice, per direction)
        edges_features_good += [edge_features, edge_features]
        edges_features_bad += [edge_features, edge_features]

        # edge indices
        # new indices for atoms i and j are position in list of interacting atoms
        edge_indices_good += [[i, j], [j, i]]
        edge_indices_bad += [[i, j], [j, i]]
    #print(edge_indices_good)
    # edge features of the protein (num_of_lig_atoms+1 to num_of_lig_atoms+num_of_lig_prot)
    for bond in protein.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # with the if statement below only interacting atoms and atoms within num_of_bonds are included.
        edge_features = []
        # Feature 1: Bond type (#1-4) 0 is for interactions (see below)
        if args.bond_feature_bond_order is True:
            edge_features += one_hot(bond.GetBondTypeAsDouble(),
                                     [0, 1, 1.5, (2 or 3)])

        # Feature 2: Conjugation (#5)
        if args.bond_feature_conjugation is True:
            edge_features.append(bond.GetIsConjugated())

        if args.bond_feature_bond_order is False and args.bond_feature_conjugation is False:
            edge_features.append(1)

        # bond (0) or interaction (1) (#6)
        # edge_features.append(0)

        # bond distance (#6-7)
        edge_features.append(0)
        edge_features.append(0)
        if args.LJ_or_one_hot is False:
            edge_features.append(0)
            edge_features.append(0)

        # Append edge features to matrix (twice, per direction)
        edges_features_good += [edge_features, edge_features]
        edges_features_bad += [edge_features, edge_features]

        # edge indices
        # new indices for atoms i and j are position in list of interacting atoms
        edge_indices_good += [[i + num_of_lig_atoms, j + num_of_lig_atoms],
                              [j + num_of_lig_atoms, i + num_of_lig_atoms]]
        edge_indices_bad += [[i + num_of_lig_atoms, j + num_of_lig_atoms],
                             [j + num_of_lig_atoms, i + num_of_lig_atoms]]

    ligand_good_coordinates = np.array(ligand_good.GetConformers()[0].GetPositions())
    ligand_bad_coordinates = np.array(ligand_bad.GetConformers()[0].GetPositions())
    protein_coordinates = np.array(protein.GetConformers()[0].GetPositions())

    # TODO: not sure this is the right way to deal with water
    cutoff_water = 1.5  # For displaced water molecules
    cutoff = args.max_interaction_distance

    # get the distance matrix (with only distance < threshold) then convert into 1/r^2
    cartesian_distance_matrix_interactions_good = distance_matrix(ligand_good_coordinates, protein_coordinates)
    cartesian_distance_matrix_interactions_good = (cutoff > cartesian_distance_matrix_interactions_good) * cartesian_distance_matrix_interactions_good
    cartesian_distance_matrix_interactions_good = (cartesian_distance_matrix_interactions_good > cutoff_water) * cartesian_distance_matrix_interactions_good
    cartesian_distance_matrix_interactions_bad = distance_matrix(ligand_bad_coordinates, protein_coordinates)
    cartesian_distance_matrix_interactions_bad = (cutoff > cartesian_distance_matrix_interactions_bad) * cartesian_distance_matrix_interactions_bad
    cartesian_distance_matrix_interactions_bad = (cartesian_distance_matrix_interactions_bad > cutoff_water) * cartesian_distance_matrix_interactions_bad

    for i in range(cartesian_distance_matrix_interactions_good.shape[0]):
        for j in range(cartesian_distance_matrix_interactions_good.shape[1]):
            if cartesian_distance_matrix_interactions_good[i][j] > cutoff_water:
                edge_features = []

                # Features 1: bond order (#1-4)
                if args.bond_feature_bond_order is True:
                    edge_features += one_hot(0,
                                             [0, 1, 1.5, (2 or 3)])

                # Feature 2: Conjugation (#5)
                if args.bond_feature_conjugation is True:
                    edge_features.append(0)

                if args.bond_feature_bond_order is False and args.bond_feature_conjugation is False:
                    edge_features.append(0)

                if args.LJ_or_one_hot is True:
                    # bond distances (#6-7)
                    edge_features.append(cartesian_distance_matrix_interactions_good[i][j])
                    edge_features.append(cartesian_distance_matrix_interactions_good[i][j])
                else:
                    # bond distance (#6-9) < 3.0, < 4.0, <5.5, < 8.0
                    dist = cartesian_distance_matrix_interactions_good
                    if dist < 3.0:
                        edge_features.append(1)
                        edge_features.append(0)
                        edge_features.append(0)
                        edge_features.append(0)
                    elif dist < 4.0:
                        edge_features.append(0)
                        edge_features.append(1)
                        edge_features.append(0)
                        edge_features.append(0)
                    elif dist < 5.5:
                        edge_features.append(0)
                        edge_features.append(0)
                        edge_features.append(1)
                        edge_features.append(0)
                    elif dist < 8.0:
                        edge_features.append(0)
                        edge_features.append(0)
                        edge_features.append(0)
                        edge_features.append(1)

                # edge indices (protein atoms are after ligand atoms in the complex)
                # The non-bonded interactions are in the top right and bottom left quadrants.
                edge_indices_good += [[i, j + num_of_lig_atoms],
                                      [j + num_of_lig_atoms, i]]

                #print('edge features: \n', i, j, edge_features)
                # Append edge features to matrix (twice, per direction)
                edges_features_good += [edge_features, edge_features]

            if cartesian_distance_matrix_interactions_bad[i][j] > cutoff_water:
                edge_features = []

                # Features 1: bond order (#1-4)
                if args.bond_feature_bond_order is True:
                    edge_features += one_hot(0,
                                             [0, 1, 1.5, (2 or 3)])

                # Feature 2: Conjugation (#5)
                if args.bond_feature_conjugation is True:
                    edge_features.append(0)

                if args.bond_feature_bond_order is False and args.bond_feature_conjugation is False:
                    edge_features.append(0)

                if args.LJ_or_one_hot is True:
                    # bond distances (#6-7)
                    edge_features.append(cartesian_distance_matrix_interactions_bad[i][j])
                    edge_features.append(cartesian_distance_matrix_interactions_bad[i][j])
                else:
                    # bond distance (#6-9) < 3.0, < 4.0, <5.5, < 8.0
                    dist = cartesian_distance_matrix_interactions_bad
                    if dist < 3.0:
                        edge_features.append(1)
                        edge_features.append(0)
                        edge_features.append(0)
                        edge_features.append(0)
                    elif dist < 4.0:
                        edge_features.append(0)
                        edge_features.append(1)
                        edge_features.append(0)
                        edge_features.append(0)
                    elif dist < 5.5:
                        edge_features.append(0)
                        edge_features.append(0)
                        edge_features.append(1)
                        edge_features.append(0)
                    elif dist < 8.0:
                        edge_features.append(0)
                        edge_features.append(0)
                        edge_features.append(0)
                        edge_features.append(1)

                # edge indices (protein atoms are after ligand atoms in the complex)
                # The non-bonded interactions are in the top right and bottom left quadrants.
                edge_indices_bad += [[i, j + num_of_lig_atoms],
                                     [j + num_of_lig_atoms, i]]

                # Append edge features to matrix (twice, per direction)
                edges_features_bad += [edge_features, edge_features]

    #print("featurizer 595: ", edge_indices_good)
    edges_features_good = np.array(edges_features_good)
    edges_features_good = torch.tensor(edges_features_good, dtype=torch.float32)
    edge_indices_good = torch.tensor(edge_indices_good)
    #print("featurizer 599: ", edge_indices_good)
    edge_indices_good = edge_indices_good.t().to(torch.long).view(2, -1)
    #print("featurizer 601: ", edge_indices_good)
    edges_features_bad = np.array(edges_features_bad)
    edges_features_bad = torch.tensor(edges_features_bad, dtype=torch.float32)
    edge_indices_bad = torch.tensor(edge_indices_bad)
    edge_indices_bad = edge_indices_bad.t().to(torch.long).view(2, -1)

    # edge_indices_0, edges_features_0,
    return edge_indices_good, edges_features_good, edge_indices_bad, edges_features_bad


def get_edge_features_pairs_from_graphs(ligand_good, ligand_bad, protein, graphProtein, graphMol_bad, graphMol_good, mol_number, args):
    edges_features_good = []
    edge_indices_good = []
    edges_features_bad = []
    edge_indices_bad = []
    five_bond_features = False

    num_of_lig_atoms = 0
    for atom in ligand_good.GetAtoms():
       if atom.GetSymbol() != 'H':
           num_of_lig_atoms += 1
    # num_of_lig_atoms = ligand_good.GetNumAtoms()
    num_of_prot_atoms = protein.GetNumAtoms()

    # Identify water molecules
    water = []
    for atom in protein.GetAtoms():
        if atom.GetSymbol() == 'H':
            continue
        if atom.GetSymbol() == 'O' and (atom.GetNumExplicitHs() + atom.GetNumImplicitHs() == 2):
            water.append(1)
        else:
            # print("This is not water: ", atom.GetIdx())
            water.append(0)
    if mol_number == 1:
        print('| The protein cavity file has %3s water molecules. %64s |' % (sum(water), ' '))

    # edge features for the ligand (atoms 1 to num_of_lig_atoms)
    count = 0
    for line in graphMol_good:
        count += 1
        lines = graphMol_good[line].split('##')
        index_bonds = lines.index('bond_features:')
        if index_bonds == -1:
            print('| Problem with the graphs from Forecaster %73s |' % ' ')
            return False

        num_of_bonds = int(lines[index_bonds + 1])
        num_of_features = int(lines[index_bonds + 2])
        if mol_number == 1:
            print('| The first molecule has %3s bonds each with %2s features %58s |' % (num_of_bonds, num_of_features, ' '))

        #TODO: remove the line below:
        #num_of_features = 7
        # print(num_of_atoms, num_of_features)
        for i in range(num_of_bonds):
            edge_features = []
            str = lines[index_bonds + i + 3].replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').split(' ')
            if len(str) < 2:
                break
            atom1 = int(str[2])
            atom2 = int(str[3])

            # The first items are old and new bond numbers then old atom numbers (with hydrogens) and new atom numbers (after hydrogens are removed)
            #for j in range(num_of_features):
            #    edge_features += [float(str[j+6])]

            # We remove the first column (0 bond order) and the last one (second distance)
            if five_bond_features is True:
                for j in range(1, 6, 1):
                    edge_features += [float(str[j + 6])]
            else:
                for j in range(0, num_of_features, 1):
                    edge_features += [float(str[j+6])]

            # Append node features to matrix (total 34 features)
            # print('774', edge_features, '\n')
            edges_features_good += [edge_features, edge_features]
            edges_features_bad += [edge_features, edge_features]
            edge_indices_good += [[atom1, atom2], [atom2, atom1]]
            edge_indices_bad += [[atom1, atom2], [atom2, atom1]]
            # print(edges_features_good)

    for line in graphProtein:
        count += 1
        lines = graphProtein[line].split('##')
        index_bonds = lines.index('bond_features:')
        if index_bonds == -1:
            print('| Problem with the graphs from Forecaster %73s |' % ' ')
            return False

        num_of_bonds = int(lines[index_bonds + 1])
        num_of_features = int(lines[index_bonds + 2])
        if mol_number == 1:
            print('| The protein cavity has %3s bonds each with %2s features %58s |' % (num_of_bonds, num_of_features, ' '))
            print('| Now processing all the other poses if any              %58s |' % ' ')

        #TODO: remove the line below:
        #num_of_features = 7
        for i in range(num_of_bonds):
            edge_features = []
            str = lines[index_bonds + i + 3].replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').split(' ')
            if len(str) < 2:
                break
            atom1 = int(str[2])
            atom2 = int(str[3])
            # The first items are old and new bond numbers then old atom numbers (with hydrogens) and new atom numbers (after hydrogens are removed)
            #for j in range(num_of_features):
            #    edge_features += [float(str[j+6])]
            # We remove the first column (0 bond order) and the last one (second distance)
            if five_bond_features is True:
                for j in range(1, 6, 1):
                    edge_features += [float(str[j + 6])]

            else:
                for j in range(0, num_of_features, 1):
                    edge_features += [float(str[j+6])]

            # Append node features to matrix (total 34 features)
            edges_features_good += [edge_features, edge_features]
            edges_features_bad += [edge_features, edge_features]
            edge_indices_good += [[atom1 + num_of_lig_atoms, atom2 + num_of_lig_atoms], [atom2 + num_of_lig_atoms, atom1 + num_of_lig_atoms]]
            edge_indices_bad += [[atom1 + num_of_lig_atoms, atom2 + num_of_lig_atoms], [atom2 + num_of_lig_atoms, atom1 + num_of_lig_atoms]]

    #print('| Now we compute the intermolecular interaction distances %57s |' % ' ')
    # Now we compute interactions
    ligand_good_coordinates = np.array(ligand_good.GetConformers()[0].GetPositions())
    ligand_bad_coordinates = np.array(ligand_bad.GetConformers()[0].GetPositions())
    protein_coordinates = np.array(protein.GetConformers()[0].GetPositions())
    # print('| HERE 1038 |', ligand_good_coordinates.shape, '\n', ligand_good_coordinates)
    # print('| HERE 1039 |', protein_coordinates.shape, '\n', protein_coordinates)

    # TODO: not sure this is the right way to deal with water
    cutoff_water = 1.5  # For displaced water molecules
    cutoff = args.max_interaction_distance

    # get the distance matrix (with only distance < threshold) then convert into 1/r^2
    cartesian_distance_matrix_interactions_good = distance_matrix(ligand_good_coordinates, protein_coordinates)

    cartesian_distance_matrix_interactions_good = (cutoff > cartesian_distance_matrix_interactions_good) * cartesian_distance_matrix_interactions_good
    # cartesian_distance_matrix_interactions_good = (cartesian_distance_matrix_interactions_good > cutoff_water) * cartesian_distance_matrix_interactions_good
    cartesian_distance_matrix_interactions_bad = distance_matrix(ligand_bad_coordinates, protein_coordinates)
    cartesian_distance_matrix_interactions_bad = (cutoff > cartesian_distance_matrix_interactions_bad) * cartesian_distance_matrix_interactions_bad
    # cartesian_distance_matrix_interactions_bad = (cartesian_distance_matrix_interactions_bad > cutoff_water) * cartesian_distance_matrix_interactions_bad

    #print('HERE 1053: ', cartesian_distance_matrix_interactions_good.shape)
    #print('HERE 1055: ', cartesian_distance_matrix_interactions_good.shape[0], flush=True)

    #print('|     first pose %98s |' % ' ', flush=True)
    for i in range(cartesian_distance_matrix_interactions_good.shape[0]):
        #print('HERE 1058: ', flush=True)
        for j in range(cartesian_distance_matrix_interactions_good.shape[1]):
            if cartesian_distance_matrix_interactions_good[i][j] > cutoff_water:  # or water[j] == 0:
                edge_features = []

                #print(' interaction between %3s and %3s: %6.2f (water: %s)' % (i, j, cartesian_distance_matrix_interactions_good[i][j], water[j]))
                # Features 1: bond order (#1-3)
                if args.bond_feature_bond_order is True:
                    if five_bond_features is False:
                        edge_features.append(0)
                    edge_features.append(0)
                    edge_features.append(0)
                    edge_features.append(0)

                # Feature 2: Conjugation (#4)
                if args.bond_feature_conjugation is True:
                    edge_features.append(0)

                if args.bond_feature_bond_order is False and args.bond_feature_conjugation is False:
                    edge_features.append(0)

                if args.LJ_or_one_hot is True:
                    # bond distances (#6-7)
                    edge_features.append(cartesian_distance_matrix_interactions_good[i][j])
                    if five_bond_features is False:
                        edge_features.append(cartesian_distance_matrix_interactions_good[i][j])
                else:
                    # bond distance (#6-9) < 3.0, < 4.0, <5.5, < 8.0
                    dist = cartesian_distance_matrix_interactions_good
                    if dist < 3.0:
                        edge_features.append(1)
                        edge_features.append(0)
                        edge_features.append(0)
                        edge_features.append(0)
                    elif dist < 4.0:
                        edge_features.append(0)
                        edge_features.append(1)
                        edge_features.append(0)
                        edge_features.append(0)
                    elif dist < 5.5:
                        edge_features.append(0)
                        edge_features.append(0)
                        edge_features.append(1)
                        edge_features.append(0)
                    elif dist < 8.0:
                        edge_features.append(0)
                        edge_features.append(0)
                        edge_features.append(0)
                        edge_features.append(1)

                # TODO remove when testing done
                #if i == 0: #and j == 0:
                #    print('edge features in complex:\n', edge_features)
                #time.sleep(25)
                # edge indices (protein atoms are after ligand atoms in the complex)
                # The non-bonded interactions are in the top right and bottom left quadrants.
                edge_indices_good += [[i, j + num_of_lig_atoms],
                                      [j + num_of_lig_atoms, i]]

                #print('edge features: \n', i, j, edge_features)
                # Append edge features to matrix (twice, per direction)
                #print('890', edge_features, '\n')
                edges_features_good += [edge_features, edge_features]
                #if i == 0:
                #    print('edge_indices_good', i, j+ num_of_lig_atoms, cartesian_distance_matrix_interactions_good[i][j])

    #print('|     second pose %98s |' % ' ', flush=True)
    for i in range(cartesian_distance_matrix_interactions_bad.shape[0]):
        for j in range(cartesian_distance_matrix_interactions_bad.shape[1]):
            if cartesian_distance_matrix_interactions_bad[i][j] > cutoff_water:  # or water[j] is False:
                edge_features = []

                # Features 1: bond order (#1-4)
                if args.bond_feature_bond_order is True:
                    if five_bond_features is False:
                        edge_features.append(0)
                    edge_features.append(0)
                    edge_features.append(0)
                    edge_features.append(0)

                # Feature 2: Conjugation (#5)
                if args.bond_feature_conjugation is True:
                    edge_features.append(0)

                # Feature 3: if no feature (#1)
                if args.bond_feature_bond_order is False and args.bond_feature_conjugation is False:
                    edge_features.append(0)

                if args.LJ_or_one_hot is True:
                    # bond distances (#6-7)
                    edge_features.append(cartesian_distance_matrix_interactions_bad[i][j])
                    if five_bond_features is False:
                        edge_features.append(cartesian_distance_matrix_interactions_bad[i][j])
                else:
                    # bond distance (#6-9) < 3.0, < 4.0, <5.5, < 8.0
                    dist = cartesian_distance_matrix_interactions_bad
                    if dist < 3.0:
                        edge_features.append(1)
                        edge_features.append(0)
                        edge_features.append(0)
                        edge_features.append(0)
                    elif dist < 4.0:
                        edge_features.append(0)
                        edge_features.append(1)
                        edge_features.append(0)
                        edge_features.append(0)
                    elif dist < 5.5:
                        edge_features.append(0)
                        edge_features.append(0)
                        edge_features.append(1)
                        edge_features.append(0)
                    elif dist < 8.0:
                        edge_features.append(0)
                        edge_features.append(0)
                        edge_features.append(0)
                        edge_features.append(1)

                # edge indices (protein atoms are after ligand atoms in the complex)
                # The non-bonded interactions are in the top right and bottom left quadrants.
                edge_indices_bad += [[i, j + num_of_lig_atoms],
                                     [j + num_of_lig_atoms, i]]

                # Append edge features to matrix (twice, per direction)
                edges_features_bad += [edge_features, edge_features]
                #print('950', edge_features, '\n')
                #if i == 0:
                #    print('edge_indices_bad', i, j+ num_of_lig_atoms, cartesian_distance_matrix_interactions_bad[i][j])

    try:
        #print("featurizer 1182: ", edges_features_good)
        edges_features_good = np.array(edges_features_good)
        edges_features_good = torch.tensor(edges_features_good, dtype=torch.float32)
        #print("featurizer 1184: ", edges_features_good)
        edge_indices_good = torch.tensor(edge_indices_good)
        edge_indices_good = edge_indices_good.t().to(torch.long).view(2, -1)

        edges_features_bad = np.array(edges_features_bad)
        edges_features_bad = torch.tensor(edges_features_bad, dtype=torch.float32)
        edge_indices_bad = torch.tensor(edge_indices_bad)
        edge_indices_bad = edge_indices_bad.t().to(torch.long).view(2, -1)
    except:
        return False

    return edge_indices_good, edges_features_good, edge_indices_bad, edges_features_bad


def get_distance_matrix(mol):
    distance_matrix_mol = rdmolops.GetDistanceMatrix(mol)
    return distance_matrix_mol


def get_labels(label):
    label = np.array([label])
    return torch.tensor(label, dtype=torch.float32)


def get_sigmoid_value(rmsd):
    label = 1 / (1 + math.exp((rmsd * 10) + 20))
    label = np.array([label])
    return torch.tensor(label, dtype=torch.float32)


def get_mol_charge(mol):
    mol_formal_charge = rdmolops.GetFormalCharge(mol)
    return torch.tensor(mol_formal_charge, dtype=torch.float32)


def get_mol_nrot(mol):
    mol_nrot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return torch.tensor(mol_nrot, dtype=torch.float32)


def get_mol_natoms(mol):
    mol_natoms = 0
    for atom in mol.GetAtoms():
        mol_natoms += 1
    return torch.tensor(mol_natoms, dtype=torch.float32)


def is_HBA_HBD(atom):
    HBA = 0
    HBD = 0
    HB = []
    if atom.GetSymbol() == 'O' or atom.GetSymbol() == 'S':
        HBA = 1
        if atom.GetNumImplicitHs() == 1 or atom.GetNumExplicitHs() == 1:
            HBD = 1

    if atom.GetSymbol() == 'N':
        # pyridine but not pyrrole
        if atom.GetIsAromatic() and atom.GetNumImplicitHs() == 0 and atom.GetNumExplicitHs() == 0:
            HBA = 1
        if atom.GetHybridization() is Chem.rdchem.HybridizationType.SP2 and atom.GetNumImplicitHs() == 0 and \
                atom.GetNumExplicitHs() == 0 and atom.GetFormalCharge() == 0:
            HBA = 1

        if atom.GetNumImplicitHs() >= 1 or atom.GetNumExplicitHs() >= 1:
            HBD = 1
    HB.append(HBA)
    HB.append(HBD)

    return HB
