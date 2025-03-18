# Author: Nic Moitessier
# McGill University, Montreal, QC, Canada


import torch
import pickle
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from rdkit import Chem
from rdkit.Geometry import Point3D


# Check for GPU
def set_cuda_visible_device(ngpus):
    import subprocess
    import os
    empty = []
    for i in range(8):
        command = 'nvidia-smi -i '+str(i)+' | grep "0%" | wc -l'
        output = subprocess.check_output(command, shell=True).decode("utf-8")
        if int(output) == 1:
            empty.append(i)
    if len(empty) < ngpus:
        print('| available gpus are less than required')
        exit(-1)
    cmd = ''
    for i in range(ngpus):        
        cmd += str(empty[i])+','
    return cmd


# Count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Load data
def load_data(file_name):
    with open(file_name, "rb") as f:
        conts = f.read()
    data = pickle.loads(conts)
    return data


# Compute MAE
def compute_mae(data, ref):
    mae = []
    for i in range(len(data)):
        mae.append((ref[i] - data[i]))
    return mae


# Search item in a list
def search(item, list_items):
    for i in range(len(list_items)):
        if list_items[i] == item:
            return True
    return False


# Compute accuracy and other metrics
def calculate_metrics(y_pred, y_true, train_or_test, keys, args):
    seen = []
    pred = []
    label = []
    FN = []
    FP = []
    TN = []
    TP = []
    for i in range(9):
        FN.append(0)
        FP.append(0)
        TN.append(0)
        TP.append(0)
    # if train_or_test == 'train':
    #    print('Number of items: %6.0f %6.0f' % (len(y_pred), len(y_true)))
    #    print('Averages: %6.4f %6.4f ' % (sum(y_pred)/len(y_pred), sum(y_true)/len(y_true)))
    #    print('Min: %6.4f %6.4f  ' % (min(y_pred), min(y_true)))
    #    print('Max: %6.4f %6.4f' % (max(y_pred), max(y_true)))
    #    print('First 100 items: ')
    #    for i in range(100):
    #        print('%4.0f %s %6.4f %6.4f' % (i+1, keys[i], y_pred[i], y_true[i]))
    #    print('Last 100 items: ')
    #    for i in range(100):
    #        print('%4.0f %s %6.4f %6.4f' % (i+1, keys[len(y_pred)-101+i], y_pred[len(y_pred)-101+i], y_true[len(y_pred)-101+i]))

    # Take the best prediction in case the ligand appears more than once (tautomers)
    for i in range(len(y_pred)):
        # if search(keys[i], seen) is False:
        seen.append(keys[i])
        pred.append(y_pred[i])
        label.append(y_true[i])
        # else:
        #    print('| already seen: ', keys[i])
        #    idx = seen.index(keys[i])
        #    for threshold in range(9):
        #        if abs(pred[idx] - label[idx]) > abs(y_pred[i] - y_true[i]):
        #            pred[idx] = y_pred[i]
    # if args.verbose > 3:
    #  print('| %7.0f unique compounds out of %-7.0f                                                                          |' % (len(pred), len(y_pred)))
    #  print('|-------------------------------------------------------------------------------------------------------------------|')

    # Count the number of false and true positives and negatives (FP, TP, FN, TN) at different thresholds (0.1, 0.1,..., 0.9)
    for i in range(len(pred)):
        for threshold in range(1, 10, 1):
            # if compound active
            if label[i] > 0.5:
                # if prediction greater than threshold -> predicted active: TP
                if pred[i] >= 0.1 * threshold:
                    TP[threshold-1] += 1
                # if prediction lower -> predicted inactive: FN
                else:
                    FN[threshold-1] += 1
            # if compound inactive
            else:
                # if prediction lower than threshold -> predicted inactive: TN
                if pred[i] <= 0.1 * threshold:
                    TN[threshold-1] += 1
                # if prediction greater than threshold -> predicted active: FP
                else:
                    FP[threshold-1] += 1

    # Compute some metrics
    mae = mean_absolute_error(label, pred)
    mse = mean_squared_error(label, pred)
    rmse = mean_squared_error(label, pred, squared=False)
    roc = roc_auc_score(label, pred)

    if train_or_test == 'train':
        print('| Training set:                                   %-66s|      ' % ' ')
    else:
        print('| Testing set:                                    %-66s|      ' % ' ')
    print('| MAE (mean absolute error):                      %-66.3f|' % mae)
    print('| MSE (mean squared error):                       %-66.3f|' % mse)
    print('| RMSE (root mean square error):                  %-66.3f|' % rmse)
    print('| AUROC (area under receiver operating curve):    %-66.3f|' % roc)
    print('| Total number of molecules, actives, inactives:  %-8.0f %-8.0f %-48.0f|' % ((TP[0] + TN[0] + FP[0] + FN[0]),
                                                                                        (TP[0] + FN[0]),
                                                                                        (TN[0] + FP[0])))
    print('|-------------------------------------------------------------------------------------------------------------------|')
    print('| Threshold               |', end='')
    for i in range(9):
        print(' %-7.2f |' % ((i + 1) * 0.1), end='')

    print('\n| True positive (TP)      |', end='')
    for i in range(9):
        print(' %-7.0f |' % TP[i], end='')

    print('\n| True negative (TN)      |', end='')
    for i in range(9):
        print(' %-7.0f |' % TN[i], end='')

    print('\n| False positive (FP)     |', end='')
    for i in range(9):
        print(' %-7.0f |' % FP[i], end='')

    print('\n| False negative (FN)     |', end='')
    for i in range(9):
        print(' %-7.0f |' % FN[i], end='')

    print('\n| Accuracy                |', end='')
    for i in range(9):
        print(' %-7.3f |' % ((TP[i] + TN[i]) / float(TP[i] + TN[i] + FP[i] + FN[i])), end='')

    print('\n| Sensitivity/act. recall |', end='')
    for i in range(9):
        if (TP[i] + FN[i]) == 0:
            print(' 1.000   |', end='')
        else:
            print(' %-7.3f |' % (float(TP[i]) / float(TP[i] + FN[i])), end='')

    print('\n| Specificity/in. recall  |', end='')
    for i in range(9):
        if (FP[i] + TN[i]) == 0:
            print(' 1.000   |', end='')
        else:
            print(' %-7.3f |' % (float(TN[i]) / float(FP[i] + TN[i])), end='')

    print('\n| PPV/active precision    |', end='')
    for i in range(9):
        if (TP[i] + FP[i]) == 0:
            print(' 1.000   |', end='')
        else:
            print(' %-7.3f |' % (float(TP[i]) / float(TP[i] + FP[i])), end='')

    print('\n| NPV/inactive precision  |', end='')
    for i in range(9):
        if (TN[i] + FN[i]) == 0:
            print(' 1.000   |', end='')
        else:
            print(' %-7.3f |' % (float(TN[i]) / float(TN[i] + FN[i])), end='')

    print('\n|-------------------------------------------------------------------------------------------------------------------|', flush=True)
    print('| NOTES:                                          %-66s|' % ' ')
    print('| Accuracy: how often the prediction is right     %-66s|' % ' ')
    if args.train_mode == 'docking':
        print('| AUROC: probability that the model will be able to distinguish between correct and incorrect pose.                 |')
        print('| Sensitivity: proportion of poses predicted to be correct among those that are correct.                            |')
        print('| Specificity: proportion of poses predicted to be incorrect among those that are incorrect.                        |')
        print('| Positive predictive value (PPV): probability that a predicted correct pose is correct.                            |')
        print('| Negative predictive value (NPV): probability that a predicted incorrect pose is incorrect.                        |')
    else:
        print('| AUROC: probability that the model will be able to distinguish between active and inactive molecules.              |')
        print('| Sensitivity: proportion of molecules predicted to be active among those that are experimentally active.           |')
        print('| Specificity: proportion of molecules predicted to be inactive among those that are experimentally inactive.       |')
        print('| Positive predictive value (PPV): probability that a predicted active is experimentally active.                    |')
        print('| Negative predictive value (NPV): probability that a predicted inactive is experimentally inactive.                |')
    print('|-------------------------------------------------------------------------------------------------------------------|', flush=True)


# Compute accuracy and other metrics
def calculate_metrics_pairs(y_pred, y_true, train_or_test, keys):
    seen = []
    pred = []
    label = []
    FN = []
    FP = []
    TN = []
    TP = []
    for i in range(9):
        FN.append(0)
        FP.append(0)
        TN.append(0)
        TP.append(0)

    # Take the best prediction in case the ligand appears more than once (tautomers)
    for i in range(len(y_pred)):
        # if search(keys[i], seen) is False:
        seen.append(keys[i])
        pred.append(y_pred[i])
        label.append(y_true[i])
        # else:
        #    print('| already seen: ', keys[i])
        #    idx = seen.index(keys[i])
        #    for threshold in range(9):
        #        if abs(pred[idx] - label[idx]) > abs(y_pred[i] - y_true[i]):
        #            pred[idx] = y_pred[i]
    # if args.verbose > 3:
    #  print('| %7.0f unique compounds out of %-7.0f                                                                          |' % (len(pred), len(y_pred)))
    #  print('|-------------------------------------------------------------------------------------------------------------------|')

    # print('| utils calculate_metrics_pairs(), 221:  ')
    # print('| Preds:  ', pred)
    # print('| Labels: ', label)
    # Count the number of false position (FN), ... at different thresholds (0.1, 0.1,..., 0.9)
    for i in range(len(pred)):
        for threshold in range(1, 10, 1):
            # if compound active
            if label[i] > 0.0:
                # if prediction greater than threshold -> predicted active; TP
                if pred[i] >= 0.2 * threshold - 1.0:
                    TP[threshold-1] += 1
                # if prediction lower -> predicted inactive: FN
                else:
                    FN[threshold-1] += 1
            # if compound inactive
            else:
                # if prediction lower than threshold -> predicted inactive: TN
                if pred[i] <= 0.2 * threshold - 1.0:
                    TN[threshold-1] += 1
                # if prediction greater than threshold -> predicted active: FP
                else:
                    FP[threshold-1] += 1

    # Compute some metrics
    mae = mean_absolute_error(pred, label)
    mse = mean_squared_error(pred, label)
    rmse = mean_squared_error(pred, label, squared=False)
    roc = roc_auc_score(label, pred)

    if train_or_test == 'train':
        print('| Training set:                                   %-66s|' % ' ')
    else:
        print('| Testing set:                                    %-66s|' % ' ')
    print('| MAE (mean absolute error):                      %-66.3f|' % mae)
    print('| MSE (mean squared error):                       %-66.3f|' % mse)
    print('| RMSE (root mean square error):                  %-66.3f|' % rmse)
    print('| AUROC (area under receiver operating curve):    %-66.3f|' % roc)
    print('| Total number of molecules, actives, inactives:  %-8.0f %-8.0f %-48.0f|' % ((TP[0] + TN[0] + FP[0] + FN[0]),
                                                                                        (TP[0] + FN[0]),
                                                                                        (TN[0] + FP[0])))
    print('|-------------------------------------------------------------------------------------------------------------------|')
    print('| Threshold               |', end='')
    for i in range(9):
        print(' %-7.2f |' % ((i + 1) * 0.2 - 1.0), end='')

    print('\n| True positive (TP)      |', end='')
    for i in range(9):
        print(' %-7.0f |' % TP[i], end='')

    print('\n| True negative (TN)      |', end='')
    for i in range(9):
        print(' %-7.0f |' % TN[i], end='')

    print('\n| False positive (FP)     |', end='')
    for i in range(9):
        print(' %-7.0f |' % FP[i], end='')

    print('\n| False negative (FN)     |', end='')
    for i in range(9):
        print(' %-7.0f |' % FN[i], end='')

    print('\n| Accuracy                |', end='')
    for i in range(9):
        print(' %-7.3f |' % ((TP[i] + TN[i]) / float(TP[i] + TN[i] + FP[i] + FN[i])), end='')

    print('\n| Sensitivity/act. recall |', end='')
    for i in range(9):
        if (TP[i] + FN[i]) == 0:
            print(' 1.000   |', end='')
        else:
            print(' %-7.3f |' % (float(TP[i]) / float(TP[i] + FN[i])), end='')

    print('\n| Specificity/in. recall  |', end='')
    for i in range(9):
        if (FP[i] + TN[i]) == 0:
            print(' 1.000   |', end='')
        else:
            print(' %-7.3f |' % (float(TN[i]) / float(FP[i] + TN[i])), end='')

    print('\n| PPV/active precision    |', end='')
    for i in range(9):
        if (TP[i] + FP[i]) == 0:
            print(' 1.000   |', end='')
        else:
            print(' %-7.3f |' % (float(TP[i]) / float(TP[i] + FP[i])), end='')

    print('\n| NPV/inactive precision  |', end='')
    for i in range(9):
        if (TN[i] + FN[i]) == 0:
            print(' 1.000   |', end='')
        else:
            print(' %-7.3f |' % (float(TN[i]) / float(TN[i] + FN[i])), end='')

    print('\n|-------------------------------------------------------------------------------------------------------------------|', flush=True)
    print('| NOTES:                                          %-66s|' % ' ')
    print('| Accuracy: how often the prediction is right     %-66s|' % ' ')
    print('| Sensitivity: proportion of molecules predicted to be active among those that are experimentally active.           |')
    print('| Specificity: proportion of molecules predicted to be inactive among those that are experimentally inactive.       |')
    print('| Positive predictive value (PPV): probability that a predicted active is experimentally active.                    |')
    print('| Negative predictive value (NPV): probability that a predicted inactive is experimentally inactive.                |')
    print('|-------------------------------------------------------------------------------------------------------------------|', flush=True)


# Is character a digit?
def isDigit(char):
    if char == '0' or char == '1' or char == '2' or char == '3' or char == '4' or char == '5' or char == '6' \
       or char == '7' or char == '8' or char == '9':
        return True
    return False


# Splits a list into two lists
def split_list(original_list, fraction):
    section = int(len(original_list) * fraction)
    return original_list[:section], original_list[section:]


# The following function adds a formal charge to N with 4 bonds
def add_formal_charges(mol):
    mol.UpdatePropertyCache(strict=False)
    # print('Carbon carboxylate: ', mol.GetAtomWithIdx(30).GetExplicitValence(), mol.GetAtomWithIdx(30).GetIsAromatic())

    # correcting the use of "ar" for carboxylates and phosphates
    # for at in mol.GetAtoms():
    #    if (at.GetSymbol() == 'C' or at.GetSymbol() == 'P') and at.GetIsAromatic() is False:
    #        #print(' Carbon ', at.GetIdx())
    #        idx = at.GetIdx()
    #        found = 0
    #        for bond in mol.GetBonds():
    #            #print('bond: ', bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),bond.GetBondTypeAsDouble())
    #            if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
    #                if bond.GetBeginAtomIdx() == idx and mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol() == 'O':
    #                    if found == 0:
    #                        #print(' Oxygen ar bond is set to 1 ', at.GetIdx())
    #                        bond.SetBondType(Chem.rdchem.BondType.SINGLE)
    #                        mol.GetAtomWithIdx(bond.GetEndAtomIdx()).SetFormalCharge(-1)
    #                        #print(' Oxygen formal charge adjusted ', mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetFormalCharge(),
    #                               mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetIdx())
    #                        found = 1
    #                    elif found == 1:
    #                        #print(' Oxygen ar bond is set to 2 ', at.GetIdx())
    #                        bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
    #                        mol.GetAtomWithIdx(bond.GetEndAtomIdx()).SetFormalCharge(0)
    #                        #print(' Oxygen formal charge adjusted ', mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetFormalCharge(),
    #                              mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetIdx())
    #                        found = 0
    #
    #                if bond.GetEndAtomIdx() == idx and mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol() == 'O':
    #                    if found == 0:
    #                        #print(' Oxygen ar bond is set to 1 ', at.GetIdx())
    #                        bond.SetBondType(Chem.rdchem.BondType.SINGLE)
    #                        mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).SetFormalCharge(-1)
    #                        print(' Oxygen formal charge adjusted ', mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetFormalCharge(),
    #                             mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetIdx())
    #                        found = 1
    #                    elif found == 1:
    #                        #print(' Oxygen ar bond is set to 2 ', at.GetIdx())
    #                        bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
    #                        mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).SetFormalCharge(0)
    #                        #print(' Oxygen formal charge adjusted ', mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetFormalCharge(),
    #                              mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetIdx())
    #                        found = 0

    for at in mol.GetAtoms():
        if at.GetAtomicNum() == 7:
            if at.GetExplicitValence() == 4:
                mol.GetAtomWithIdx(at.GetIdx()).SetFormalCharge(1)
                # print(' Nitrogen formal charge adjusted ', at.GetFormalCharge(), at.GetIdx())
        # if at.GetExplicitValence() == 3 and at.GetIsAromatic() is True:
        #    mol.GetAtomWithIdx(at.GetIdx()).SetFormalCharge(1)
        #    # print('formal charge adjusted ', at.GetFormalCharge(), at.GetIdx())
        if at.GetAtomicNum() == 5:
            if at.GetExplicitValence() == 4:
                mol.GetAtomWithIdx(at.GetIdx()).SetFormalCharge(-1)
                # print(' Nitrogen formal charge adjusted ', at.GetFormalCharge(), at.GetIdx())
        # if at.GetExplicitValence() == 3 and at.GetIsAromatic() is True:
        #    mol.GetAtomWithIdx(at.GetIdx()).SetFormalCharge(1)
        #    # print('formal charge adjusted ', at.GetFormalCharge(), at.GetIdx())
        if at.GetAtomicNum() == 8:
            # print('Oxygen valence: ', at.GetExplicitValence(), at.GetIsAromatic())
            if at.GetExplicitValence() == 1:
                mol.GetAtomWithIdx(at.GetIdx()).SetFormalCharge(-1)
                # print(' Oxygen formal charge adjusted ', at.GetFormalCharge(), at.GetIdx())
            # In case of carboxylate using "ar" bonds
            # ERROR: we cannot have a non-integer formal charge
            # if at.GetExplicitValence() == 1.5:
            #    mol.GetAtomWithIdx(at.GetIdx()).SetFormalCharge(-0.5)
            #    print(' Oxygen formal charge adjusted ', at.GetFormalCharge(), at.GetIdx())
    # print('Now: ', mol.GetAtomWithIdx(28).GetFormalCharge())


# Swaps atoms in a file to get a different order (i.e., a different graph)
def swap_atoms(mol, from_idx, to_idx):
    # new_mol = copy.deepcopy(mol)
    atom1 = mol.GetAtomWithIdx(from_idx)
    atom2 = mol.GetAtomWithIdx(to_idx)

    mw = Chem.RWMol(mol)  # The bonds are reordered when doing this...

    # now we have a rewritable molecule, we switch atoms and their coordinates.
    mw.ReplaceAtom(to_idx, atom1)
    mw.ReplaceAtom(from_idx, atom2)
    coordinates_1 = mw.GetConformer().GetAtomPosition(to_idx)
    coordinates_2 = mw.GetConformer().GetAtomPosition(from_idx)
    mw.GetConformer().SetAtomPosition(from_idx, Point3D(coordinates_1.x, coordinates_1.y, coordinates_1.z))
    mw.GetConformer().SetAtomPosition(to_idx, Point3D(coordinates_2.x, coordinates_2.y, coordinates_2.z))

    # Now we had the corresponding bonds and remove the old ones.
    toBeRemoved = []
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() == from_idx:
            mw.AddBond(to_idx, bond.GetEndAtomIdx(), bond.GetBondType())
            toBeRemoved.append(bond.GetIdx())
        if bond.GetEndAtomIdx() == from_idx:
            mw.AddBond(bond.GetBeginAtomIdx(), to_idx, bond.GetBondType())
            toBeRemoved.append(bond.GetIdx())
        if bond.GetBeginAtomIdx() == to_idx:
            mw.AddBond(from_idx, bond.GetEndAtomIdx(), bond.GetBondType())
            toBeRemoved.append(bond.GetIdx())
        if bond.GetEndAtomIdx() == to_idx:
            mw.AddBond(bond.GetBeginAtomIdx(), from_idx, bond.GetBondType())
            toBeRemoved.append(bond.GetIdx())

    for i in range(len(toBeRemoved)-1, -1, -1):
        mw.RemoveBond(mol.GetBondWithIdx(toBeRemoved[i]).GetBeginAtomIdx(),
                      mol.GetBondWithIdx(toBeRemoved[i]).GetEndAtomIdx())

    return mw


# scrambles the atom list to get a different atom orders (ie a different graph)
def scramble_atom_list(mol, rng):
    swap = 0
    while swap < 1:
        from_idx = int(rng.random() * (mol.GetNumAtoms() - 1)) + 1
        to_idx = int(rng.random() * (mol.GetNumAtoms() - 1)) + 1
        # print(' NOW WE SCRAMBLE ', from_idx, to_idx, flush=True)
        if notBound(mol, from_idx, to_idx) is False:
            continue
        if from_idx != to_idx:
            mol = swap_atoms(mol, from_idx, to_idx)
            swap += 1

    return mol


# scrambles the atom list to get a different atom orders (ie a different graph)
def scramble_atom_list_pairs(mol1, mol2, rng):
    swap = 0
    while swap < 10:
        from_idx = int(rng.random() * (mol1.GetNumAtoms() - 1)) + 1
        to_idx = int(rng.random() * (mol1.GetNumAtoms() - 1)) + 1
        # print(' NOW WE SCRAMBLE ', from_idx, to_idx, flush=True)
        if notBound(mol1, from_idx, to_idx) is False:
            continue
        if from_idx != to_idx:
            mol1 = swap_atoms(mol1, from_idx, to_idx)
            mol2 = swap_atoms(mol2, from_idx, to_idx)
            swap += 1

    return mol1, mol2


# The following function ensures the atoms swapped are not in a bond or angle. This causes the swapping function to crash
def notBound(mol, from_idx, to_idx):
    for bond in mol.GetBonds():
        # print(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), from_idx, to_idx)
        if bond.GetBeginAtomIdx() == from_idx and bond.GetEndAtomIdx() == to_idx:
            return False
        if bond.GetBeginAtomIdx() == to_idx and bond.GetEndAtomIdx() == from_idx:
            return False

    for bond1 in mol.GetBonds():
        # print(bond1.GetBeginAtomIdx(), bond1.GetEndAtomIdx(), from_idx, to_idx)
        if bond1.GetBeginAtomIdx() == from_idx:
            for bond2 in mol.GetBonds():
                if bond2.GetBeginAtomIdx() == to_idx and bond2.GetEndAtomIdx() == bond1.GetEndAtomIdx():
                    return False
                if bond2.GetEndAtomIdx() == to_idx and bond2.GetBeginAtomIdx() == bond1.GetEndAtomIdx():
                    return False

        if bond1.GetEndAtomIdx() == from_idx:
            for bond2 in mol.GetBonds():
                if bond2.GetBeginAtomIdx() == to_idx and bond2.GetEndAtomIdx() == bond1.GetBeginAtomIdx():
                    return False
                if bond2.GetEndAtomIdx() == to_idx and bond2.GetBeginAtomIdx() == bond1.GetBeginAtomIdx():
                    return False

        if bond1.GetBeginAtomIdx() == to_idx:
            for bond2 in mol.GetBonds():
                if bond2.GetBeginAtomIdx() == from_idx and bond2.GetEndAtomIdx() == bond1.GetEndAtomIdx():
                    return False
                if bond2.GetEndAtomIdx() == from_idx and bond2.GetBeginAtomIdx() == bond1.GetEndAtomIdx():
                    return False

        if bond1.GetEndAtomIdx() == to_idx:
            for bond2 in mol.GetBonds():
                if bond2.GetBeginAtomIdx() == from_idx and bond2.GetEndAtomIdx() == bond1.GetBeginAtomIdx():
                    return False
                if bond2.GetEndAtomIdx() == from_idx and bond2.GetBeginAtomIdx() == bond1.GetBeginAtomIdx():
                    return False

    return True


# Moving optimizer to device. ( https://github.com/pytorch/pytorch/issues/8741)
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def get_weights(train_keys, test_keys, args):
    if args.train_mode != "docking":
        print('|-------------------------------------------------------------------------------------------------------------------|')
        print('| Assigning weights to each sample (to ensure balanced batches)...                                                  |', flush=True)

    # Find the proteins represented in either train or test set
    train_protein_list = []
    test_protein_list = []

    for key in train_keys:
        pos = key.find('_')
        if key[0:pos] not in train_protein_list:
            train_protein_list.append(key[0:pos])

    for key in test_keys:
        pos = key.find('_')
        if key[0:pos] not in test_protein_list:
            test_protein_list.append(key[0:pos])

    # print('prepare-set 141: \n', train_protein_list, '\n', test_protein_list)
    dic_train_actives_size = {}
    dic_train_inactives_size = {}
    dic_test_actives_size = {}
    dic_test_inactives_size = {}
    dic_train_pairs_size = {}
    dic_test_pairs_size = {}
    if args.good_and_bad_pairs is False:
        # Assigning weights per protein (higher weights for those with fewer compounds) to ensure balanced set later on
        for protein in train_protein_list:
            dic_train_actives_size[protein] = len([0 for k in train_keys if '_active_' in k and protein in k])
            dic_train_inactives_size[protein] = len([0 for k in train_keys if '_inactive' in k and protein in k])
        for protein in test_protein_list:
            dic_test_actives_size[protein] = len([0 for k in test_keys if '_active_' in k and protein in k])
            dic_test_inactives_size[protein] = len([0 for k in test_keys if '_inactive_' in k and protein in k])
    else:
        for protein in train_protein_list:
            dic_train_pairs_size[protein] = len([0 for k in train_keys if protein in k])
        for protein in test_protein_list:
            dic_test_pairs_size[protein] = len([0 for k in test_keys if protein in k])

    if args.good_and_bad_pairs is False:
        print('|-------------------------------------------------------------------------------------------------------------------|')
        print('| First identifying actives/inactives in the training and testing sets.                                             |')
        if args.verbose > 2:
            print('| Number of actives per protein in the training set:                                                                |')
            print('|----------------------------------------- - - - - - -                                                              |', flush=True)
        count = 0
        for i, key in enumerate(dic_train_actives_size):
            if args.verbose > 2:
                print('| %17s: %7s                                                                                        |'
                      % (key, str(dic_train_actives_size[key])))
            count += dic_train_actives_size[key]
        if args.verbose > 2:
            print('|-------------------                                                                                                |')
            print('|   Total: %7s actives in the training set                                                                      |' % count)
            print('|--------------------------------------------------------------                                                     |', flush=True)
            print('| Number of actives per protein in the training set:                                                                |')
            print('|----------------------------------------- - - - - - -                                                              |', flush=True)
        else:
            print('|   %7s actives in the training set                                                                             |' % count)

        train_set_size = count

        count = 0
        for i, key in enumerate(dic_train_inactives_size):
            if args.verbose > 2:
                print('| %17s: %7s                                                                                        |'
                      % (key, str(dic_train_inactives_size[key])))
            count += dic_train_inactives_size[key]
        if args.verbose > 2:
            print('|-------------------                                                                                                |')
            print('|   Total: %7s inactives in the training set                                                                    |' % count)
            print('|--------------------------------------------------------------                                                     |', flush=True)
            print('| Number of inactives per protein in the testing set:                                                               |')
            print('|----------------------------------------- - - - - - -                                                              |', flush=True)
        else:
            print('|   %7s inactives in the training set                                                                           |' % count)

        train_set_size += count

        count = 0
        for i, key in enumerate(dic_test_actives_size):
            if args.verbose > 2:
                print('| %17s: %7s                                                                                        |'
                      % (key, str(dic_test_actives_size[key])))
            count += dic_test_actives_size[key]
        if args.verbose > 2:
            print('|-------------------                                                                                                |')
            print('|   Total: %7s actives in the testing set                                                                       |' % count)
            print('|--------------------------------------------------------------                                                     |', flush=True)

            print('| Number of actives per protein in the testing set:                                                                 |')
            print('|----------------------------------------- - - - - - -                                                              |', flush=True)
        else:
            print('|   %7s actives in the testing set                                                                              |' % count)

        test_set_size = count

        count = 0
        for i, key in enumerate(dic_test_inactives_size):
            if args.verbose > 2:
                print('| %17s: %7s                                                                                        |'
                      % (key, str(dic_test_inactives_size[key])))
            count += dic_test_inactives_size[key]
        if args.verbose > 2:
            print('|-------------------                                                                                                |')
            print('|   Total: %7s inactives in the testing set                                                                     |' % count)
        else:
            print('|   %7s inactives in the testing set                                                                            |' % count)

    else:
        print('|-------------------------------------------------------------------------------------------------------------------|')
        print('| Identifying pairs in the training and testing sets.                                                               |')
        if args.verbose > 2:
            print('| Number of pairs per protein in the training set:                                                                  |')
            print('|----------------------------------------- - - - - - -                                                              |', flush=True)
        count = 0
        for i, key in enumerate(dic_train_pairs_size):
            if args.verbose > 2:
                print('| %17s: %7s                                                                                        |'
                      % (key, str(dic_train_pairs_size[key])))
            count += dic_train_pairs_size[key]
        if args.verbose > 2:
            print('|-------------------                                                                                                |')
            print('|   Total: %7s pairs in the training set                                                                        |' % count)
            print('|--------------------------------------------------------------                                                     |', flush=True)
        else:
            print('|   %7s pairs in the training set                                                                               |' % count)

        train_set_size = count

        count = 0
        for i, key in enumerate(dic_test_pairs_size):
            if args.verbose > 2:
                print('| %17s: %7s                                                                                        |'
                      % (key, str(dic_test_pairs_size[key])))
            count += dic_test_pairs_size[key]
        if args.verbose > 2:
            print('|-------------------                                                                                                |')
            print('|   Total: %7s pairs in the testing set                                                                         |' % count)
            print('|--------------------------------------------------------------                                                     |', flush=True)

            print('| Number of pairs per protein in the testing set:                                                                   |')
            print('|----------------------------------------- - - - - - -                                                              |', flush=True)
        else:
            print('|   %7s pairs in the testing set                                                                                |' % count)

        test_set_size = count

    if args.train_mode == "docking":
        train_weights = [1 for k in train_keys]
    else:
        print('|-------------------------------------------------------------------------------------------------------------------|')
        print('| Now computing weights based on occurrence (ie if a protein is overrepresented, it will be given a lower weight to |')
        print('| compensate (selected less often during the training).                                                             |', flush=True)
        test_set_size += count

        # Initialize the weights to 0
        train_weights = [0 for k in train_keys]
        count = 0
        if args.good_and_bad_pairs is False:
            for item in train_keys:
                for protein in train_protein_list:
                    if '_active' in item and protein in item:
                        if dic_train_actives_size[protein] == 0:
                            count -= 1
                        elif args.train_mode == 'scoring':
                            train_weights[count] = 1.0 / dic_train_actives_size[protein]
                        else:
                            train_weights[count] = 1.0
                    if '_inactive' in item and protein in item:
                        if dic_train_inactives_size[protein] == 0:
                            count -= 1
                        elif args.train_mode == 'scoring':
                            train_weights[count] = 1.0 / dic_train_inactives_size[protein]
                        else:
                            train_weights[count] = 1.0
                count += 1
        else:
            for item in train_keys:
                for protein in train_protein_list:
                    if dic_train_pairs_size[protein] == 0:
                        count -= 1
                    elif args.train_mode == 'scoring':
                        train_weights[count] = 1.0 / dic_train_pairs_size[protein]
                    else:
                        train_weights[count] = 1.0
                count += 1

    return train_weights, train_set_size, test_set_size


def restrict_set(keys, args):
    new_keys = []

    for key in keys:
        filename = key
        filename_split = filename.split("_")
        system_num = int(filename_split[2])

        if system_num <= args.max_num_of_systems:
            new_keys.append(filename)
        #    print(filename, system_num, ' kept')
        # else:
        #    print(filename, system_num, ' removed')

    print('| Selecting data points from dataset, from %7.0f to %-7.0f %53s |' % (len(keys), len(new_keys), ' '))
    # print(new_keys)
    return new_keys


def initialize_model(model, device, args):
    for param in model.parameters():
        if param.dim() == 1:
            continue
            # nn.init.constant(param, 0)
        else:
            nn.init.xavier_normal_(param, gain=args.initializer_gain)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    return model


def get_Forecaster_graph(mol_num, graph_file, printIt):
    data = {}
    counter = -1
    correct_molecule = False
    with open(graph_file) as f:
        lines = [l.strip() for l in f.readlines()]
        for line in lines:
            #if printIt:
            #    print('get Forecaster_graph 0: ', counter, mol_num, ' ------- ', line)
            if line.startswith('atom_features') or line.startswith('node_features'):
                #if printIt:
                #    print('get Forecaster_graph 1: ', counter, mol_num, ' ------- ', line)
                counter += 1
                if counter == mol_num:
                    data[counter] = line + '##'
            elif counter == mol_num and line == '$$$$':
                return data
            elif counter == mol_num:
                data[counter] += line + '##'
            elif counter > mol_num:
                return data
    #for k, v in data.items():
    #    print(f'{k} --> {v}')
