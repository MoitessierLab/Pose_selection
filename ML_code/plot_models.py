# Author: Anne Labarre
# Date created: 2024-06-24
# McGill University, Montreal, QC, Canada

import os
import xgboost as xgb
import json
import pandas as pd
import numpy as np
#from keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import random
from prepare_input_data import split_set
import plotly.express as px
import shap
from sklearn.inspection import permutation_importance # pip install -U scikit-learn
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib_venn import venn3, venn3_circles
#from venny4py.venny4py import *
from scipy import stats
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import csv
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


cwd = os.getcwd()

# Function to convert the activity back to RMSD
def act_to_RMSD(y):
    # Parameters of the logarithmic equation
    k = -0.1
    a = 2.25
    b = 1

    if y <= 0:
        RMSD = 4.00
        return RMSD
    if y >= 1.00:
        RMSD = 0.01
        return RMSD
    else:
        RMSD = -k*np.log(b/y - 1) + a
        return RMSD

def drop_features(wholeSet):
    #pm = {'drop_all': ['RMSD', 'Name', 'Label']}
    pm = {'drop_all': ['RMSD', 'Name', 'Label', 'Energy']}
    #pm = {'drop_all': ['RMSD', 'Name', 'Label', 'Energy', 'MatchScore']}
    #pm = {'drop_all': ['RMSD', 'Name', 'Energy', 'MatchScore', 'Act/Inact?']}
    #pm = {'drop_all': ['RMSD', 'Name', 'Energy', 'MatchScore', 'Bonds', 'Angles', 'Torsions', 'Angles_M', 'Tors_M', 'vdW_14', 'Elec_14', 'vdW_15', 'Elec_15', 'vdW', 'Elec', 'HBonds', 'HB_M', 'Bond_M', 'Wat_vdW', 'Wat_Elec', 'Wat_HB', 'No_Wat', 'Act/Inact?']}
    for drop in pm['drop_all']:
        wholeSet = wholeSet.drop([drop], axis=1)

    pm = {'keep_15': ['vdW_15', 'ARG_SC_HB', 'vdW_14', 'Energy', 'Wat_HB', 'Elec', 'No_Wat', 'LEU_SC_vdW', 'HBonds', 'Angles', 'ASN_BB_vdW', 'ALA_SC_vdW', 'MatchScore', 'ASP_SC_HB', 'GLU_BB_vdW']}
    #for col in wholeSet.columns:
    #    if col not in pm['keep_15']:
    #        wholeSet = wholeSet.drop([col], axis=1)

    return wholeSet

def plot():
    return 1

def plot_RMSDvsEnergy(args):

    PDBname = '3zm5'
    data = pd.read_csv('CSV/' + PDBname + '_filtered.csv', index_col=False)

    x_values = data['Energy']
    y_values = data['RMSD']

    fig, ax = plt.subplots()
    # Add a horizontal dotted line at y=2 with grey color
    ax.axhline(y=2, color='grey', linestyle='--', linewidth=1)  # Grey color
    # Plot the filtered energy values on top
    ax.scatter(x_values, y_values)
    #ax.scatter(x_values[x_values < -80], y_values[x_values < -80], c='indianred')
    #ax.scatter(x_values[(x_values < -41.5) & (y_values < 4)], y_values[(x_values < -41.5) & (y_values < 4)], c='indianred')
    ax.scatter(x_values[y_values < 2], y_values[y_values < 2], c='green')
    ax.set_xlim(data['Energy'].min() - 5, 0)

    # Add titles and labels
    ax.set_title('Evolving population (PDB ' + PDBname + ')')
    ax.set_xlabel("Energy (kcal/mol)")
    ax.set_ylabel("RMSD (Å)")

    # Optionally, add a legend
    #ax.legend()

    # plt.scatter(x_values, y_values)
    # plt.title("Evolving population")
    # plt.xlabel("Energy (kcal/mol)")
    # plt.ylabel("RMSD (Å)")
    # plt.xlim(data['Energy'].min() - 5, 0)
    # plt.axhline(y=2, color='grey', linestyle='--', linewidth=1)  # 'r' for red color, '--' for dotted line
    plt.savefig('data_distribution/RMSDvsEnergy_' + PDBname + '.png')

    histplot = px.density_heatmap(data, x="Energy", y="RMSD", nbinsx=15, nbinsy=15, marginal_x="histogram", marginal_y="histogram")
    histplot.write_image('data_distribution/EnergyRMSD_heatmap_' + PDBname + '.png')

def plot_datasetRMSDdistribution(args):

    #train_set_X, train_set_y, test_set_X, test_set_y, features_X = split_set(args)
    train_set_X, train_set_y, val_set_X, val_set_y, test_set_X, test_set_y, features_X = split_set(args)
    train_set_X_df = pd.DataFrame(train_set_X, columns=features_X)
    train_set_y_df = pd.DataFrame(train_set_y, columns=['Label'])
    test_set_X_df = pd.DataFrame(test_set_X, columns=features_X)
    test_set_y_df = pd.DataFrame(test_set_y, columns=['Label'])

    #histplot = sns.histplot(data=dataset['RMSD'], kde=True, stat='percent')
    plt.figure(figsize=(8, 3))
    ax1 = train_set_X_df['RMSD'].hist(bins=40) #, density=True, stacked=True, label='Energy+20')
    ax1.set_xlabel('RMSD (Å)', fontsize=12)
    ax1.set_ylabel('Number of poses', fontsize=12)
    #set_title('RMSD distribution (E cryst. struct. +50)')
    #ax1.set_xlabel('RMSD (Å)')
    #ax1.set_ylabel('Number of poses')
    #histplot.set(xlabel='RMSD (Å)', ylabel='Number of poses')
    #fig = histplot.get_figure()
    plt.tight_layout()
    plt.savefig('data_distribution/No_vs_RMSD.png')
    plt.clf()
    
    #histplot = sns.histplot(data=dataset['Label'], kde=True, stat='percent')
    ax2 = train_set_X_df['Label'].hist(bins=40) #, density=True, stacked=True, label='Energy+20')
    ax2.set_xlabel('label', fontsize=12)
    ax2.set_ylabel('Number of poses', fontsize=12)
    #set_title('Normalized RMSD distribution (E cryst. struct. +50)')
    #ax2.set_xlabel('Activity label')
    #ax2.set_ylabel('Number of poses'
    #plt.tight_layout()
    #plt.show()
    #histplot.set(xlabel='label', ylabel='Number of poses')
    #fig = histplot.get_figure()
    plt.tight_layout()
    plt.savefig('data_distribution/No_vs_label.png')

    # RMSD train set
    histplot = sns.histplot(data=train_set_X_df['RMSD'], kde=True, stat='percent')
    # dataset['RMSD'].hist(bins=40, ax=ax1) #, density=True, stacked=True, label='Energy+20')
    # set_title('RMSD distribution (E cryst. struct. +50)')
    # ax1.set_xlabel('RMSD (Å)')
    # ax1.set_ylabel('Number of poses')
    histplot.set(title='RMSD distribution (train set)', xlabel='RMSD (Å)', ylabel='Percent (%)')
    fig = histplot.get_figure()
    fig.savefig('data_distribution/datasetRMSDdistrubution_trainset.png')

    # Normalized RMSD train set
    fig.clf()
    histplot = sns.histplot(data=train_set_y_df['Label'], kde=True, stat='percent')
    # dataset['Label'].hist(bins=40, ax=ax2) #, density=True, stacked=True, label='Energy+20')
    # set_title('Normalized RMSD distribution (E cryst. struct. +50)')
    # ax2.set_xlabel('Activity label')
    # ax2.set_ylabel('Number of poses'
    # plt.tight_layout()
    # plt.show()
    histplot.set(title='Normalized RMSD distribution (train set)', xlabel='Normalized RMSD', ylabel='Percent (%)')
    fig = histplot.get_figure()
    #fig.savefig(cwd + '\Figures\DataExploration\\' + dataset_name + '_Normalized_RMSD_distribution.png')
    plt.savefig('data_distribution/datasetNormalizedRMSDdistrubution_trainset.png')

    # RMSD test set
    fig.clf()
    histplot = sns.histplot(data=test_set_X_df['RMSD'], kde=True, stat='percent')
    histplot.set(title='RMSD distribution (test set)', xlabel='RMSD (Å)', ylabel='Percent (%)')
    fig = histplot.get_figure()
    fig.savefig('data_distribution/datasetRMSDdistrubution_testset.png')

    # Normalized RMSD test set
    fig.clf()
    histplot = sns.histplot(data=test_set_y_df['Label'], kde=True, stat='percent')
    histplot.set(title='Normalized RMSD distribution (test set)', xlabel='Normalized RMSD', ylabel='Percent (%)')
    fig = histplot.get_figure()
    plt.savefig('data_distribution/datasetNormalizedRMSDdistrubution_testset.png')

    # Energy train set
    fig.clf()
    histplot = sns.histplot(data=train_set_X_df['Energy'], bins=1000, kde=True, stat='percent')
    histplot.set(title='Energy distribution (train set)', xlabel='Energy (kcal/mol)', ylabel='Percent (%)')
    fig = histplot.get_figure()
    fig.savefig('data_distribution/datasetEnergyDistrubution_trainset.png')

    # Energy test set
    fig.clf()
    histplot = sns.histplot(data=test_set_X_df['Energy'], bins=1000, kde=True, stat='percent')
    histplot.set(title='Energy distribution (test set)', xlabel='Energy (kcal/mol)', ylabel='Percent (%)')
    fig = histplot.get_figure()
    fig.savefig('data_distribution/datasetEnergyDistrubution_testset.png')

def plot_datasetEnergyRMSDdistribution(args):

    train_set_X, train_set_y, test_set_X, test_set_y, features_X = split_set(args)
    train_set_X_df = pd.DataFrame(train_set_X, columns=features_X)
    train_set_y_df = pd.DataFrame(train_set_y, columns=['Label'])
    test_set_X_df = pd.DataFrame(test_set_X, columns=features_X)
    test_set_y_df = pd.DataFrame(test_set_y, columns=['Label'])

    #df = px.train_set_X_df.tips()
    histplot_train = px.density_heatmap(train_set_X_df, x="Energy", y="RMSD", marginal_x="histogram", marginal_y="histogram")
    #histplot = sns.histplot(data=train_set_X_df, x=train_set_X_df['RMSD'], y=train_set_X_df['Energy'], bins=10, pmax=1, cbar=True)
    #fig = histplot.get_figure()
    histplot_train.write_image('data_distribution/datasetEnergyRMSDdistrubution_trainset.png')

    histplot_test = px.density_heatmap(test_set_X_df, x="Energy", y="RMSD", marginal_x="histogram", marginal_y="histogram")
    histplot_test.write_image('data_distribution/datasetEnergyRMSDdistrubution_testset.png')

def plot_scatterplot_RMSDvsEnergy(args):

    def scatter_hist(x, y, ax, ax_histx, ax_histy, color):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        ax.scatter(x, y, color=color)

        # now determine nice limits by hand:
        binwidth = 0.25
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax / binwidth) + 1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, color=color, density='True')
        ax_histy.hist(y, orientation='horizontal', color=color, density='True')

    train_set_X, train_set_y, test_set_X, test_set_y, features_X = split_set(args)
    train_set_X_df = pd.DataFrame(train_set_X, columns=features_X)
    train_set_y_df = pd.DataFrame(train_set_y, columns=['Label'])
    test_set_X_df = pd.DataFrame(test_set_X, columns=features_X)
    test_set_y_df = pd.DataFrame(test_set_y, columns=['Label'])

    good_poses_train = train_set_X_df[train_set_X_df['RMSD'] <= 2]
    bad_poses_train = train_set_X_df[train_set_X_df['RMSD'] > 2]

    good_poses_test = test_set_X_df[test_set_X_df['RMSD'] <= 2]
    bad_poses_test = test_set_X_df[test_set_X_df['RMSD'] > 2]



    # Start with a square Figure.
    fig = plt.figure(figsize=(6, 6))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal Axes and the main Axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    # Draw the scatter plot and marginals.
    scatter_hist(good_poses_train['Energy'], good_poses_train['RMSD'], ax, ax_histx, ax_histy, "green")
    scatter_hist(bad_poses_train['Energy'], bad_poses_train['RMSD'], ax, ax_histx, ax_histy, "indianred")
    plt.savefig('graphs/RMSDvsEnergy_mismatch.png')


def plot_accuracy_not_working_well(args):

# ----------- Load the model

    if args.model == 'LR':
        with open(cwd + '/models/' + args.model_name, 'rb') as pickle_file:
            model = pickle.load(pickle_file)
            graph_title = "LR"
    elif args.model == 'XGBr':
        model = xgb.XGBRegressor()
        model.load_model(cwd + '/models/' + args.model_name)
        graph_title = "XGBr"

# ----------- Load the data to test the model on

    if args.set == 'Astex':
        dir = '/AstexSet_top10/'
        graph_title = "Astex set" + graph_title
    elif args.set == 'Fitted':
        dir = '/FittedSet_top10/'
        graph_title = "Fitted set" + graph_title
    elif args.set == 'PDBBind':
        dir = '/PDBBindSet_top10/'
        graph_title = "PDB set" + graph_title
    elif args.set == 'CSV':
        dir = '/CSV/'
        graph_title = "PDB full set" + graph_title
    elif args.set == 'TestSet':
        dir = '/TestSet/'
        graph_title = "Test set" + graph_title

    total = 0 # Total number of PDBs in a set

    # Out of 10 runs, how many PDBs have the lowest predicted RMSD below 2A
    #predicted_accuracy = 0
    # Out of 10 runs, for the lowest predicted RMSD (if below 2A), how many PDBs have the corresponding RMSD also below 2A
    #corresponding_accuracy = 0

    # Accuracy by looking at the RMSD value corresponding to the lowest energy out of 10 runs
    energy_raw = {0.0: 0, 0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0, 1.25: 0, 1.5: 0, 1.75: 0, 2.0: 0, 2.25: 0, 2.5: 0, 2.75: 0, 3.0: 0, 3.25: 0, 3.5: 0, 3.75: 0, 4.0: 0, 4.25: 0, 4.5: 0, 4.75: 0, 5.0: 0}

    # Accuracy by looking at the lowest RMSD out of 10 runs
    rmsd_raw = {0.0: 0, 0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0, 1.25: 0, 1.5: 0, 1.75: 0, 2.0: 0, 2.25: 0, 2.5: 0, 2.75: 0, 3.0: 0, 3.25: 0, 3.5: 0, 3.75: 0, 4.0: 0, 4.25: 0, 4.5: 0, 4.75: 0, 5.0: 0}

    # Accuracy by looking at the *predicted* RMSD value corresponding to the lowest energy out of 10 runs
    energy_pred_raw = {0.0: 0, 0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0, 1.25: 0, 1.5: 0, 1.75: 0, 2.0: 0, 2.25: 0, 2.5: 0, 2.75: 0, 3.0: 0, 3.25: 0, 3.5: 0, 3.75: 0, 4.0: 0, 4.25: 0, 4.5: 0, 4.75: 0, 5.0: 0}

    # Accuracy by looking at the lowest *predicted* RMSD out of 10 runs
    rmsd_pred_raw = {0.0: 0, 0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0, 1.25: 0, 1.5: 0, 1.75: 0, 2.0: 0, 2.25: 0, 2.5: 0, 2.75: 0, 3.0: 0, 3.25: 0, 3.5: 0, 3.75: 0, 4.0: 0, 4.25: 0, 4.5: 0, 4.75: 0, 5.0: 0}

    # Accuracy by looking at the lowest *predicted* RMSD out of 10 runs
    rmsd_random_raw = {0.0: 0, 0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0, 1.25: 0, 1.5: 0, 1.75: 0, 2.0: 0, 2.25: 0, 2.5: 0, 2.75: 0, 3.0: 0, 3.25: 0, 3.5: 0, 3.75: 0, 4.0: 0, 4.25: 0, 4.5: 0, 4.75: 0, 5.0: 0}

    # Out of 10 runs, how many PDBs have the lowest predicted RMSD below 2A
    predicted_accuracy = {0.0: 0, 0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0, 1.25: 0, 1.5: 0, 1.75: 0, 2.0: 0, 2.25: 0, 2.5: 0, 2.75: 0, 3.0: 0, 3.25: 0, 3.5: 0, 3.75: 0, 4.0: 0, 4.25: 0, 4.5: 0, 4.75: 0, 5.0: 0}

    # Out of 10 runs, for the lowest predicted RMSD (if below 2A), how many PDBs have the corresponding RMSD also below 2A
    corresponding_accuracy = {0.0: 0, 0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0, 1.25: 0, 1.5: 0, 1.75: 0, 2.0: 0, 2.25: 0, 2.5: 0, 2.75: 0, 3.0: 0, 3.25: 0, 3.5: 0, 3.75: 0, 4.0: 0, 4.25: 0, 4.5: 0, 4.75: 0, 5.0: 0}

    # Out of 10 runs, if you pick a random pose out of the 10 runs, how many PDBs have the corresponding RMSD also below 2A
    # This will give an estimate of what the lower bound accuracy is and tell us how much better than random the model is
    random_accuracy = {0.0: 0, 0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0, 1.25: 0, 1.5: 0, 1.75: 0, 2.0: 0, 2.25: 0, 2.5: 0, 2.75: 0, 3.0: 0, 3.25: 0, 3.5: 0, 3.75: 0, 4.0: 0, 4.25: 0, 4.5: 0, 4.75: 0, 5.0: 0}

    # Features needed for predictions
    # Type 1
    #feature_names = ['Energy','MatchScore','GLY_BB_vdW','GLY_BB_ES','GLY_BB_HB','ALA_BB_vdW','ALA_BB_ES','ALA_BB_HB','ALA_SC_vdW','ALA_SC_ES','VAL_BB_vdW','VAL_BB_ES','VAL_BB_HB','VAL_SC_vdW','VAL_SC_ES','LEU_BB_vdW','LEU_BB_ES','LEU_BB_HB','LEU_SC_vdW','LEU_SC_ES','ILE_BB_vdW','ILE_BB_ES','ILE_BB_HB','ILE_SC_vdW','ILE_SC_ES','PRO_BB_vdW','PRO_BB_ES','PRO_BB_HB','PRO_SC_vdW','PRO_SC_ES','ASP_BB_vdW','ASP_BB_ES','ASP_BB_HB','ASP_SC_vdW','ASP_SC_ES','ASP_SC_HB','GLU_BB_vdW','GLU_BB_ES','GLU_BB_HB','GLU_SC_vdW','GLU_SC_ES','GLU_SC_HB','ASN_BB_vdW','ASN_BB_ES','ASN_BB_HB','ASN_SC_vdW','ASN_SC_ES','ASN_SC_HB','GLN_BB_vdW','GLN_BB_ES','GLN_BB_HB','GLN_SC_vdW','GLN_SC_ES','GLN_SC_HB','LYS_BB_vdW','LYS_BB_ES','LYS_BB_HB','LYS_SC_vdW','LYS_SC_ES','LYS_SC_HB','ARG_BB_vdW','ARG_BB_ES','ARG_BB_HB','ARG_SC_vdW','ARG_SC_ES','ARG_SC_HB','TRP_BB_vdW','TRP_BB_ES','TRP_BB_HB','TRP_SC_vdW','TRP_SC_ES','TYR_BB_vdW','TYR_BB_ES','TYR_BB_HB','TYR_SC_vdW','TYR_SC_ES','TYR_SC_HB','PHE_BB_vdW','PHE_BB_ES','PHE_BB_HB','PHE_SC_vdW','PHE_SC_ES','HIS_BB_vdW','HIS_BB_ES','HIS_BB_HB','HIS_SC_vdW','HIS_SC_ES','HIS_SC_HB','CYS_BB_vdW','CYS_BB_ES','CYS_BB_HB','CYS_SC_vdW','CYS_SC_ES','CYS_SC_HB','SER_BB_vdW','SER_BB_ES','SER_BB_HB','SER_SC_vdW','SER_SC_ES','SER_SC_HB','THR_BB_vdW','THR_BB_ES','THR_BB_HB','THR_SC_vdW','THR_SC_ES','THR_SC_HB','MET_BB_vdW','MET_BB_ES','MET_BB_HB','MET_SC_vdW','MET_SC_ES','MET_SC_HB','Bonds','Angles','Torsions','Angles_M','Tors_M','vdW_14','Elec_14','vdW_15','Elec_15','vdW','Elec','HBonds','HB_M','Bond_M','Wat_vdW','Wat_Elec','Wat_HB','No_Wat']
    # Type 2
    #feature_names = ['MatchScore','GLY_BB_vdW','GLY_BB_ES','GLY_BB_HB','ALA_BB_vdW','ALA_BB_ES','ALA_BB_HB','ALA_SC_vdW','ALA_SC_ES','VAL_BB_vdW','VAL_BB_ES','VAL_BB_HB','VAL_SC_vdW','VAL_SC_ES','LEU_BB_vdW','LEU_BB_ES','LEU_BB_HB','LEU_SC_vdW','LEU_SC_ES','ILE_BB_vdW','ILE_BB_ES','ILE_BB_HB','ILE_SC_vdW','ILE_SC_ES','PRO_BB_vdW','PRO_BB_ES','PRO_BB_HB','PRO_SC_vdW','PRO_SC_ES','ASP_BB_vdW','ASP_BB_ES','ASP_BB_HB','ASP_SC_vdW','ASP_SC_ES','ASP_SC_HB','GLU_BB_vdW','GLU_BB_ES','GLU_BB_HB','GLU_SC_vdW','GLU_SC_ES','GLU_SC_HB','ASN_BB_vdW','ASN_BB_ES','ASN_BB_HB','ASN_SC_vdW','ASN_SC_ES','ASN_SC_HB','GLN_BB_vdW','GLN_BB_ES','GLN_BB_HB','GLN_SC_vdW','GLN_SC_ES','GLN_SC_HB','LYS_BB_vdW','LYS_BB_ES','LYS_BB_HB','LYS_SC_vdW','LYS_SC_ES','LYS_SC_HB','ARG_BB_vdW','ARG_BB_ES','ARG_BB_HB','ARG_SC_vdW','ARG_SC_ES','ARG_SC_HB','TRP_BB_vdW','TRP_BB_ES','TRP_BB_HB','TRP_SC_vdW','TRP_SC_ES','TYR_BB_vdW','TYR_BB_ES','TYR_BB_HB','TYR_SC_vdW','TYR_SC_ES','TYR_SC_HB','PHE_BB_vdW','PHE_BB_ES','PHE_BB_HB','PHE_SC_vdW','PHE_SC_ES','HIS_BB_vdW','HIS_BB_ES','HIS_BB_HB','HIS_SC_vdW','HIS_SC_ES','HIS_SC_HB','CYS_BB_vdW','CYS_BB_ES','CYS_BB_HB','CYS_SC_vdW','CYS_SC_ES','CYS_SC_HB','SER_BB_vdW','SER_BB_ES','SER_BB_HB','SER_SC_vdW','SER_SC_ES','SER_SC_HB','THR_BB_vdW','THR_BB_ES','THR_BB_HB','THR_SC_vdW','THR_SC_ES','THR_SC_HB','MET_BB_vdW','MET_BB_ES','MET_BB_HB','MET_SC_vdW','MET_SC_ES','MET_SC_HB','Bonds','Angles','Torsions','Angles_M','Tors_M','vdW_14','Elec_14','vdW_15','Elec_15','vdW','Elec','HBonds','HB_M','Bond_M','Wat_vdW','Wat_Elec','Wat_HB','No_Wat']
    feature_names = ['GLY_BB_vdW','GLY_BB_ES','GLY_BB_HB','ALA_BB_vdW','ALA_BB_ES','ALA_BB_HB','ALA_SC_vdW','ALA_SC_ES','VAL_BB_vdW','VAL_BB_ES','VAL_BB_HB','VAL_SC_vdW','VAL_SC_ES','LEU_BB_vdW','LEU_BB_ES','LEU_BB_HB','LEU_SC_vdW','LEU_SC_ES','ILE_BB_vdW','ILE_BB_ES','ILE_BB_HB','ILE_SC_vdW','ILE_SC_ES','PRO_BB_vdW','PRO_BB_ES','PRO_BB_HB','PRO_SC_vdW','PRO_SC_ES','ASP_BB_vdW','ASP_BB_ES','ASP_BB_HB','ASP_SC_vdW','ASP_SC_ES','ASP_SC_HB','GLU_BB_vdW','GLU_BB_ES','GLU_BB_HB','GLU_SC_vdW','GLU_SC_ES','GLU_SC_HB','ASN_BB_vdW','ASN_BB_ES','ASN_BB_HB','ASN_SC_vdW','ASN_SC_ES','ASN_SC_HB','GLN_BB_vdW','GLN_BB_ES','GLN_BB_HB','GLN_SC_vdW','GLN_SC_ES','GLN_SC_HB','LYS_BB_vdW','LYS_BB_ES','LYS_BB_HB','LYS_SC_vdW','LYS_SC_ES','LYS_SC_HB','ARG_BB_vdW','ARG_BB_ES','ARG_BB_HB','ARG_SC_vdW','ARG_SC_ES','ARG_SC_HB','TRP_BB_vdW','TRP_BB_ES','TRP_BB_HB','TRP_SC_vdW','TRP_SC_ES','TYR_BB_vdW','TYR_BB_ES','TYR_BB_HB','TYR_SC_vdW','TYR_SC_ES','TYR_SC_HB','PHE_BB_vdW','PHE_BB_ES','PHE_BB_HB','PHE_SC_vdW','PHE_SC_ES','HIS_BB_vdW','HIS_BB_ES','HIS_BB_HB','HIS_SC_vdW','HIS_SC_ES','HIS_SC_HB','CYS_BB_vdW','CYS_BB_ES','CYS_BB_HB','CYS_SC_vdW','CYS_SC_ES','CYS_SC_HB','SER_BB_vdW','SER_BB_ES','SER_BB_HB','SER_SC_vdW','SER_SC_ES','SER_SC_HB','THR_BB_vdW','THR_BB_ES','THR_BB_HB','THR_SC_vdW','THR_SC_ES','THR_SC_HB','MET_BB_vdW','MET_BB_ES','MET_BB_HB','MET_SC_vdW','MET_SC_ES','MET_SC_HB','Bonds','Angles','Torsions','Angles_M','Tors_M','vdW_14','Elec_14','vdW_15','Elec_15','vdW','Elec','HBonds','HB_M','Bond_M','Wat_vdW','Wat_Elec','Wat_HB','No_Wat']
    # Type 3
    #feature_names = ['GLY_BB_vdW','GLY_BB_ES','GLY_BB_HB','ALA_BB_vdW','ALA_BB_ES','ALA_BB_HB','ALA_SC_vdW','ALA_SC_ES','VAL_BB_vdW','VAL_BB_ES','VAL_BB_HB','VAL_SC_vdW','VAL_SC_ES','LEU_BB_vdW','LEU_BB_ES','LEU_BB_HB','LEU_SC_vdW','LEU_SC_ES','ILE_BB_vdW','ILE_BB_ES','ILE_BB_HB','ILE_SC_vdW','ILE_SC_ES','PRO_BB_vdW','PRO_BB_ES','PRO_BB_HB','PRO_SC_vdW','PRO_SC_ES','ASP_BB_vdW','ASP_BB_ES','ASP_BB_HB','ASP_SC_vdW','ASP_SC_ES','ASP_SC_HB','GLU_BB_vdW','GLU_BB_ES','GLU_BB_HB','GLU_SC_vdW','GLU_SC_ES','GLU_SC_HB','ASN_BB_vdW','ASN_BB_ES','ASN_BB_HB','ASN_SC_vdW','ASN_SC_ES','ASN_SC_HB','GLN_BB_vdW','GLN_BB_ES','GLN_BB_HB','GLN_SC_vdW','GLN_SC_ES','GLN_SC_HB','LYS_BB_vdW','LYS_BB_ES','LYS_BB_HB','LYS_SC_vdW','LYS_SC_ES','LYS_SC_HB','ARG_BB_vdW','ARG_BB_ES','ARG_BB_HB','ARG_SC_vdW','ARG_SC_ES','ARG_SC_HB','TRP_BB_vdW','TRP_BB_ES','TRP_BB_HB','TRP_SC_vdW','TRP_SC_ES','TYR_BB_vdW','TYR_BB_ES','TYR_BB_HB','TYR_SC_vdW','TYR_SC_ES','TYR_SC_HB','PHE_BB_vdW','PHE_BB_ES','PHE_BB_HB','PHE_SC_vdW','PHE_SC_ES','HIS_BB_vdW','HIS_BB_ES','HIS_BB_HB','HIS_SC_vdW','HIS_SC_ES','HIS_SC_HB','CYS_BB_vdW','CYS_BB_ES','CYS_BB_HB','CYS_SC_vdW','CYS_SC_ES','CYS_SC_HB','SER_BB_vdW','SER_BB_ES','SER_BB_HB','SER_SC_vdW','SER_SC_ES','SER_SC_HB','THR_BB_vdW','THR_BB_ES','THR_BB_HB','THR_SC_vdW','THR_SC_ES','THR_SC_HB','MET_BB_vdW','MET_BB_ES','MET_BB_HB','MET_SC_vdW','MET_SC_ES','MET_SC_HB']
    # Type 4
    #feature_names = ['vdW_15', 'ARG_SC_HB', 'vdW_14', 'Energy', 'Wat_HB', 'Elec', 'No_Wat', 'LEU_SC_vdW', 'HBonds', 'Angles', 'ASN_BB_vdW', 'ALA_SC_vdW', 'MatchScore', 'ASP_SC_HB', 'GLU_BB_vdW']


    for file in os.listdir(cwd + dir):
        if os.path.isfile(cwd + dir + file):
            print(file)
            pdb = pd.read_csv(cwd + dir + file, index_col=False)

            total = total + 1

            # Find the RMSD that corresponds to the minimum energy
            RMSD = pdb.loc[pdb['Energy'].idxmin(), 'RMSD']

            # Find the lowest RMSD
            RMSD_min = pdb['RMSD'].min()
            print(file.split('_')[0], ", Min RMSD:", RMSD_min, ", index:", pdb['RMSD'].idxmin())

            # Scan for RMSD thresholds between 0.00 and 5.00, incrementing by 0.25 (total of 21 threshold points)
            for threshold in np.linspace(0, 5, 21):
                if (RMSD <= threshold):
                    energy_raw[threshold] = energy_raw[threshold] + 1
                if (RMSD_min <= threshold):
                    rmsd_raw[threshold] = rmsd_raw[threshold] + 1

            ## Predict the activity label (so the value between 0-1) using the ML model
            pred_accuracy = 1
            if pred_accuracy == 1:
                if args.normalize == 'yes':
                    pdb_drop = drop_features(pdb)  # drop some features because different models were trained using different number of features
                    pdb_normalized = preprocessing.normalize(pdb_drop)  # normalize the dataset since it was trained on normalized data
                    activity_pred = model.predict(pdb_normalized)  # predict the output
                else:
                    pdb_drop = drop_features(pdb)
                    activity_pred = model.predict(pdb[feature_names])

                RMSD_pred_array = []  # store the predicted RMSD values for the PDBs in this file (so for the protein)

                for i in activity_pred:
                    RMSD_pred = act_to_RMSD(i)  # convert the activity to the RMSD for each PDB in activity_pred
                    RMSD_pred_array.append(RMSD_pred)  # add the predicted RMSD value to another array (RMSD_pred_array)

                pdb['Activity_pred'] = pd.Series(activity_pred)  # add the Activity array to a dictionary
                pdb['RMSD_pred'] = pd.Series(RMSD_pred_array)  # add the predicted RMSD array to a dictionary

                # Find the RMSD_pred that corresponds to the minimum energy
                RMSD_pred_best_E = pdb.loc[pdb['Energy'].idxmin(), 'RMSD_pred']  # based on the best energy, what is the predicted RMSD
                corresponding_RMSD = pdb.loc[pdb['RMSD_pred'].idxmin(), 'RMSD']  # based on the minimum predicted RMSD, what is the RMSD

                # Find the lowest RMSD_pred
                RMSD_pred_min = pdb['RMSD_pred'].min()

                # Pick a random predicted RMSD
                RMSD_random = random.choice(pdb['RMSD'])
                # RMSD_pred_random = random.choice(pdb['RMSD_pred']) # select a random entry from the column RMSD_pred
                # RMSD_random = pdb.loc[random.choice(pdb['RMSD_pred']), 'RMSD'] # based on the random predicted RMSD, what is the RMSD

                for threshold in np.linspace(0, 5, 21):  # RMSD values between 0 and 5, with increments of 0.25

                    # Accuracy for the predicted RMSD
                    if RMSD_pred_min <= threshold:
                        predicted_accuracy[threshold] = predicted_accuracy[threshold] + 1

                    # Does the corresponding RMSD represent a good pose (below 2A)
                    if corresponding_RMSD <= threshold:
                        corresponding_accuracy[threshold] = corresponding_accuracy[threshold] + 1

                    # Accuracy of the random RMSD
                    if RMSD_random <= threshold:
                        random_accuracy[threshold] = random_accuracy[threshold] + 1

                # Scan for RMSD thresholds between 0.00 and 5.00, incrementing by 0.25 (total of 21 threshold points such that 0.00 and 5.00 are included)
                for threshold in np.linspace(0, 5, 21):
                    if (RMSD_pred_best_E <= threshold):
                        energy_pred_raw[threshold] = energy_pred_raw[threshold] + 1
                    if (RMSD_pred_min <= threshold):
                        rmsd_pred_raw[threshold] = rmsd_pred_raw[threshold] + 1

    # Calculate the percentages for each threshold value
    energy_percent = {key: value * 100 / total for key, value in energy_raw.items()}
    rmsd_percent = {key: value * 100 / total for key, value in rmsd_raw.items()}

    energy_pred_percent = {key: value * 100 / total for key, value in energy_pred_raw.items()}
    rmsd_pred_percent = {key: value * 100 / total for key, value in rmsd_pred_raw.items()}

    predicted_accuracy_percent = {key: value * 100 / total for key, value in predicted_accuracy.items()}
    corresponding_accuracy_percent = {key: value * 100 / total for key, value in corresponding_accuracy.items()}

    random_accuracy_percent = {key: value * 100 / total for key, value in random_accuracy.items()}

    ## GRAPH
    x = list(energy_percent.keys())
    y1 = list(energy_percent.values())
    y2 = list(rmsd_percent.values())

    print("x-axis: ", list(energy_percent.keys()))
    print("energy: ", list(energy_percent.values()))
    print("rmsd: ", list(rmsd_percent.values()))
    print("random: ", list(random_accuracy_percent.values()))
    print("ML: ", list(corresponding_accuracy_percent.values()))

    plt.plot(x, y1, label="Best Energy", color='blue')
    plt.plot(x, y2, label="Best RMSD", color='red')

    if pred_accuracy == 1:
        y3 = list(energy_pred_percent.values())
        y4 = list(rmsd_pred_percent.values())
        y5 = list(predicted_accuracy_percent.values())
        y6 = list(corresponding_accuracy_percent.values())
        y7 = list(random_accuracy_percent.values())

        # plt.plot(x, y3, label = "Predicted Energy")
        # plt.plot(x, y4, label = "Predicted RMSD")
        # plt.plot(x, y5, label = "Predicted RMSD")
        plt.plot(x, y6, label="Predicted RMSD", color='green')
        plt.plot(x, y7, label="Random RMSD", color='orange')

    ax = plt.gca()

    font1 = {'family': 'arial', 'color': 'black', 'size': 14}
    font2 = {'family': 'arial', 'color': 'black', 'size': 12}

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    # Set the x axis label of the current axis.
    plt.xlabel('RMSD (Angstrom)', fontdict=font1)
    # Set the y axis label of the current axis.
    plt.ylabel('Accuracy (%)', fontdict=font1)
    # show a legend on the plot
    plt.legend(prop={'size': 12})
    # show gridlines
    plt.grid()
    # Set a title of the current axes.
    plt.title(graph_title, fontdict=font1)
    # Add extra white space at the bottom so that the label is not cut off
    plt.subplots_adjust(bottom=0.13)

    #plt.show()

    # Save the figure.
    plt.savefig(cwd + '/graphs/' + args.model_name + graph_title + '.png')

    #y_pred = model.predict(X_test)

    #accuracy = accuracy_score(y_test, y_pred)

    #print(accuracy)

    # plt.title("Normalized RMSD values", fontdict = font1)
    # plt.xlabel("RMSD (Å)", fontdict = font2)
    # plt.ylabel("Normalized RMSD", fontdict = font2)

def predict_accuracy(args):

    train_set_X, train_set_y, test_set_X, test_set_y, features = split_set(args)

    train_set_y.loc[train_set_X['RMSD'] <= 2] = 1
    train_set_y.loc[train_set_X['RMSD'] > 2] = 0

    test_set_y.loc[test_set_X['RMSD'] <= 2] = 1
    test_set_y.loc[test_set_X['RMSD'] > 2] = 0

    features_to_drop = ['RMSD']
    RMSDs_train = train_set_X['RMSD']
    RMSDs_test = test_set_X['RMSD']
    train_set_X = train_set_X.drop(columns=features_to_drop, axis=1)
    test_set_X = test_set_X.drop(columns=features_to_drop, axis=1)

    if args.model == 'LRc':
        with open(cwd + '/models/' + args.model_name, 'rb') as pickle_file:
            model = pickle.load(pickle_file)

    # make predictions for train data
    y_pred_train = model.predict(train_set_X)
    predictions_train = [round(value) for value in y_pred_train]
    # evaluate predictions
    accuracy_train = accuracy_score(train_set_y, predictions_train)
    print('| Model accuracy on the train set: %.2f%%                                                                           |' % (accuracy_train * 100.0))

    # make predictions for test data
    y_pred_test = model.predict(test_set_X)
    predictions_test = [round(value) for value in y_pred_test]
    # evaluate predictions
    accuracy_test = accuracy_score(test_set_y, predictions_test)
    print('| Model accuracy on the test set: %.2f%%                                                                            |' % (accuracy_test * 100.0))

    #predictions_train_series = pd.Series(predictions_train)
    #RMSD_pred = pd.concat([RMSDs_train.reset_index(drop=True), pd.Series(predictions_train, name='Label')], axis=1)
    RMSD_pred = pd.concat([RMSDs_test.reset_index(drop=True), pd.Series(predictions_test, name='Label')], axis=1)
    #RMSD_pred = pd.concat([RMSDs_test, predictions_test], axis=1)
    #print(RMSD_pred)
    #accuracies = pd.DataFrame(columns=['RMSD threshold', 'Good', 'Bad'])
    #accuracies['RMSD threshold'] = np.arange(0, 5.25, 0.25)

    thresholds = [0.25, 0.5, 0.75, 1, 2, 3, 4, 5]
    thresholds = np.arange(0, 5.5, 0.5)

    # Calculate the number of '1's for each threshold
    # for threshold in thresholds:
    #     count_1 = RMSD_pred[(RMSD_pred['RMSD'] <= threshold) & (RMSD_pred['Label'] == 1)].shape[0]
    #     count_0 = RMSD_pred[(RMSD_pred['RMSD'] <= threshold) & (RMSD_pred['Label'] == 0)].shape[0]
    #     print(f"Number of '1's with RMSD below {threshold}: {count}")

    for i in range(len(thresholds) - 1):
        lower_bound = thresholds[i]
        upper_bound = thresholds[i + 1]

        count_1 = RMSD_pred[(RMSD_pred['RMSD'] >= lower_bound) & (RMSD_pred['RMSD'] < upper_bound) & (RMSD_pred['Label'] == 1)].shape[0]
        count_0 = RMSD_pred[(RMSD_pred['RMSD'] >= lower_bound) & (RMSD_pred['RMSD'] < upper_bound) & (RMSD_pred['Label'] == 0)].shape[0]

        print(f"Number of '1's with RMSD between {lower_bound} and {upper_bound}: {count_1}")
        print(f"Number of '0's with RMSD between {lower_bound} and {upper_bound}: {count_0}")
        acc = count_1 / (count_0 + count_1) * 100
        print(f"Accuracy at RMSD between {lower_bound} and {upper_bound}: {acc}")

def apply_transpose(args):

    coefficients = pd.read_csv('models/LRc_coefficients.csv', index_col=False)
    coef = pd.DataFrame()
    coef.columns = coefficients['features'].values
    coef.iloc[0] = coefficients['B'].values
    print(coef)

    train_list = []
    test_list = []

    graph_set = []  # What data is used to plot the graph (either test, train or both)

    # Use the splitting from https://arxiv.org/html/2308.09639v2#S5
    if os.path.isfile(dir + args.list_trainset) is True and os.path.isfile(dir + args.list_testset) is True:

        print('| Training set selected from %-86s |' % args.list_trainset)
        with open(dir + args.list_trainset, 'rb') as list_f:
            for line in list_f:
                train_list.append(line.strip().decode('UTF-8'))
        print('| Testing set selected from %-86s  |' % args.list_testset)
        with open(dir + args.list_testset, 'rb') as list_f:
            for line in list_f:
                test_list.append(line.strip().decode('UTF-8'))

        if args.test_on == "test":
            graph_set = test_list
        elif args.test_on == "train":
            graph_set = train_list
        elif args.test_on == "full":
            graph_set = test_list + train_list

    else:
        print('| Please provide a train and test list.                                                                      |')

    accuracies = pd.DataFrame(columns=['RMSD threshold', 'Lowest energy', 'Lowest RMSD', 'Random', '+ Coefficients'])
    accuracies['RMSD threshold'] = np.arange(0, 5.25, 0.25)
    accuracies['Lowest energy'] = 0
    accuracies['Lowest RMSD'] = 0
    accuracies['Random'] = 0
    accuracies['Predicted'] = 0

    for pdb_id in graph_set:
        if os.path.isfile(dir + pdb_id + extension):
            pdb = pd.read_csv(dir + pdb_id + extension, index_col=False)
            if len(pdb.index) > 0 :
                pdb = pdb.dropna()  # remove rows with NaN
                total = total + 1

                #pdb_coef = pdb.multiply(coef.iloc[0], axis=1)

                for index, value in accuracies['RMSD threshold'].items():

                    min_energy = PDBdata['Energies'].idxmin()

                    # Lowest energy
                    if PDBdata.loc[min_energy, 'RMSD'] <= value:
                        accuracies.at[index, 'Lowest energy'] += 1

                    # Lowest RMSD
                    if PDBdata['RMSD'].min() <= value:
                        accuracies.at[index, 'Lowest RMSD'] += 1

                    # Random
                    if random['RMSD'].iloc[0] <= value:
                        # accuracies['Random'] = accuracies['Random'] + 1
                        accuracies.at[index, 'Random'] += 1

                    # With modified LRc coefficients
                    if PDBdata.loc[PDBdata['RMSD_ML'].idxmin(), 'RMSD'] <= value:
                        accuracies.at[index, '+ Coefficients'] += 1


    accuracies['Lowest energy'] = (accuracies['Lowest energy'] / total) * 100
    accuracies['Lowest RMSD'] = (accuracies['Lowest RMSD'] / total) * 100
    accuracies['Random'] = (accuracies['Random'] / total) * 100
    accuracies['+ Coefficients'] = (accuracies['+ Coefficients'] / total) * 100

    #print('Accuracies %:')
    #print(accuracies)

    plt.plot(accuracies['RMSD threshold'], accuracies['Lowest energy'], label="Lowest energy", color='blue')
    plt.plot(accuracies['RMSD threshold'], accuracies['Lowest RMSD'], label="Lowest RMSD", color='red')
    plt.plot(accuracies['RMSD threshold'], accuracies['Random'],label="Random", color='orange')
    plt.plot(accuracies['RMSD threshold'], accuracies['+ Coefficients'], label="+ Coefficients", color='green')


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
    plt.legend(prop={'size': 12})
    # show gridlines
    plt.grid()
    # Set a title of the current axes.
    plt.title(args.model + ' ' + args.set, fontdict=font1)
    # Add extra white space at the bottom so that the label is not cut off
    plt.subplots_adjust(bottom=0.13)

    plt.savefig(cwd + '/graphs/' + args.model + '_' + args.set + '.png')

def plot_accuracy_graph(args):
    # This function does not predict the accuracy, only plots the accuracy of a given data set

    # ----------- Load the set to test the model on

    train_list = []
    test_list = []

    graph_set = []  # What data is used to plot the graph (either test, train or both)

    # Use the splitting from https://arxiv.org/html/2308.09639v2#S5
    #if os.path.isfile(args.CSV_data_dir + args.list_trainset) is True and os.path.isfile(args.CSV_data_dir + args.list_testset) is True:
    if os.path.isfile(args.CSV_data_dir + args.list_trainset) is True:

        print('| Training set selected from %-86s |' % args.list_trainset)
        with open(args.CSV_data_dir + args.list_trainset, 'rb') as list_f:
            for line in list_f:
                train_list.append(line.strip().decode('UTF-8'))

        # if args.teston == "test":
        #     graph_set = test_list
        # elif args.teston == "train":
        #     graph_set = train_list
        # elif args.list_testset == "full":
        #     graph_set = test_list + train_list

    else:
        print('| Please provide a train and test list. |')

    # ----------- Load the data to test the model on

    if args.set == 'Astex':
        dir = '/AstexSet_top10/'
        graph_title = "Astex set"
    elif args.set == 'Fitted':
        dir = '/FittedSet_top10/'
        graph_title = "Fitted set"
    elif args.set == 'PDBBind':
        dir = '/PDBBindSet_top10/'
        graph_title = "PDB set"
    elif args.set == 'PDB_10':
        dir = '/PDB_10/'
        graph_title = "PDB set mismatch"
    elif args.set == 'PDB_test':
        dir = '/PDB_10/'
        graph_title = "PDB set mismatch"
    elif args.set == 'CSV':
        dir = '/CSV/'
        graph_title = "PDB full set"
    elif args.set == 'CSV_vdW05x':
        dir = '/CSV_vdW05x/'
        graph_title = "PDB full set"
    elif args.set == 'CSV_vdW025x':
        dir = '/CSV_vdW025x/'
        graph_title = "PDB full set"
    elif args.set == 'CSV_HB05x':
        dir = '/CSV_HB05x/'
        graph_title = "PDB full set"
    elif args.set == 'CSV_HB025x':
        dir = '/CSV_HB025x/'
        graph_title = "PDB full set"
    elif args.set == 'TestSet':
        dir = '/TestSet/'
        graph_title = "Test set"
    else:
        dir = '/' + args.CSV_data_dir

    accuracies = pd.DataFrame(columns=['RMSD threshold', 'Lowest energy', 'Lowest RMSD', 'Random'])
    accuracies['RMSD threshold'] = np.arange(0, 5.25, 0.25)
    accuracies['Lowest energy'] = 0
    accuracies['Lowest RMSD'] = 0
    accuracies['Random'] = 0

    total = 0

    list_problematic_files = []
    list_ok_files = []

    #for file in os.listdir(cwd + dir):
    for name in train_list:
        #file = name + '_top10.csv'
        file = name + args.CSV_suffix
        if os.path.isfile(cwd + dir + file) and (os.stat(cwd + dir + file).st_size != 0):
            print(name)
            pdb = pd.read_csv(cwd + dir + file, index_col=False)
            if len(pdb.index) == 10:
                total = total + 1

                ### redo
                PDBdata = pd.DataFrame(columns=['Energies', 'RMSD'])
                PDBdata['Energies'] = pdb['Energy']
                PDBdata['RMSD'] = pdb['RMSD']


                random = PDBdata.sample()
                for index, value in accuracies['RMSD threshold'].items():

                    min_energy = PDBdata['Energies'].idxmin()

                    # Lowest energy
                    if PDBdata.loc[min_energy, 'RMSD'] <= value:
                        accuracies.at[index, 'Lowest energy'] += 1

                    # Lowest RMSD
                    if PDBdata['RMSD'].min() <= value:
                        accuracies.at[index, 'Lowest RMSD'] += 1

                    # Random
                    if random['RMSD'].iloc[0] <= value:
                        # accuracies['Random'] = accuracies['Random'] + 1
                        accuracies.at[index, 'Random'] += 1
                list_ok_files.append(name)
            else:
                print(name + ' has ' + str(len(pdb.index)) + ' rows')
                list_problematic_files.append(name)
        else:
            print(name + ' does not exist or is empty')
            list_problematic_files.append(name)

    with open('list_problematic_files.txt', 'w') as f:
        for line in list_problematic_files:
            f.write(f"{line}\n")
    with open('list_10_clean.txt', 'w') as f:
        for line in list_ok_files:
            f.write(f"{line}\n")

    print('Total: ' + str(total))

    print('Accuracies raw:')
    print(accuracies)

    accuracies['Lowest energy'] = (accuracies['Lowest energy'] / total) * 100
    accuracies['Lowest RMSD'] = (accuracies['Lowest RMSD'] / total) * 100
    accuracies['Random'] = (accuracies['Random'] / total) * 100

    print('Accuracies %:')
    print(accuracies)

    plt.plot(accuracies['RMSD threshold'], accuracies['Lowest energy'], label="Lowest energy", color='blue')
    plt.plot(accuracies['RMSD threshold'], accuracies['Lowest RMSD'], label="Lowest RMSD", color='red')
    plt.plot(accuracies['RMSD threshold'], accuracies['Random'], label="Random", color='orange')

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
    plt.legend(prop={'size': 12})
    # show gridlines
    plt.grid()
    # Set a title of the current axes.
    plt.title('Accuracy PDB set', fontdict=font1)
    # Add extra white space at the bottom so that the label is not cut off
    plt.subplots_adjust(bottom=0.13)

    plt.savefig('PDBBind_Accuracy.png')
    #plt.savefig(cwd + '/graphs/Accuracy_PDBset_HB05x' + args.set + '.png')

def plot_accuracy(args):

# ----------- Load the data to test the model on

    # Folders with the top 10 runs
    if args.set == 'Astex':
        dir = 'AstexSet_top10/'
        extension = '_scored-results_bitstring.csv'
        graph_title = "Astex set, " + args.model
    elif args.set == 'Fitted':
        dir = 'FittedSet_top10/'
        graph_title = "Fitted set, " + args.model
        extension = '_score-results_bitstring_top10poses.csv'
    elif args.set == 'PDBBind':
        dir = 'PDB_10/'
        extension = '_top10.csv'
        graph_title = "PDB set, " + args.model
    elif args.set == 'PDBBind_predict':
        dir = 'predictions/'
        extension = '_top10.csv'
        graph_title = "PDB set, " + args.model

    # Folders with mismatch pairs
    elif args.set == 'PDB_mismatch':
        dir = 'PDB_10/'
        extension = '_mismatch.csv'
        graph_title = "PDB set (mismatch)" + args.model


    # idkkkk
    # elif args.set == 'PDBBind_test':
    #     dir = '/PDB_test_10/'
    #     graph_title = "PDB set" + graph_title
    # elif args.set == 'CSV':
    #     dir = '/CSV/'
    #     graph_title = "PDB full set" + graph_title
    # elif args.set == 'TestSet':
    #     dir = '/TestSet/'
    #     graph_title = "Test set" + graph_title

# ----------- Load the set to test the model on

    train_list = []
    test_list = []

    graph_set = []  # What data is used to plot the graph (either test, train or both)

    # Use the splitting from https://arxiv.org/html/2308.09639v2#S5
    if os.path.isfile(dir + args.list_trainset) is True and os.path.isfile(dir + args.list_testset) is True:

        print('| Training set selected from %-86s |' % args.list_trainset)
        with open(dir + args.list_trainset, 'rb') as list_f:
            for line in list_f:
                train_list.append(line.strip().decode('UTF-8'))
        print('| Testing set selected from %-86s  |' % args.list_testset)
        with open(dir + args.list_testset, 'rb') as list_f:
            for line in list_f:
                test_list.append(line.strip().decode('UTF-8'))

        if args.test_on == "test":
            graph_set = test_list
        elif args.test_on == "train":
            graph_set = train_list
        elif args.test_on == "full":
            graph_set = test_list + train_list

    else:
        print('| Please provide a train and test list.                                                                      |')

# ----------- Load the model

    if args.model == 'LR':
        with open(cwd + '/models/' + args.model_name, 'rb') as pickle_file:
            model = pickle.load(pickle_file)
    elif args.model == 'LRc':
        with open(cwd + '/models/' + args.model_name, 'rb') as pickle_file:
            model = pickle.load(pickle_file)
    elif args.model == 'XGBr':
        model = xgb.XGBRegressor()
        model.load_model(cwd + '/models/' + args.model_name)
        #feature_names = model.get_booster().feature_names
    elif args.model == 'XGBc':
        model = xgb.XGBClassifier()
        model.load_model(cwd + '/models/' + args.model_name)
        #feature_names = model.get_booster().feature_names

    # Features needed for predictions
    # Type 1
    #feature_names = ['Energy','MatchScore','GLY_BB_vdW','GLY_BB_ES','GLY_BB_HB','ALA_BB_vdW','ALA_BB_ES','ALA_BB_HB','ALA_SC_vdW','ALA_SC_ES','VAL_BB_vdW','VAL_BB_ES','VAL_BB_HB','VAL_SC_vdW','VAL_SC_ES','LEU_BB_vdW','LEU_BB_ES','LEU_BB_HB','LEU_SC_vdW','LEU_SC_ES','ILE_BB_vdW','ILE_BB_ES','ILE_BB_HB','ILE_SC_vdW','ILE_SC_ES','PRO_BB_vdW','PRO_BB_ES','PRO_BB_HB','PRO_SC_vdW','PRO_SC_ES','ASP_BB_vdW','ASP_BB_ES','ASP_BB_HB','ASP_SC_vdW','ASP_SC_ES','ASP_SC_HB','GLU_BB_vdW','GLU_BB_ES','GLU_BB_HB','GLU_SC_vdW','GLU_SC_ES','GLU_SC_HB','ASN_BB_vdW','ASN_BB_ES','ASN_BB_HB','ASN_SC_vdW','ASN_SC_ES','ASN_SC_HB','GLN_BB_vdW','GLN_BB_ES','GLN_BB_HB','GLN_SC_vdW','GLN_SC_ES','GLN_SC_HB','LYS_BB_vdW','LYS_BB_ES','LYS_BB_HB','LYS_SC_vdW','LYS_SC_ES','LYS_SC_HB','ARG_BB_vdW','ARG_BB_ES','ARG_BB_HB','ARG_SC_vdW','ARG_SC_ES','ARG_SC_HB','TRP_BB_vdW','TRP_BB_ES','TRP_BB_HB','TRP_SC_vdW','TRP_SC_ES','TYR_BB_vdW','TYR_BB_ES','TYR_BB_HB','TYR_SC_vdW','TYR_SC_ES','TYR_SC_HB','PHE_BB_vdW','PHE_BB_ES','PHE_BB_HB','PHE_SC_vdW','PHE_SC_ES','HIS_BB_vdW','HIS_BB_ES','HIS_BB_HB','HIS_SC_vdW','HIS_SC_ES','HIS_SC_HB','CYS_BB_vdW','CYS_BB_ES','CYS_BB_HB','CYS_SC_vdW','CYS_SC_ES','CYS_SC_HB','SER_BB_vdW','SER_BB_ES','SER_BB_HB','SER_SC_vdW','SER_SC_ES','SER_SC_HB','THR_BB_vdW','THR_BB_ES','THR_BB_HB','THR_SC_vdW','THR_SC_ES','THR_SC_HB','MET_BB_vdW','MET_BB_ES','MET_BB_HB','MET_SC_vdW','MET_SC_ES','MET_SC_HB','Bonds','Angles','Torsions','Angles_M','Tors_M','vdW_14','Elec_14','vdW_15','Elec_15','vdW','Elec','HBonds','HB_M','Bond_M','Wat_vdW','Wat_Elec','Wat_HB','No_Wat']
    # Type 2 - No Energy
    feature_names = ['MatchScore','GLY_BB_vdW','GLY_BB_ES','GLY_BB_HB','ALA_BB_vdW','ALA_BB_ES','ALA_BB_HB','ALA_SC_vdW','ALA_SC_ES','VAL_BB_vdW','VAL_BB_ES','VAL_BB_HB','VAL_SC_vdW','VAL_SC_ES','LEU_BB_vdW','LEU_BB_ES','LEU_BB_HB','LEU_SC_vdW','LEU_SC_ES','ILE_BB_vdW','ILE_BB_ES','ILE_BB_HB','ILE_SC_vdW','ILE_SC_ES','PRO_BB_vdW','PRO_BB_ES','PRO_BB_HB','PRO_SC_vdW','PRO_SC_ES','ASP_BB_vdW','ASP_BB_ES','ASP_BB_HB','ASP_SC_vdW','ASP_SC_ES','ASP_SC_HB','GLU_BB_vdW','GLU_BB_ES','GLU_BB_HB','GLU_SC_vdW','GLU_SC_ES','GLU_SC_HB','ASN_BB_vdW','ASN_BB_ES','ASN_BB_HB','ASN_SC_vdW','ASN_SC_ES','ASN_SC_HB','GLN_BB_vdW','GLN_BB_ES','GLN_BB_HB','GLN_SC_vdW','GLN_SC_ES','GLN_SC_HB','LYS_BB_vdW','LYS_BB_ES','LYS_BB_HB','LYS_SC_vdW','LYS_SC_ES','LYS_SC_HB','ARG_BB_vdW','ARG_BB_ES','ARG_BB_HB','ARG_SC_vdW','ARG_SC_ES','ARG_SC_HB','TRP_BB_vdW','TRP_BB_ES','TRP_BB_HB','TRP_SC_vdW','TRP_SC_ES','TYR_BB_vdW','TYR_BB_ES','TYR_BB_HB','TYR_SC_vdW','TYR_SC_ES','TYR_SC_HB','PHE_BB_vdW','PHE_BB_ES','PHE_BB_HB','PHE_SC_vdW','PHE_SC_ES','HIS_BB_vdW','HIS_BB_ES','HIS_BB_HB','HIS_SC_vdW','HIS_SC_ES','HIS_SC_HB','CYS_BB_vdW','CYS_BB_ES','CYS_BB_HB','CYS_SC_vdW','CYS_SC_ES','CYS_SC_HB','SER_BB_vdW','SER_BB_ES','SER_BB_HB','SER_SC_vdW','SER_SC_ES','SER_SC_HB','THR_BB_vdW','THR_BB_ES','THR_BB_HB','THR_SC_vdW','THR_SC_ES','THR_SC_HB','MET_BB_vdW','MET_BB_ES','MET_BB_HB','MET_SC_vdW','MET_SC_ES','MET_SC_HB','Bonds','Angles','Torsions','Angles_M','Tors_M','vdW_14','Elec_14','vdW_15','Elec_15','vdW','Elec','HBonds','HB_M','Bond_M','Wat_vdW','Wat_Elec','Wat_HB','No_Wat']
    # Type 2.1 - No Energy, No MatchScore
    #feature_names = ['GLY_BB_vdW','GLY_BB_ES','GLY_BB_HB','ALA_BB_vdW','ALA_BB_ES','ALA_BB_HB','ALA_SC_vdW','ALA_SC_ES','VAL_BB_vdW','VAL_BB_ES','VAL_BB_HB','VAL_SC_vdW','VAL_SC_ES','LEU_BB_vdW','LEU_BB_ES','LEU_BB_HB','LEU_SC_vdW','LEU_SC_ES','ILE_BB_vdW','ILE_BB_ES','ILE_BB_HB','ILE_SC_vdW','ILE_SC_ES','PRO_BB_vdW','PRO_BB_ES','PRO_BB_HB','PRO_SC_vdW','PRO_SC_ES','ASP_BB_vdW','ASP_BB_ES','ASP_BB_HB','ASP_SC_vdW','ASP_SC_ES','ASP_SC_HB','GLU_BB_vdW','GLU_BB_ES','GLU_BB_HB','GLU_SC_vdW','GLU_SC_ES','GLU_SC_HB','ASN_BB_vdW','ASN_BB_ES','ASN_BB_HB','ASN_SC_vdW','ASN_SC_ES','ASN_SC_HB','GLN_BB_vdW','GLN_BB_ES','GLN_BB_HB','GLN_SC_vdW','GLN_SC_ES','GLN_SC_HB','LYS_BB_vdW','LYS_BB_ES','LYS_BB_HB','LYS_SC_vdW','LYS_SC_ES','LYS_SC_HB','ARG_BB_vdW','ARG_BB_ES','ARG_BB_HB','ARG_SC_vdW','ARG_SC_ES','ARG_SC_HB','TRP_BB_vdW','TRP_BB_ES','TRP_BB_HB','TRP_SC_vdW','TRP_SC_ES','TYR_BB_vdW','TYR_BB_ES','TYR_BB_HB','TYR_SC_vdW','TYR_SC_ES','TYR_SC_HB','PHE_BB_vdW','PHE_BB_ES','PHE_BB_HB','PHE_SC_vdW','PHE_SC_ES','HIS_BB_vdW','HIS_BB_ES','HIS_BB_HB','HIS_SC_vdW','HIS_SC_ES','HIS_SC_HB','CYS_BB_vdW','CYS_BB_ES','CYS_BB_HB','CYS_SC_vdW','CYS_SC_ES','CYS_SC_HB','SER_BB_vdW','SER_BB_ES','SER_BB_HB','SER_SC_vdW','SER_SC_ES','SER_SC_HB','THR_BB_vdW','THR_BB_ES','THR_BB_HB','THR_SC_vdW','THR_SC_ES','THR_SC_HB','MET_BB_vdW','MET_BB_ES','MET_BB_HB','MET_SC_vdW','MET_SC_ES','MET_SC_HB','Bonds','Angles','Torsions','Angles_M','Tors_M','vdW_14','Elec_14','vdW_15','Elec_15','vdW','Elec','HBonds','HB_M','Bond_M','Wat_vdW','Wat_Elec','Wat_HB','No_Wat']
    # Type 3
    #feature_names = ['GLY_BB_vdW','GLY_BB_ES','GLY_BB_HB','ALA_BB_vdW','ALA_BB_ES','ALA_BB_HB','ALA_SC_vdW','ALA_SC_ES','VAL_BB_vdW','VAL_BB_ES','VAL_BB_HB','VAL_SC_vdW','VAL_SC_ES','LEU_BB_vdW','LEU_BB_ES','LEU_BB_HB','LEU_SC_vdW','LEU_SC_ES','ILE_BB_vdW','ILE_BB_ES','ILE_BB_HB','ILE_SC_vdW','ILE_SC_ES','PRO_BB_vdW','PRO_BB_ES','PRO_BB_HB','PRO_SC_vdW','PRO_SC_ES','ASP_BB_vdW','ASP_BB_ES','ASP_BB_HB','ASP_SC_vdW','ASP_SC_ES','ASP_SC_HB','GLU_BB_vdW','GLU_BB_ES','GLU_BB_HB','GLU_SC_vdW','GLU_SC_ES','GLU_SC_HB','ASN_BB_vdW','ASN_BB_ES','ASN_BB_HB','ASN_SC_vdW','ASN_SC_ES','ASN_SC_HB','GLN_BB_vdW','GLN_BB_ES','GLN_BB_HB','GLN_SC_vdW','GLN_SC_ES','GLN_SC_HB','LYS_BB_vdW','LYS_BB_ES','LYS_BB_HB','LYS_SC_vdW','LYS_SC_ES','LYS_SC_HB','ARG_BB_vdW','ARG_BB_ES','ARG_BB_HB','ARG_SC_vdW','ARG_SC_ES','ARG_SC_HB','TRP_BB_vdW','TRP_BB_ES','TRP_BB_HB','TRP_SC_vdW','TRP_SC_ES','TYR_BB_vdW','TYR_BB_ES','TYR_BB_HB','TYR_SC_vdW','TYR_SC_ES','TYR_SC_HB','PHE_BB_vdW','PHE_BB_ES','PHE_BB_HB','PHE_SC_vdW','PHE_SC_ES','HIS_BB_vdW','HIS_BB_ES','HIS_BB_HB','HIS_SC_vdW','HIS_SC_ES','HIS_SC_HB','CYS_BB_vdW','CYS_BB_ES','CYS_BB_HB','CYS_SC_vdW','CYS_SC_ES','CYS_SC_HB','SER_BB_vdW','SER_BB_ES','SER_BB_HB','SER_SC_vdW','SER_SC_ES','SER_SC_HB','THR_BB_vdW','THR_BB_ES','THR_BB_HB','THR_SC_vdW','THR_SC_ES','THR_SC_HB','MET_BB_vdW','MET_BB_ES','MET_BB_HB','MET_SC_vdW','MET_SC_ES','MET_SC_HB']
    # Type 4
    #feature_names = ['vdW_15', 'ARG_SC_HB', 'vdW_14', 'Energy', 'Wat_HB', 'Elec', 'No_Wat', 'LEU_SC_vdW', 'HBonds', 'Angles', 'ASN_BB_vdW', 'ALA_SC_vdW', 'MatchScore', 'ASP_SC_HB', 'GLU_BB_vdW']

    accuracies = pd.DataFrame(columns=['RMSD threshold', 'Lowest energy', 'Lowest RMSD', 'Random', 'Predicted'])
    accuracies['RMSD threshold'] = np.arange(0, 5.25, 0.25)
    accuracies['Lowest energy'] = 0
    accuracies['Lowest RMSD'] = 0
    accuracies['Random'] = 0
    accuracies['Predicted'] = 0

    total = 0
    seed = 42

    #try:
    for pdb_id in graph_set:
        if os.path.isfile(dir + pdb_id + extension):

    #for file in os.listdir(cwd + dir):
        #if os.path.isfile(cwd + dir + file):
            pdb = pd.read_csv(dir + pdb_id + extension, index_col=False)

            if len(pdb.index) > 0 :
                pdb = pdb.dropna()  # remove rows with NaN

                total = total + 1

                ### redo
                PDBdata = pd.DataFrame(columns=['Energies','Label','RMSD','Label_ML','RMSD_ML'])
                PDBdata['Energies'] = pdb['Energy']
                PDBdata['Label'] = pdb['Label']
                PDBdata['RMSD'] = pdb['RMSD']

                if args.normalize == 'yes':
                    pdb_drop = drop_features(pdb)  # drop some features because different models were trained using different number of features
                    pdb_normalized = preprocessing.normalize(pdb_drop)  # normalize the dataset since it was trained on normalized data
                    PDBdata['Label_ML'] = model.predict(pdb_normalized)  # predict the output

                else:
                    pdb_drop = drop_features(pdb)
                    PDBdata['Label_ML'] = model.predict(pdb[feature_names])

                if (args.model == "LR") or (args.model == "XGBr"):
                    for index, value in PDBdata['Label_ML'].items():
                        PDBdata.at[index, 'RMSD_ML'] = act_to_RMSD(value)
                elif (args.model == "XGBc") or  (args.model == "LRc"):
                    PDBdata.loc[PDBdata['Label_ML'] == 1, 'RMSD_ML'] = 1
                    PDBdata.loc[PDBdata['Label_ML'] == 0, 'RMSD_ML'] = 3

                random = PDBdata.sample()
                for index, value in accuracies['RMSD threshold'].items():

                    min_energy = PDBdata['Energies'].idxmin()

                    # Lowest energy
                    if PDBdata.loc[min_energy, 'RMSD'] <= value:
                        accuracies.at[index, 'Lowest energy'] += 1

                    # Lowest RMSD
                    if PDBdata['RMSD'].min() <= value:
                        accuracies.at[index, 'Lowest RMSD'] += 1

                    # Random
                    if random['RMSD'].iloc[0] <= value:
                        #accuracies['Random'] = accuracies['Random'] + 1
                        accuracies.at[index, 'Random'] += 1

                    # Predicted
                    #if PDBdata.loc[min_energy, 'RMSD_ML'] <= value:
                    #    accuracies.at[index, 'Predicted'] += 1

                    # Predicted lowest RMSD
                    #if PDBdata['RMSD_ML'].min() <= value:
                    #    accuracies.at[index, 'Predicted'] += 1

                    # Predicted
                    #lowestRMSD=PDBdata['RMSD'].idxmin()
                    #if PDBdata.loc[lowestRMSD, 'RMSD_ML'] <= value:
                    #   accuracies.at[index, 'Predicted'] += 1

                    # Predicted
                    if PDBdata.loc[PDBdata['RMSD_ML'].idxmin(), 'RMSD'] <= value:
                       accuracies.at[index, 'Predicted'] += 1

    #except:
    #    print("your shit didn't work lol")

    #print('Total: ' + str(total))

    #print('Accuracies raw:')
    #print(accuracies)

    accuracies['Lowest energy'] = (accuracies['Lowest energy'] / total) * 100
    accuracies['Lowest RMSD'] = (accuracies['Lowest RMSD'] / total) * 100
    accuracies['Random'] = (accuracies['Random'] / total) * 100
    accuracies['Predicted'] = (accuracies['Predicted'] / total) * 100

    #print('Accuracies %:')
    #print(accuracies)

    plt.plot(accuracies['RMSD threshold'], accuracies['Lowest energy'], label="Lowest energy", color='blue')
    plt.plot(accuracies['RMSD threshold'], accuracies['Lowest RMSD'], label="Lowest RMSD", color='red')
    plt.plot(accuracies['RMSD threshold'], accuracies['Random'],label="Random", color='orange')
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
    plt.legend(prop={'size': 12})
    # show gridlines
    plt.grid()
    # Set a title of the current axes.
    plt.title(args.model + ' ' + args.set, fontdict=font1)
    # Add extra white space at the bottom so that the label is not cut off
    plt.subplots_adjust(bottom=0.13)

    plt.savefig(cwd + '/graphs/' + args.model + '_' + args.set + '.png')


def plot_accuracy_classifier(args):

# ----------- Load the set to test the model on

    train_list = []
    test_list = []

    graph_set = [] # What data is used to plot the graph (either test, train or both)

    # Use the splitting from https://arxiv.org/html/2308.09639v2#S5
    if os.path.isfile(args.CSV_data_dir + args.list_trainset) is True and os.path.isfile(args.CSV_data_dir + args.list_testset) is True:

        print('| Training set selected from %-86s |' % args.list_trainset)
        with open(args.CSV_data_dir + args.list_trainset, 'rb') as list_f:
            for line in list_f:
                train_list.append(line.strip().decode('UTF-8'))
        print('| Testing set selected from %-86s  |' % args.list_testset)
        with open(args.CSV_data_dir + args.list_testset, 'rb') as list_f:
            for line in list_f:
                test_list.append(line.strip().decode('UTF-8'))

        if args.teston == "test":
            graph_set = test_list
        elif args.teston == "train":
            graph_set = train_list
        elif args.list_testset == "full":
            graph_set = test_list + train_list

    else:
        print('| Please provide a train and test list. |')

# ----------- Load the model

    if args.model == 'XGBc':
        model = xgb.XGBClassifier()
        model.load_model(cwd + '/models/' + args.model_name)
        graph_title = "XGBc"

# ----------- Load the data to test the model on

    if args.set == 'Astex':
        dir = '/AstexSet_top10/'
        graph_title = "Astex set" + graph_title
    elif args.set == 'Fitted':
        dir = '/FittedSet_top10/'
        graph_title = "Fitted set" + graph_title
    elif args.set == 'PDBBind':
        dir = '/PDBBindSet_top10/'
        graph_title = "PDB set" + graph_title
    elif args.set == 'PDB_10':
        dir = '/PDB_10/'
        graph_title = "PDB set mismatch" + graph_title
    elif args.set == 'PDB_test':
        dir = '/PDB_10/'
        graph_title = "PDB set mismatch" + graph_title
    elif args.set == 'CSV':
        dir = '/CSV/'
        graph_title = "PDB full set" + graph_title
    elif args.set == 'CSV_vdW15x':
        dir = '/CSV_vdW15x/'
        graph_title = "PDB full set" + graph_title
    elif args.set == 'TestSet':
        dir = '/TestSet/'
        graph_title = "Test set" + graph_title


    # Features needed for predictions
    # Type 1
    #feature_names = ['Energy','MatchScore','GLY_BB_vdW','GLY_BB_ES','GLY_BB_HB','ALA_BB_vdW','ALA_BB_ES','ALA_BB_HB','ALA_SC_vdW','ALA_SC_ES','VAL_BB_vdW','VAL_BB_ES','VAL_BB_HB','VAL_SC_vdW','VAL_SC_ES','LEU_BB_vdW','LEU_BB_ES','LEU_BB_HB','LEU_SC_vdW','LEU_SC_ES','ILE_BB_vdW','ILE_BB_ES','ILE_BB_HB','ILE_SC_vdW','ILE_SC_ES','PRO_BB_vdW','PRO_BB_ES','PRO_BB_HB','PRO_SC_vdW','PRO_SC_ES','ASP_BB_vdW','ASP_BB_ES','ASP_BB_HB','ASP_SC_vdW','ASP_SC_ES','ASP_SC_HB','GLU_BB_vdW','GLU_BB_ES','GLU_BB_HB','GLU_SC_vdW','GLU_SC_ES','GLU_SC_HB','ASN_BB_vdW','ASN_BB_ES','ASN_BB_HB','ASN_SC_vdW','ASN_SC_ES','ASN_SC_HB','GLN_BB_vdW','GLN_BB_ES','GLN_BB_HB','GLN_SC_vdW','GLN_SC_ES','GLN_SC_HB','LYS_BB_vdW','LYS_BB_ES','LYS_BB_HB','LYS_SC_vdW','LYS_SC_ES','LYS_SC_HB','ARG_BB_vdW','ARG_BB_ES','ARG_BB_HB','ARG_SC_vdW','ARG_SC_ES','ARG_SC_HB','TRP_BB_vdW','TRP_BB_ES','TRP_BB_HB','TRP_SC_vdW','TRP_SC_ES','TYR_BB_vdW','TYR_BB_ES','TYR_BB_HB','TYR_SC_vdW','TYR_SC_ES','TYR_SC_HB','PHE_BB_vdW','PHE_BB_ES','PHE_BB_HB','PHE_SC_vdW','PHE_SC_ES','HIS_BB_vdW','HIS_BB_ES','HIS_BB_HB','HIS_SC_vdW','HIS_SC_ES','HIS_SC_HB','CYS_BB_vdW','CYS_BB_ES','CYS_BB_HB','CYS_SC_vdW','CYS_SC_ES','CYS_SC_HB','SER_BB_vdW','SER_BB_ES','SER_BB_HB','SER_SC_vdW','SER_SC_ES','SER_SC_HB','THR_BB_vdW','THR_BB_ES','THR_BB_HB','THR_SC_vdW','THR_SC_ES','THR_SC_HB','MET_BB_vdW','MET_BB_ES','MET_BB_HB','MET_SC_vdW','MET_SC_ES','MET_SC_HB','Bonds','Angles','Torsions','Angles_M','Tors_M','vdW_14','Elec_14','vdW_15','Elec_15','vdW','Elec','HBonds','HB_M','Bond_M','Wat_vdW','Wat_Elec','Wat_HB','No_Wat']
    # Type 2 - No Energy
    #feature_names = ['MatchScore','GLY_BB_vdW','GLY_BB_ES','GLY_BB_HB','ALA_BB_vdW','ALA_BB_ES','ALA_BB_HB','ALA_SC_vdW','ALA_SC_ES','VAL_BB_vdW','VAL_BB_ES','VAL_BB_HB','VAL_SC_vdW','VAL_SC_ES','LEU_BB_vdW','LEU_BB_ES','LEU_BB_HB','LEU_SC_vdW','LEU_SC_ES','ILE_BB_vdW','ILE_BB_ES','ILE_BB_HB','ILE_SC_vdW','ILE_SC_ES','PRO_BB_vdW','PRO_BB_ES','PRO_BB_HB','PRO_SC_vdW','PRO_SC_ES','ASP_BB_vdW','ASP_BB_ES','ASP_BB_HB','ASP_SC_vdW','ASP_SC_ES','ASP_SC_HB','GLU_BB_vdW','GLU_BB_ES','GLU_BB_HB','GLU_SC_vdW','GLU_SC_ES','GLU_SC_HB','ASN_BB_vdW','ASN_BB_ES','ASN_BB_HB','ASN_SC_vdW','ASN_SC_ES','ASN_SC_HB','GLN_BB_vdW','GLN_BB_ES','GLN_BB_HB','GLN_SC_vdW','GLN_SC_ES','GLN_SC_HB','LYS_BB_vdW','LYS_BB_ES','LYS_BB_HB','LYS_SC_vdW','LYS_SC_ES','LYS_SC_HB','ARG_BB_vdW','ARG_BB_ES','ARG_BB_HB','ARG_SC_vdW','ARG_SC_ES','ARG_SC_HB','TRP_BB_vdW','TRP_BB_ES','TRP_BB_HB','TRP_SC_vdW','TRP_SC_ES','TYR_BB_vdW','TYR_BB_ES','TYR_BB_HB','TYR_SC_vdW','TYR_SC_ES','TYR_SC_HB','PHE_BB_vdW','PHE_BB_ES','PHE_BB_HB','PHE_SC_vdW','PHE_SC_ES','HIS_BB_vdW','HIS_BB_ES','HIS_BB_HB','HIS_SC_vdW','HIS_SC_ES','HIS_SC_HB','CYS_BB_vdW','CYS_BB_ES','CYS_BB_HB','CYS_SC_vdW','CYS_SC_ES','CYS_SC_HB','SER_BB_vdW','SER_BB_ES','SER_BB_HB','SER_SC_vdW','SER_SC_ES','SER_SC_HB','THR_BB_vdW','THR_BB_ES','THR_BB_HB','THR_SC_vdW','THR_SC_ES','THR_SC_HB','MET_BB_vdW','MET_BB_ES','MET_BB_HB','MET_SC_vdW','MET_SC_ES','MET_SC_HB','Bonds','Angles','Torsions','Angles_M','Tors_M','vdW_14','Elec_14','vdW_15','Elec_15','vdW','Elec','HBonds','HB_M','Bond_M','Wat_vdW','Wat_Elec','Wat_HB','No_Wat']
    # Type 2.1 - No Energy, No MatchScore
    #feature_names = ['GLY_BB_vdW','GLY_BB_ES','GLY_BB_HB','ALA_BB_vdW','ALA_BB_ES','ALA_BB_HB','ALA_SC_vdW','ALA_SC_ES','VAL_BB_vdW','VAL_BB_ES','VAL_BB_HB','VAL_SC_vdW','VAL_SC_ES','LEU_BB_vdW','LEU_BB_ES','LEU_BB_HB','LEU_SC_vdW','LEU_SC_ES','ILE_BB_vdW','ILE_BB_ES','ILE_BB_HB','ILE_SC_vdW','ILE_SC_ES','PRO_BB_vdW','PRO_BB_ES','PRO_BB_HB','PRO_SC_vdW','PRO_SC_ES','ASP_BB_vdW','ASP_BB_ES','ASP_BB_HB','ASP_SC_vdW','ASP_SC_ES','ASP_SC_HB','GLU_BB_vdW','GLU_BB_ES','GLU_BB_HB','GLU_SC_vdW','GLU_SC_ES','GLU_SC_HB','ASN_BB_vdW','ASN_BB_ES','ASN_BB_HB','ASN_SC_vdW','ASN_SC_ES','ASN_SC_HB','GLN_BB_vdW','GLN_BB_ES','GLN_BB_HB','GLN_SC_vdW','GLN_SC_ES','GLN_SC_HB','LYS_BB_vdW','LYS_BB_ES','LYS_BB_HB','LYS_SC_vdW','LYS_SC_ES','LYS_SC_HB','ARG_BB_vdW','ARG_BB_ES','ARG_BB_HB','ARG_SC_vdW','ARG_SC_ES','ARG_SC_HB','TRP_BB_vdW','TRP_BB_ES','TRP_BB_HB','TRP_SC_vdW','TRP_SC_ES','TYR_BB_vdW','TYR_BB_ES','TYR_BB_HB','TYR_SC_vdW','TYR_SC_ES','TYR_SC_HB','PHE_BB_vdW','PHE_BB_ES','PHE_BB_HB','PHE_SC_vdW','PHE_SC_ES','HIS_BB_vdW','HIS_BB_ES','HIS_BB_HB','HIS_SC_vdW','HIS_SC_ES','HIS_SC_HB','CYS_BB_vdW','CYS_BB_ES','CYS_BB_HB','CYS_SC_vdW','CYS_SC_ES','CYS_SC_HB','SER_BB_vdW','SER_BB_ES','SER_BB_HB','SER_SC_vdW','SER_SC_ES','SER_SC_HB','THR_BB_vdW','THR_BB_ES','THR_BB_HB','THR_SC_vdW','THR_SC_ES','THR_SC_HB','MET_BB_vdW','MET_BB_ES','MET_BB_HB','MET_SC_vdW','MET_SC_ES','MET_SC_HB','Bonds','Angles','Torsions','Angles_M','Tors_M','vdW_14','Elec_14','vdW_15','Elec_15','vdW','Elec','HBonds','HB_M','Bond_M','Wat_vdW','Wat_Elec','Wat_HB','No_Wat']
    # Type 3
    #feature_names = ['GLY_BB_vdW','GLY_BB_ES','GLY_BB_HB','ALA_BB_vdW','ALA_BB_ES','ALA_BB_HB','ALA_SC_vdW','ALA_SC_ES','VAL_BB_vdW','VAL_BB_ES','VAL_BB_HB','VAL_SC_vdW','VAL_SC_ES','LEU_BB_vdW','LEU_BB_ES','LEU_BB_HB','LEU_SC_vdW','LEU_SC_ES','ILE_BB_vdW','ILE_BB_ES','ILE_BB_HB','ILE_SC_vdW','ILE_SC_ES','PRO_BB_vdW','PRO_BB_ES','PRO_BB_HB','PRO_SC_vdW','PRO_SC_ES','ASP_BB_vdW','ASP_BB_ES','ASP_BB_HB','ASP_SC_vdW','ASP_SC_ES','ASP_SC_HB','GLU_BB_vdW','GLU_BB_ES','GLU_BB_HB','GLU_SC_vdW','GLU_SC_ES','GLU_SC_HB','ASN_BB_vdW','ASN_BB_ES','ASN_BB_HB','ASN_SC_vdW','ASN_SC_ES','ASN_SC_HB','GLN_BB_vdW','GLN_BB_ES','GLN_BB_HB','GLN_SC_vdW','GLN_SC_ES','GLN_SC_HB','LYS_BB_vdW','LYS_BB_ES','LYS_BB_HB','LYS_SC_vdW','LYS_SC_ES','LYS_SC_HB','ARG_BB_vdW','ARG_BB_ES','ARG_BB_HB','ARG_SC_vdW','ARG_SC_ES','ARG_SC_HB','TRP_BB_vdW','TRP_BB_ES','TRP_BB_HB','TRP_SC_vdW','TRP_SC_ES','TYR_BB_vdW','TYR_BB_ES','TYR_BB_HB','TYR_SC_vdW','TYR_SC_ES','TYR_SC_HB','PHE_BB_vdW','PHE_BB_ES','PHE_BB_HB','PHE_SC_vdW','PHE_SC_ES','HIS_BB_vdW','HIS_BB_ES','HIS_BB_HB','HIS_SC_vdW','HIS_SC_ES','HIS_SC_HB','CYS_BB_vdW','CYS_BB_ES','CYS_BB_HB','CYS_SC_vdW','CYS_SC_ES','CYS_SC_HB','SER_BB_vdW','SER_BB_ES','SER_BB_HB','SER_SC_vdW','SER_SC_ES','SER_SC_HB','THR_BB_vdW','THR_BB_ES','THR_BB_HB','THR_SC_vdW','THR_SC_ES','THR_SC_HB','MET_BB_vdW','MET_BB_ES','MET_BB_HB','MET_SC_vdW','MET_SC_ES','MET_SC_HB']
    # Type 4
    #feature_names = ['vdW_15', 'ARG_SC_HB', 'vdW_14', 'Energy', 'Wat_HB', 'Elec', 'No_Wat', 'LEU_SC_vdW', 'HBonds', 'Angles', 'ASN_BB_vdW', 'ALA_SC_vdW', 'MatchScore', 'ASP_SC_HB', 'GLU_BB_vdW']

    accuracies = pd.DataFrame(columns=['RMSD threshold', 'Lowest energy', 'Lowest RMSD', 'Random', 'Predicted'])
    accuracies['RMSD threshold'] = np.arange(0, 5.25, 0.25)
    accuracies['Lowest energy'] = 0
    accuracies['Lowest RMSD'] = 0
    accuracies['Random'] = 0
    accuracies['Predicted'] = 0

    total = 0
    seed = 42

    for file in os.listdir(cwd + dir):
        if os.path.isfile(cwd + dir + file):
            #print(file)
            pdb = pd.read_csv(cwd + dir + file, index_col=False)
            total = total + 1

            ### redo
            PDBdata = pd.DataFrame(columns=['Energies','Label','RMSD','Label_ML','RMSD_ML'])
            PDBdata['Energies'] = pdb['Energy']
            PDBdata['Label'] = pdb['Label']
            PDBdata['RMSD'] = pdb['RMSD']

            if args.normalize == 'yes':
                pdb_drop = drop_features(pdb)  # drop some features because different models were trained using different number of features
                pdb_normalized = preprocessing.normalize(pdb_drop)  # normalize the dataset since it was trained on normalized data
                PDBdata['Label_ML'] = model.predict(pdb_normalized)  # predict the output

            else:
                pdb_drop = drop_features(pdb)
                PDBdata['Label_ML'] = model.predict(pdb[feature_names])

            PDBdata.loc[PDBdata['Label_ML'] == 1, 'RMSD_ML'] = 1.75
            PDBdata.loc[PDBdata['Label_ML'] == 0, 'RMSD_ML'] = 2.5

            random = PDBdata.sample()
            for index, value in accuracies['RMSD threshold'].items():

                min_energy = PDBdata['Energies'].idxmin()

                # Lowest energy
                if PDBdata.loc[min_energy, 'RMSD'] <= value:
                    accuracies.at[index, 'Lowest energy'] += 1

                # Lowest RMSD
                if PDBdata['RMSD'].min() <= value:
                    accuracies.at[index, 'Lowest RMSD'] += 1

                # Random
                if random['RMSD'].iloc[0] <= value:
                    #accuracies['Random'] = accuracies['Random'] + 1
                    accuracies.at[index, 'Random'] += 1

                # Predicted
                #if PDBdata.loc[min_energy, 'RMSD_ML'] <= value:
                #    accuracies.at[index, 'Predicted'] += 1

                if PDBdata.loc[PDBdata['RMSD_ML'].idxmin(), 'RMSD'] <= value:
                   accuracies.at[index, 'Predicted'] += 1


    print('Total: ' + str(total))

    print('Accuracies raw:')
    print(accuracies)

    accuracies['Lowest energy'] = (accuracies['Lowest energy'] / total) * 100
    accuracies['Lowest RMSD'] = (accuracies['Lowest RMSD'] / total) * 100
    accuracies['Random'] = (accuracies['Random'] / total) * 100
    accuracies['Predicted'] = (accuracies['Predicted'] / total) * 100

    print('Accuracies %:')
    print(accuracies)

    plt.plot(accuracies['RMSD threshold'], accuracies['Lowest energy'], label="Lowest energy", color='blue')
    plt.plot(accuracies['RMSD threshold'], accuracies['Lowest RMSD'], label="Lowest RMSD", color='red')
    plt.plot(accuracies['RMSD threshold'], accuracies['Random'],label="Random", color='orange')
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
    plt.legend(prop={'size': 12})
    # show gridlines
    plt.grid()
    # Set a title of the current axes.
    plt.title('XGBc Astex', fontdict=font1)
    # Add extra white space at the bottom so that the label is not cut off
    plt.subplots_adjust(bottom=0.13)

    plt.savefig(cwd + '/graphs/' + args.model + '_' + args.set +'.png')

def XGB_feature_importance(args, model):

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(45, 25), dpi=300)

    df = pd.DataFrame()
    df['Features'] = np.array(model.get_booster().feature_names).tolist()
    df['FeatureImportances'] = model.feature_importances_.tolist()

    # Cover
    xgb.plot_importance(model._Booster, ax=ax1, title="Cover", ylabel="Feature", importance_type='cover', max_num_features=130, grid=False, show_values=False)
    df['Cover'] = df['Features'].map(model.get_booster().get_score(importance_type='cover'))

    # Total cover
    xgb.plot_importance(model._Booster, ax=ax2, title="Total cover", ylabel="Feature", importance_type='total_cover', max_num_features=130, grid=False, show_values=False)
    df['TotalCover'] = df['Features'].map(model.get_booster().get_score(importance_type='total_cover'))

    # Gain
    xgb.plot_importance(model._Booster, ax=ax3, title="Gain", ylabel="Feature", importance_type='gain', max_num_features=130, grid=False, show_values=False)
    df['Gain'] = df['Features'].map(model.get_booster().get_score(importance_type='gain'))

    # Total gain
    xgb.plot_importance(model._Booster, ax=ax4, title="Total gain", ylabel="Feature", importance_type='total_gain', max_num_features=130, grid=False, show_values=False)
    df['TotalGain'] = df['Features'].map(model.get_booster().get_score(importance_type='total_gain'))

    # Weight
    xgb.plot_importance(model._Booster, ax=ax5, title="Weight", ylabel="Feature", importance_type='weight', max_num_features=130, grid=False, show_values=False)
    df['Weight'] = df['Features'].map(model.get_booster().get_score(importance_type='weight'))

    #fig_func, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(45, 25), dpi=300)
    #fig_func.suptitle('Feature importances using XGBr built-in functions', size=50)

    xgb.plot_importance(model._Booster, ax=ax6, title="Feature importance", ylabel="Feature", importance_type='weight', max_num_features=130, grid=False, show_values=False)
    #sorted_idx = model.feature_importances_.argsort()
    #ax6.barh(np.array(model.get_booster().feature_names)[sorted_idx], model.feature_importances_[sorted_idx])

    #ax1.barh(np.array(model.get_booster().feature_names)[sorted_idx], model.feature_importances_[sorted_idx])

    #df.to_csv('XGB_feature_importance.csv', index=False)
    plt.savefig(args.name + '_XGB_feature_importance.png')
    return df['FeatureImportances'], df['Cover'], df['TotalCover'], df['Gain'], df['TotalGain'], df['Weight']

def SHAP_values(args, model, train_set_X):

    # Get SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_set_X, check_additivity=False)

    df = pd.DataFrame(shap_values)
    df_SHAP = pd.DataFrame()
    df_SHAP['Features'] = np.array(model.get_booster().feature_names).tolist()
    # df_SHAP['SHAP'] = shap_values
    df_SHAP['SHAP_mean'] = df.mean()

    # Plot SHAP values
    plt.clf()
    shap.summary_plot(shap_values, train_set_X, max_display=len(train_set_X.columns), show=False)
    plt.savefig(args.name + '_SHAP_values.png')

    # Plot SHAP values with label 0
    # plt.clf()
    # shap.summary_plot(shap_values[0], train_set_X, max_display=len(train_set_X.columns), show=False)
    # plt.savefig(args.name + '_SHAP_values_0.png')
    # plt.clf()
    # shap.summary_plot(shap_values[0], train_set_X, max_display=20, show=False)
    # plt.savefig(args.name + '_SHAP_values_0_top20.png')

    # Plot SHAP values with label 1
    # plt.clf()
    # shap.summary_plot(shap_values[1], train_set_X, max_display=len(train_set_X.columns), show=False)
    # plt.savefig(args.name + '_SHAP_values_1.png')
    # plt.clf()
    # shap.summary_plot(shap_values[1], train_set_X, max_display=20, show=False)
    # plt.savefig(args.name + '_SHAP_values_1_top20.png')

    # Plot SHAP values
    plt.clf()
    shap.summary_plot(shap_values, train_set_X, max_display=10, show=False)
    plt.savefig(args.name + '_SHAP_values_top10.png')

    # Plot SHAP values
    plt.clf()
    shap.summary_plot(shap_values, train_set_X, max_display=15, show=False)
    plt.savefig(args.name + '_SHAP_values_top15.png')

    # Plot SHAP values
    plt.clf()
    shap.summary_plot(shap_values, train_set_X, max_display=20, show=False)
    plt.savefig(args.name + '_SHAP_values_top20.png')

    # Plot SHAP values
    plt.clf()
    shap.summary_plot(shap_values, train_set_X, max_display=20, show=False)
    plt.gca().invert_yaxis()
    plt.savefig(args.name + '_SHAP_values_top20_horizontal.png')

    plt.clf()
    shap.summary_plot(shap_values, train_set_X, plot_type="bar", max_display=len(train_set_X.columns), show=False)
    plt.savefig(args.name + '_SHAP_values_mean.png')

    #df_SHAP.to_csv('SHAP_values_mean.csv', index=False)
    return df_SHAP['SHAP_mean']

def permutation_imp(args, model, test_set_X, test_set_y):

    # Get the permutation importance values
    perm_importance = permutation_importance(model, test_set_X, test_set_y)
    sorted_idx = perm_importance.importances_mean.argsort()

    df = pd.DataFrame()
    df['Features'] = np.array(model.get_booster().feature_names).tolist()
    df['Permutations'] = perm_importance.importances_mean.tolist()

    # Generate a bar plot of the sorted permutation coefficients
    plt.clf()
    plt.figure()
    plt.figure(figsize=(5, 28), dpi=300)
    plt.barh(np.array(model.get_booster().feature_names)[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.savefig(args.name + '_Permutation_importance.png', bbox_inches='tight')

    # Generate a heat map
    def correlation_heatmap(train, filename):
        plt.clf()
        correlations = train.corr()

        fig_heat, ax = plt.subplots(figsize=(75, 75))
        sns.heatmap(correlations, center=0, fmt='.2f', cmap="bwr_r",
                    square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
        plt.savefig(args.name + filename + '.png');

    correlation_heatmap(test_set_X[np.array(model.get_booster().feature_names)[sorted_idx]], '_Permutation_importance_HeatMap_sorted')
    correlation_heatmap(test_set_X[np.array(model.get_booster().feature_names)], '_Permutation_importance_HeatMap')

    # df.to_csv('Permutation_importance.csv', index=False)
    return df['Permutations']

def plot_eval_metrics(args, model):
    results = model.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)

    # plot AUC
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['auc'], label='Train')
    ax.plot(x_axis, results['validation_1']['auc'], label='Val')
    ax.plot(x_axis, results['validation_2']['auc'], label='Test')
    ax.legend()
    plt.xlabel('epoch (n-estimator)')
    plt.ylabel('AUC')
    plt.title('XGBc AUC')
    plt.savefig(args.name + '_XGBc_AUC.png')

    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Val')
    ax.plot(x_axis, results['validation_2']['logloss'], label='Test')
    ax.legend()
    plt.xlabel('epoch (n-estimator)')
    plt.ylabel('Log Loss')
    plt.title('XGBc Log Loss')
    plt.savefig(args.name + '_XGBc_logloss.png')

    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Val')
    ax.plot(x_axis, results['validation_2']['error'], label='Test')
    ax.legend()
    plt.xlabel('epoch (n-estimator)')
    plt.ylabel('Classification Error')
    plt.title('XGBc Classification Error')
    plt.savefig(args.name + '_XGBc_error.png')

def plot_violin_plot(args):

    list = [['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE'],
            ['LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    ]
    interactions = ['BB_vdW', 'BB_ES', 'BB_HB', 'SC_vdW', 'SC_ES', 'SC_HB']
    internal = ['Energy', 'MatchScore', 'Bonds', 'Angles', 'Torsions', 'Angles_M', 'Tors_M', 'vdW_14', 'vdW_15', 'vdW', 'Elec', 'HBonds', 'HB_M', 'Bond_M', 'Wat_vdW', 'Wat_Elec', 'Wat_HB', 'No_Wat']

    train_set_X, train_set_y, val_set_X, val_set_y, test_set_X, test_set_y, colname = split_set(args)
    if args.normalize == 'yes':
        train_set_y = train_set_y.reset_index()
    train_set_temp = pd.concat([train_set_X, train_set_y], axis=1)

    #train_set = train_set_temp[(train_set_temp['Energy'] >= -25) & (train_set_temp['Energy'] <= 25)]
    train_set = train_set_temp[(train_set_temp >= -5).all(axis=1) & (train_set_temp <= 5).all(axis=1)] # Range of substraction values to plot
    # train_set_temp['Energy'] = train_set_temp['Energy'].clip(lower=-100, upper=100)
    plt.figure(figsize=(10, 6))
    plt.hist(train_set['Energy'], bins=100)
    plt.savefig('hist.png')
    plt.clf()

    # print("Min value:", train_set_temp['Energy'].min())
    # print("Max value:", train_set_temp['Energy'].max())
    # print("Mean value:", train_set_temp['Energy'].mean())
    # print(len(train_set_temp.index))
    #
    # train_set = train_set_temp[np.abs(stats.zscore(train_set_temp['Energy'])) < 0.1]
    # #train_set = train_set_temp[(np.abs(stats.zscore(train_set_temp[colname])) < 3).all(axis=1)]
    # print("Min value:", train_set['Energy'].min())
    # print("Max value:", train_set['Energy'].max())
    # print("Mean value:", train_set['Energy'].mean())
    # print(len(train_set.index))


    for i, residues1 in enumerate(list):
        plt.clf()
        # Set plot style
        #sns.set_style('white', rc={'xtick.bottom': True})
        sns.set(style="whitegrid", rc={"grid.color": 'silver', "grid.linewidth": 1, "axes.edgecolor": 'silver'})
        plt.rcParams['xtick.bottom'] = True # add tick marks
        #plt.rcParams['xtick.major.size'] = 20
        #plt.rcParams['xtick.major.width'] = 4
        plt.rcParams['xtick.color'] = 'silver'
        plt.rcParams['axes.labelcolor'] = 'black'

        # Create a figure with a grid of subplots
        fig, axes = plt.subplots(5, 2, figsize=(8.5, 11))
        #fig, axes = plt.subplots(1, 1, figsize=(8.5, 2)) # for plotting internals

        # Flatten axes for easy indexing
        axes_f = axes.flatten()

        for idx, residue in enumerate(residues1):
            features = []
            for interaction in interactions:
                feature = residue + '_' + interaction
                features.append(feature)

            for col in features:
                if col not in train_set.columns:
                    train_set[col] = float('nan')  # Add missing columns with NaN values

            data = pd.DataFrame()
            for f in features:
                d = pd.DataFrame({
                    'feat': f,  # Set the entire 'feat' column to the feature name
                    'value': train_set[f],  # Use the current feature column from train_set
                    'Label': train_set['Label']
                })
                data = pd.concat([data, d], axis=0)
            print(data)
            data = data.reset_index()

            # For regression data
            #sns.violinplot(
            #    data=train_set[features], ax=axes_f[idx], inner="quartile") #, x='Label', y=, split=True, hue='Label', palette={1: "palegreen", 0: "pink"}, ax=axes[idx], inner="quartile")
            # axes_f[idx].set_title(residue)
            # if (idx == 8) or (idx == 9):
            #     label = ['BB_vdW', 'BB_ES', 'BB_HB', 'SC_vdW', 'SC_ES', 'SC_HB', 'Label']
            #     axes_f[idx].set_xticklabels(label, rotation=75, ha='right')
            # else:
            #     axes_f[idx].set_xticks([])  # Remove x-ticks

            sns.violinplot(
                data=data, ax=axes_f[idx], inner="quartile",
                x='feat', y='value', split=True, hue='Label', palette={0: "lightgreen", 1: "pink"}, hue_order=[0,1],
                linewidth=0.5, linecolor='dimgrey', scale='width', cut=0)

            label = ['BB_vdW', 'BB_ES', 'BB_HB', 'SC_vdW', 'SC_ES', 'SC_HB']
            axes_f[idx].set_title(residue)
            if (idx == 8) or (idx == 9):
                axes_f[idx].set_xticklabels(label, rotation=75, ha='right', color='black')
            else:
                #axes_f[idx].set_xticks(np.arange(5))  # Set x-tick positions
                axes_f[idx].set_xticklabels([''] * len(label))  # Remove x-tick labels
                #axes_f[idx].set_xticks([])  # Remove x-ticks
                #axes_f[idx].tick_params(axis='x')
            axes_f[idx].set_xlabel('')  # Remove x-axis label
            axes_f[idx].set_ylabel('')  # Remove y-axis label
            axes_f[idx].get_legend().remove() # Remove legend

        legend_handles = [
            Patch(facecolor='lightgreen', edgecolor='silver', label='Label 0 (Good - Bad)'),
            Patch(facecolor='pink', edgecolor='silver', label='Label 1 (Bad - Good)')
        ]
        fig.legend(handles=legend_handles, ncol=3, loc='lower center', edgecolor='white', bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)#, , )#, title='Label',
        plt.tight_layout(rect=[0, 0.02, 1, 1])
        plt.savefig('violin_subtractions_resi_' + str(i) + '.png', dpi=300)

    ####################################################################################################33

    plt.clf()
    sns.set(style="whitegrid", rc={"grid.color": 'silver', "grid.linewidth": 1, "axes.edgecolor": 'silver'})
    plt.rcParams['xtick.bottom'] = True  # add tick marks
    plt.rcParams['xtick.color'] = 'silver'
    plt.rcParams['axes.labelcolor'] = 'black'
    fig, axes_f = plt.subplots(1, 1, figsize=(8.5, 4)) # for plotting internals

    #axes_f = axes.flatten()

    features = internal

    data = pd.DataFrame()
    for f in features:
        d = pd.DataFrame({
            'feat': f,  # Set the entire 'feat' column to the feature name
            'value': train_set[f],  # Use the current feature column from train_set
            'Label': train_set['Label']
        })
        data = pd.concat([data, d], axis=0)
    data = data.reset_index()

    sns.violinplot(
        data=data, ax=axes_f, inner="quartile",
        x='feat', y='value', split=True, hue='Label', palette={0: "lightgreen", 1: "pink"}, hue_order=[0, 1],
        linewidth=0.5, linecolor='dimgrey', scale='width', cut=0)

    label = internal
    axes_f.set_title('Non residue terms')
    axes_f.set_xticklabels(label, rotation=75, ha='right', color='black')
    axes_f.set_xlabel('')  # Remove x-axis label
    axes_f.set_ylabel('')  # Remove y-axis label
    axes_f.get_legend().remove()  # Remove legend

    legend_handles = [
        Patch(facecolor='lightgreen', edgecolor='silver', label='Label 0 (Good - Bad)'),
        Patch(facecolor='pink', edgecolor='silver', label='Label 1 (Bad - Good)')
    ]
    fig.legend(handles=legend_handles, ncol=3, edgecolor='white', loc='lower center', bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)  # , , )#, title='Label',

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig('violin_subtractions_non-resi.png', dpi=150)


def plot_features(args):

    internals = ['Bonds','Angles','Torsions','Angles_M','Tors_M','vdW_14','Elec_14','vdW_15','Elec_15','HB_M','Bond_M','Wat_vdW','Wat_Elec','Wat_HB']

    if os.path.isfile(args.CSV_data_dir + 'all_features.csv') is True:
        df = pd.read_csv(args.CSV_data_dir + 'all_features.csv', index_col=False)
        df = df.set_index('Features')

        plt.rcParams.update({
            'axes.titlesize': 16,  # Title font size
            'axes.titleweight': 'bold',  # Title font weight
            'axes.labelsize': 14,  # X and Y axis labels font size
            'legend.fontsize': 16,  # Legend font size
            'xtick.labelsize': 6,  # X-axis tick labels font size
            'ytick.labelsize': 12  # Y-axis tick labels font size
        })

        width=15

        plt.clf()
        df['FeatureImportances'].plot(kind="bar", figsize=(width, 4), width=0.8, color='darkolivegreen')
        plt.title("Feature_importances_")
        plt.xlabel(None)
        #plt.legend(labels=internals, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('FeatureImportances.png', dpi=600)

        plt.clf()
        df['Cover'].plot(kind="bar", figsize=(width, 4), width=0.8, color='darkolivegreen')
        plt.title("Cover")
        plt.xlabel(None)
        #plt.legend(labels=internals, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('Cover.png', dpi=600)

        plt.clf()
        df['TotalCover'].plot(kind="bar", figsize=(width, 4), width=0.8, color='darkolivegreen')
        plt.title("Total cover")
        plt.xlabel(None)
        #plt.legend(labels=internals, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('TotalCover.png', dpi=600)

        plt.clf()
        df['Gain'].plot(kind="bar", figsize=(width, 4), width=0.8, color='darkolivegreen')
        plt.title("Gain")
        plt.xlabel(None)
        #plt.legend(labels=internals, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('Gain.png', dpi=600)

        plt.clf()
        df['TotalGain'].plot(kind="bar", figsize=(width, 4), width=0.8, color='darkolivegreen')
        plt.title("Total gain")
        plt.xlabel(None)
        #plt.legend(labels=internals, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('TotalGain.png', dpi=600)

        plt.clf()
        df['Weight'].plot(kind="bar", figsize=(width, 4), width=0.8, color='darkolivegreen')
        plt.title("Weight")
        plt.xlabel(None)
        #plt.legend(labels=internals, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('Weight.png', dpi=600)

        plt.clf()
        df['SHAP'].plot(kind="bar", figsize=(width, 4), width=0.8, color='darkolivegreen')
        plt.title("SHAP values (mean)")
        plt.xlabel(None)
        #plt.legend(labels=internals, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('SHAP.png', dpi=600)

        plt.clf()
        df['Permutations'].plot(kind="bar", figsize=(width, 4), width=0.8, color='darkolivegreen')
        plt.title("Permutation importances")
        plt.xlabel(None)
        #plt.legend(labels=internals, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('Permutations.png', dpi=600)

        plt.clf()
        df['Coefficients'].plot(kind="bar", figsize=(width, 4), width=0.8, color='darkolivegreen')
        plt.title("Coefficients")
        plt.xlabel(None)
        #plt.legend(labels=internals, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('Coefficients.png', dpi=600)

def plot_RESI_features(args):

    residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    interactions = ['BB_vdW', 'BB_ES', 'BB_HB', 'SC_vdW', 'SC_ES', 'SC_HB']
    features = ['FeatureImportances', 'Cover', 'TotalCover', 'Gain', 'TotalGain', 'Weight', 'SHAP', 'Permutations', 'Coefficients']

    FeatureImportances = pd.DataFrame(np.nan, index=residues, columns=interactions)
    Cover = pd.DataFrame()
    TotalCover = pd.DataFrame()
    Gain = pd.DataFrame()
    TotalGain = pd.DataFrame()
    Weight = pd.DataFrame()
    SHAP = pd.DataFrame()
    Permutations = pd.DataFrame()
    Coefficients = pd.DataFrame()

    for residue in residues:
        if os.path.isfile(args.CSV_data_dir + residue + '_features.csv') is True:
            df = pd.read_csv(args.CSV_data_dir + residue + '_features.csv', index_col=False)
            if 'FeatureImportances' in df.columns:
                for interaction in interactions:
                    if df['Features'].str.contains(interaction).any():
                        rowid = df.index[df['Features'].str.contains(interaction, na=False)].tolist()
                        FeatureImportances.loc[residue, interaction] = df.loc[rowid[0], 'FeatureImportances']
            if 'Cover' in df.columns:
                for interaction in interactions:
                    if df['Features'].str.contains(interaction).any():
                        rowid = df.index[df['Features'].str.contains(interaction, na=False)].tolist()
                        Cover.loc[residue, interaction] = df.loc[rowid[0], 'Cover']
            if 'TotalCover' in df.columns:
                for interaction in interactions:
                    if df['Features'].str.contains(interaction).any():
                        rowid = df.index[df['Features'].str.contains(interaction, na=False)].tolist()
                        TotalCover.loc[residue, interaction] = df.loc[rowid[0], 'TotalCover']
            if 'Gain' in df.columns:
                for interaction in interactions:
                    if df['Features'].str.contains(interaction).any():
                        rowid = df.index[df['Features'].str.contains(interaction, na=False)].tolist()
                        Gain.loc[residue, interaction] = df.loc[rowid[0], 'Gain']
            if 'TotalGain' in df.columns:
                for interaction in interactions:
                    if df['Features'].str.contains(interaction).any():
                        rowid = df.index[df['Features'].str.contains(interaction, na=False)].tolist()
                        TotalGain.loc[residue, interaction] = df.loc[rowid[0], 'TotalGain']
            if 'Weight' in df.columns:
                for interaction in interactions:
                    if df['Features'].str.contains(interaction).any():
                        rowid = df.index[df['Features'].str.contains(interaction, na=False)].tolist()
                        Weight.loc[residue, interaction] = df.loc[rowid[0], 'Weight']
            if 'SHAP' in df.columns:
                for interaction in interactions:
                    if df['Features'].str.contains(interaction).any():
                        rowid = df.index[df['Features'].str.contains(interaction, na=False)].tolist()
                        SHAP.loc[residue, interaction] = df.loc[rowid[0], 'SHAP']
            if 'Permutations' in df.columns:
                for interaction in interactions:
                    if df['Features'].str.contains(interaction).any():
                        rowid = df.index[df['Features'].str.contains(interaction, na=False)].tolist()
                        Permutations.loc[residue, interaction] = df.loc[rowid[0], 'Permutations']
            if 'Coefficients' in df.columns:
                for interaction in interactions:
                    if df['Features'].str.contains(interaction).any():
                        rowid = df.index[df['Features'].str.contains(interaction, na=False)].tolist()
                        Coefficients.loc[residue, interaction] = df.loc[rowid[0], 'Coefficients']

    #colors = ['yellowgreen', 'turquoise', 'steelblue', 'lightsalmon', 'gold', 'plum']
    colors = ['olivedrab', 'turquoise', 'steelblue', 'coral', 'gold', 'mediumpurple']
    labels = ['vdW (BB)', 'ES (BB)', 'HB (BB)', 'vdW (SC)', 'ES (SC)', 'HB (SC)']
    plt.rcParams.update({
        'axes.titlesize': 16,  # Title font size
        'axes.titleweight': 'bold',  # Title font weight
        'axes.labelsize': 14,  # X and Y axis labels font size
        'legend.fontsize': 16,  # Legend font size
        'xtick.labelsize': 14,  # X-axis tick labels font size
        'ytick.labelsize': 14  # Y-axis tick labels font size
    })

    plt.clf()
    FeatureImportances.plot(kind="bar", figsize=(15, 4), width=0.8, color=colors)
    plt.title("Feature_importances_")
    #plt.xlabel("Residue")
    #plt.legend(labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=6, frameon=False)
    for i in range(len(FeatureImportances) + 1):
        plt.axvline(x=i - 0.5, color='grey', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig('FeatureImportances.png')

    plt.clf()
    Cover.plot(kind="bar", figsize=(15, 4), width=0.8, color=colors)
    plt.title("Cover")
    #plt.xlabel("Residue")
    #plt.legend(labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=6, frameon=False)
    for i in range(len(Cover) + 1):
        plt.axvline(x=i - 0.5, color='grey', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig('Cover.png')

    plt.clf()
    TotalCover.plot(kind="bar", figsize=(15, 4), width=0.8, color=colors)
    plt.title("Total cover")
    #plt.xlabel("Residue")
    #plt.legend(labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=6, frameon=False)
    for i in range(len(TotalCover) + 1):
        plt.axvline(x=i - 0.5, color='grey', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig('TotalCover.png')

    plt.clf()
    Gain.plot(kind="bar", figsize=(15, 4), width=0.8, color=colors)
    plt.title("Gain")
    #plt.xlabel("Residue")
    #plt.legend(labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=6, frameon=False)
    for i in range(len(Gain) + 1):
        plt.axvline(x=i - 0.5, color='grey', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig('Gain.png')

    plt.clf()
    TotalGain.plot(kind="bar", figsize=(15, 4), width=0.8, color=colors)
    plt.title("TotalGain")
    #plt.xlabel("Residue")
    #plt.legend(labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=6, frameon=False)
    for i in range(len(TotalGain) + 1):
        plt.axvline(x=i - 0.5, color='grey', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig('TotalGain.png')

    plt.clf()
    Weight.plot(kind="bar", figsize=(15, 4), width=0.8, color=colors)
    plt.title("Weight")
    #plt.xlabel("Residue")
    #plt.legend(labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=6, frameon=False)
    for i in range(len(Weight) + 1):
        plt.axvline(x=i - 0.5, color='grey', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig('Weight.png')

    plt.clf()
    SHAP.plot(kind="bar", figsize=(15, 4), width=0.8, color=colors)
    plt.title("SHAP values (mean)")
    #plt.xlabel("Residue")
    #plt.legend(labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=6, frameon=False)
    for i in range(len(SHAP) + 1):
        plt.axvline(x=i - 0.5, color='grey', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig('SHAP.png')

    plt.clf()
    Permutations.plot(kind="bar", figsize=(15, 4), width=0.8, color=colors)
    plt.title("Permutation importances")
    #plt.xlabel("Residue")
    #plt.legend(labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=6, frameon=False)
    for i in range(len(Permutations) + 1):
        plt.axvline(x=i - 0.5, color='grey', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig('Permutations.png')

    plt.clf()
    Coefficients.plot(kind="bar", figsize=(15, 5), width=0.8, color=colors)
    plt.title("Coefficients")
    #plt.xlabel("Residue")
    #plt.legend(labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=6, frameon=False)
    for i in range(len(Coefficients) + 1):
        plt.axvline(x=i - 0.5, color='grey', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig('Coefficients.png')

def plot_RESI_SHAP(args):

    train_set_X, train_set_y, val_set_X, val_set_y, test_set_X, test_set_y, features = split_set(args)
    list = [['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE'],
            ['LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']]
    interactions = ['BB_vdW', 'BB_ES', 'BB_HB', 'SC_vdW', 'SC_ES', 'SC_HB']

    for i, residues in enumerate(list):
        plt.clf()
        #plt.rcParams['xtick.bottom'] = True # add tick marks
        #plt.rcParams['xtick.major.size'] = 20
        #plt.rcParams['xtick.major.width'] = 4
        #plt.rcParams['xtick.color'] = 'silver'
        #plt.rcParams['axes.labelcolor'] = 'black'

        # Create a figure with a grid of subplots
        fig, axes = plt.subplots(5, 2, figsize=(8.5, 11))

        # Flatten axes for easy indexing
        axes_f = axes.flatten()

        for idx, residue in enumerate(residues):

            # Set the current axis for each plot
            plt.sca(axes_f[i])

            # Load model
            model = xgb.Booster()
            model.load_model('model/vdW_1x/XGBc_' + residue + '.json')

            train_set_X = train_set_X.filter(regex=f'^{residue}')
            feature_rename_dict = {
                residue + '_BB_vdW': 'vdW (BB)',
                residue + '_BB_ES': 'ES (BB)',
                residue + '_BB_HB': 'HB (BB)',
                residue + '_SC_vdW': 'vdW (SC)',
                residue + '_SC_ES': 'ES (SC)',
                residue + '_SC_HB': 'HB (SC)',
            }
            train_set_X = train_set_X.rename(columns=feature_rename_dict)

            # Get SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(train_set_X, check_additivity=False)
            #shap.plots.beeswarm(shap_values)
            shap.summary_plot(shap_values, train_set_X, max_display=len(train_set_X.columns), show=False)
            axes[i].set_title(residue, fontsize=12)

        plt.savefig('SHAP_' + i + '.png')

def plot_GNN_accuracies():

    data = [
        {
            'n_graph_layers' : 4,
            'n_FC_layers' : 3,
            'embedding_size' : 200,
            'epoch' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37],
            'train' : [0.724,0.83,0.859,0.856,0.871,0.866,0.866,0.874,0.87,0.876,0.882,0.885,0.89,0.891,0.889,0.874,0.898,0.901,0.895,0.861,0.902,0.897,0.901,0.872,0.907,0.892,0.913,0.907,0.914,0.916,0.927,0.915,0.924,0.926,0.931,0.929,0.919],
            'test' :  [0.746,0.85,0.879,0.873,0.89,0.886,0.883,0.887,0.884,0.888,0.895,0.898,0.9,0.9,0.9,0.887,0.907,0.91,0.903,0.877,0.91,0.905,0.912,0.881,0.916,0.901,0.922,0.912,0.923,0.917,0.927,0.917,0.925,0.928,0.93,0.931,0.927],
            'val' :   [0.723,0.825,0.858,0.852,0.86,0.861,0.859,0.865,0.866,0.868,0.876,0.879,0.883,0.883,0.882,0.868,0.889,0.892,0.883,0.851,0.897,0.881,0.897,0.859,0.901,0.886,0.907,0.889,0.907,0.903,0.914,0.901,0.907,0.914,0.918,0.916,0.909]
        },

        {
            'n_graph_layers' : 3,
            'n_FC_layers' : 2,
            'embedding_size' : 200,
            'epoch' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67],
            'train' : [0.836,0.849,0.869,0.867,0.859,0.883,0.887,0.887,0.895,0.893,0.889,0.905,0.91,0.921,0.922,0.911,0.917,0.925,0.915,0.931,0.926,0.937,0.937,0.933,0.931,0.932,0.939,0.922,0.942,0.945,0.945,0.945,0.939,0.945,0.94,0.943,0.94,0.951,0.947,0.935,0.949,0.94,0.945,0.948,0.947,0.954,0.918,0.954,0.953,0.952,0.952,0.943,0.951,0.95,0.954,0.95,0.96,0.957,0.953,0.959,0.953,0.961,0.958,0.959,0.945,0.947,0.928],
            'test' :  [0.854,0.87,0.883,0.879,0.869,0.897,0.896,0.896,0.902,0.899,0.897,0.912,0.912,0.922,0.92,0.907,0.917,0.923,0.919,0.926,0.928,0.932,0.935,0.929,0.924,0.933,0.937,0.924,0.935,0.938,0.939,0.94,0.932,0.938,0.934,0.934,0.935,0.941,0.938,0.93,0.939,0.933,0.937,0.937,0.939,0.942,0.916,0.943,0.938,0.94,0.939,0.933,0.937,0.939,0.941,0.937,0.942,0.94,0.938,0.941,0.939,0.943,0.942,0.939,0.933,0.936,0.917],
            'val' :   [0.822,0.845,0.859,0.854,0.852,0.871,0.881,0.878,0.883,0.88,0.885,0.889,0.898,0.904,0.901,0.887,0.903,0.906,0.902,0.907,0.911,0.92,0.919,0.914,0.912,0.915,0.921,0.904,0.923,0.927,0.924,0.927,0.919,0.927,0.916,0.923,0.918,0.928,0.924,0.918,0.927,0.916,0.922,0.924,0.923,0.93,0.901,0.929,0.926,0.928,0.927,0.926,0.925,0.929,0.928,0.924,0.933,0.927,0.927,0.928,0.921,0.93,0.929,0.927,0.915,0.923,0.904]
        },
        {
            'n_graph_layers' : 4,
            'n_FC_layers' : 3,
            'embedding_size' : 150,
            'epoch' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
            'train' : [0.777, 0.846, 0.854, 0.871, 0.883, 0.885, 0.891, 0.863, 0.897, 0.895, 0.897, 0.893, 0.888, 0.895, 0.904, 0.896, 0.916, 0.913, 0.918, 0.92, 0.922, 0.924, 0.927, 0.914, 0.927, 0.931, 0.924, 0.92, 0.933, 0.93, 0.931, 0.932, 0.936, 0.94, 0.937, 0.927, 0.94, 0.94, 0.947, 0.946, 0.925, 0.949, 0.949, 0.94, 0.944, 0.944, 0.951, 0.947, 0.953, 0.942, 0.946, 0.951, 0.953, 0.961, 0.924, 0.957, 0.951, 0.957, 0.956, 0.956, 0.96, 0.958, 0.958, 0.959, 0.958, 0.951, 0.957, 0.959, 0.958, 0.963, 0.96, 0.955, 0.96, 0.964, 0.961, 0.957, 0.966, 0.964, 0.964, 0.964, 0.963, 0.964, 0.956, 0.969, 0.962, 0.965, 0.964, 0.969, 0.969, 0.969, 0.97, 0.963, 0.968, 0.962, 0.963, 0.971, 0.969, 0.973, 0.972, 0.97, 0.965, 0.969, 0.971, 0.969, 0.967, 0.968, 0.973, 0.975, 0.971, 0.976, 0.973, 0.974, 0.973, 0.974, 0.976, 0.973, 0.976, 0.97, 0.967],
            'test' :  [0.795, 0.864, 0.875, 0.888, 0.893, 0.896, 0.902, 0.88, 0.905, 0.903, 0.9, 0.899, 0.892, 0.903, 0.907, 0.902, 0.918, 0.917, 0.921, 0.922, 0.923, 0.923, 0.926, 0.914, 0.928, 0.93, 0.922, 0.92, 0.934, 0.928, 0.932, 0.93, 0.932, 0.936, 0.935, 0.922, 0.937, 0.935, 0.941, 0.938, 0.924, 0.941, 0.942, 0.932, 0.939, 0.931, 0.939, 0.939, 0.944, 0.932, 0.938, 0.942, 0.945, 0.946, 0.916, 0.944, 0.94, 0.944, 0.944, 0.943, 0.944, 0.944, 0.941, 0.943, 0.945, 0.938, 0.947, 0.943, 0.942, 0.946, 0.945, 0.943, 0.941, 0.945, 0.941, 0.939, 0.948, 0.946, 0.946, 0.941, 0.942, 0.945, 0.94, 0.947, 0.939, 0.941, 0.943, 0.947, 0.945, 0.942, 0.947, 0.941, 0.944, 0.938, 0.938, 0.945, 0.945, 0.945, 0.947, 0.944, 0.94, 0.944, 0.943, 0.943, 0.935, 0.942, 0.944, 0.944, 0.945, 0.944, 0.945, 0.946, 0.943, 0.946, 0.946, 0.941, 0.946, 0.942, 0.937],
            'val' :   [0.786, 0.842, 0.855, 0.866, 0.871, 0.876, 0.88, 0.855, 0.883, 0.882, 0.88, 0.876, 0.866, 0.883, 0.885, 0.876, 0.902, 0.898, 0.906, 0.906, 0.908, 0.913, 0.914, 0.899, 0.91, 0.914, 0.904, 0.903, 0.916, 0.912, 0.916, 0.912, 0.914, 0.919, 0.918, 0.906, 0.923, 0.913, 0.924, 0.925, 0.902, 0.927, 0.928, 0.914, 0.923, 0.916, 0.923, 0.924, 0.929, 0.914, 0.924, 0.926, 0.933, 0.929, 0.892, 0.93, 0.925, 0.927, 0.93, 0.931, 0.93, 0.93, 0.929, 0.926, 0.93, 0.923, 0.932, 0.929, 0.926, 0.93, 0.929, 0.927, 0.93, 0.932, 0.926, 0.925, 0.935, 0.933, 0.932, 0.929, 0.933, 0.933, 0.929, 0.937, 0.925, 0.928, 0.928, 0.931, 0.93, 0.931, 0.935, 0.929, 0.932, 0.928, 0.926, 0.929, 0.936, 0.935, 0.934, 0.929, 0.925, 0.932, 0.931, 0.93, 0.924, 0.931, 0.932, 0.933, 0.932, 0.933, 0.929, 0.933, 0.929, 0.934, 0.935, 0.929, 0.933, 0.928, 0.924]
        },
        {
            'n_graph_layers' : 3,
            'n_FC_layers' : 3,
            'embedding_size' : 150,
            'epoch' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40],
            'train' : [0.729,0.815,0.855,0.868,0.878,0.848,0.83,0.887,0.891,0.901,0.873,0.912,0.891,0.913,0.916,0.92,0.925,0.932,0.923,0.929,0.926,0.913,0.934,0.934,0.928,0.905,0.939,0.928,0.938,0.936,0.934,0.943,0.944,0.93,0.909,0.936,0.936,0.944,0.95,0.939],
            'test' :  [0.75,0.842,0.869,0.879,0.884,0.863,0.844,0.893,0.897,0.901,0.882,0.91,0.901,0.915,0.917,0.924,0.924,0.929,0.916,0.925,0.925,0.912,0.93,0.926,0.922,0.9,0.928,0.919,0.925,0.923,0.923,0.935,0.933,0.921,0.899,0.924,0.927,0.93,0.934,0.93],
            'val' :   [0.731,0.822,0.841,0.857,0.864,0.841,0.821,0.871,0.878,0.887,0.866,0.895,0.882,0.903,0.903,0.91,0.907,0.913,0.904,0.912,0.908,0.891,0.917,0.912,0.908,0.884,0.918,0.911,0.92,0.914,0.91,0.923,0.918,0.905,0.89,0.915,0.91,0.917,0.923,0.917]
        },
        {
            'n_graph_layers' : 3,
            'n_FC_layers' : 2,
            'embedding_size' : 150,
            'epoch' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
            'train' : [0.816,0.842,0.859,0.872,0.876,0.875,0.889,0.895,0.899,0.89,0.901,0.909,0.919,0.924,0.911,0.902,0.928,0.92,0.927,0.93,0.924,0.924,0.936,0.935,0.936,0.935,0.937,0.938,0.942,0.94,0.94,0.911],
            'test' :  [0.843,0.858,0.873,0.883,0.884,0.88,0.897,0.897,0.907,0.899,0.902,0.916,0.92,0.922,0.911,0.903,0.927,0.923,0.926,0.929,0.922,0.923,0.926,0.931,0.932,0.929,0.934,0.934,0.933,0.931,0.936,0.909],
            'val' :   [0.806,0.829,0.848,0.863,0.866,0.863,0.876,0.882,0.891,0.876,0.885,0.897,0.907,0.912,0.895,0.889,0.913,0.907,0.913,0.912,0.904,0.912,0.917,0.918,0.918,0.915,0.918,0.921,0.923,0.915,0.918,0.892]
        },
        {
            'n_graph_layers' : 2,
            'n_FC_layers' : 2,
            'embedding_size' : 150,
            'epoch' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67],
            'train' : [0.836,0.849,0.869,0.867,0.859,0.883,0.887,0.887,0.895,0.893,0.889,0.905,0.91,0.921,0.922,0.911,0.917,0.925,0.915,0.931,0.926,0.937,0.937,0.933,0.931,0.932,0.939,0.922,0.942,0.945,0.945,0.945,0.939,0.945,0.94,0.943,0.94,0.951,0.947,0.935,0.949,0.94,0.945,0.948,0.947,0.954,0.918,0.954,0.953,0.952,0.952,0.943,0.951,0.95,0.954,0.95,0.96,0.957,0.953,0.959,0.953,0.961,0.958,0.959,0.945,0.947,0.928],
            'test' :  [0.854,0.87,0.883,0.879,0.869,0.897,0.896,0.896,0.902,0.899,0.897,0.912,0.912,0.922,0.92,0.907,0.917,0.923,0.919,0.926,0.928,0.932,0.935,0.929,0.924,0.933,0.937,0.924,0.935,0.938,0.939,0.94,0.932,0.938,0.934,0.934,0.935,0.941,0.938,0.93,0.939,0.933,0.937,0.937,0.939,0.942,0.916,0.943,0.938,0.94,0.939,0.933,0.937,0.939,0.941,0.937,0.942,0.94,0.938,0.941,0.939,0.943,0.942,0.939,0.933,0.936,0.917],
            'val' :   [0.822,0.845,0.859,0.854,0.852,0.871,0.881,0.878,0.883,0.88,0.885,0.889,0.898,0.904,0.901,0.887,0.903,0.906,0.902,0.907,0.911,0.92,0.919,0.914,0.912,0.915,0.921,0.904,0.923,0.927,0.924,0.927,0.919,0.927,0.916,0.923,0.918,0.928,0.924,0.918,0.927,0.916,0.922,0.924,0.923,0.93,0.901,0.929,0.926,0.928,0.927,0.926,0.925,0.929,0.928,0.924,0.933,0.927,0.927,0.928,0.921,0.93,0.929,0.927,0.915,0.923,0.904]
        },
        {
            'n_graph_layers' : 3,
            'n_FC_layers' : 3,
            'embedding_size' : 128,
            'epoch' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69],
            'train' : [0.821,0.858,0.849,0.869,0.881,0.874,0.887,0.891,0.902,0.9,0.906,0.912,0.915,0.917,0.916,0.918,0.923,0.921,0.922,0.914,0.916,0.925,0.927,0.931,0.928,0.928,0.933,0.934,0.937,0.933,0.942,0.941,0.935,0.94,0.932,0.945,0.926,0.942,0.938,0.944,0.93,0.944,0.947,0.915,0.943,0.947,0.945,0.944,0.95,0.948,0.95,0.946,0.949,0.946,0.948,0.918,0.953,0.952,0.955,0.946,0.955,0.949,0.954,0.952,0.956,0.946,0.95,0.953,0.954],
            'test' :  [0.842,0.876,0.864,0.886,0.894,0.881,0.895,0.894,0.906,0.904,0.911,0.915,0.914,0.918,0.918,0.919,0.928,0.921,0.921,0.915,0.911,0.927,0.925,0.923,0.926,0.924,0.925,0.927,0.933,0.927,0.933,0.933,0.927,0.931,0.921,0.935,0.921,0.934,0.931,0.932,0.917,0.93,0.931,0.902,0.927,0.932,0.93,0.929,0.935,0.93,0.93,0.934,0.932,0.929,0.928,0.906,0.933,0.933,0.933,0.927,0.933,0.929,0.934,0.93,0.933,0.927,0.928,0.93,0.932],
            'val' :   [0.813,0.844,0.843,0.86,0.869,0.858,0.87,0.871,0.885,0.883,0.889,0.894,0.897,0.898,0.898,0.897,0.908,0.905,0.9,0.897,0.885,0.91,0.91,0.908,0.91,0.911,0.905,0.905,0.915,0.915,0.913,0.914,0.905,0.914,0.906,0.917,0.901,0.912,0.912,0.911,0.902,0.917,0.918,0.878,0.911,0.917,0.913,0.911,0.918,0.913,0.92,0.919,0.918,0.915,0.91,0.886,0.917,0.915,0.92,0.915,0.917,0.911,0.917,0.917,0.92,0.913,0.918,0.917,0.921]
        },
        {
            'n_graph_layers' : 3,
            'n_FC_layers' : 2,
            'embedding_size' : 128,
            'epoch' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98],
            'train' : [0.797,0.844,0.864,0.88,0.881,0.878,0.893,0.891,0.901,0.9,0.892,0.908,0.902,0.922,0.912,0.921,0.901,0.914,0.927,0.925,0.924,0.921,0.905,0.936,0.934,0.938,0.935,0.93,0.908,0.941,0.944,0.935,0.94,0.943,0.936,0.942,0.945,0.947,0.935,0.905,0.947,0.944,0.942,0.915,0.953,0.95,0.941,0.954,0.953,0.959,0.956,0.952,0.96,0.955,0.956,0.952,0.952,0.949,0.96,0.954,0.943,0.963,0.935,0.963,0.96,0.963,0.963,0.957,0.96,0.964,0.966,0.959,0.966,0.957,0.963,0.969,0.957,0.954,0.965,0.97,0.972,0.955,0.971,0.968,0.971,0.965,0.972,0.96,0.966,0.972,0.962,0.972,0.969,0.956,0.97,0.974,0.976,0.972],
            'test' :  [0.818,0.861,0.881,0.891,0.892,0.885,0.901,0.894,0.905,0.905,0.901,0.908,0.906,0.922,0.912,0.922,0.901,0.919,0.927,0.92,0.92,0.921,0.905,0.929,0.93,0.929,0.929,0.915,0.903,0.93,0.935,0.929,0.932,0.933,0.93,0.929,0.932,0.936,0.926,0.894,0.933,0.93,0.925,0.905,0.936,0.931,0.925,0.936,0.934,0.94,0.933,0.929,0.938,0.934,0.938,0.928,0.931,0.925,0.937,0.932,0.92,0.937,0.918,0.935,0.933,0.937,0.932,0.929,0.936,0.938,0.938,0.929,0.935,0.93,0.936,0.936,0.925,0.924,0.933,0.936,0.939,0.928,0.937,0.936,0.938,0.931,0.935,0.924,0.928,0.935,0.927,0.937,0.934,0.92,0.934,0.935,0.938,0.933],
            'val' :   [0.785,0.834,0.854,0.867,0.868,0.866,0.881,0.877,0.883,0.883,0.873,0.889,0.884,0.905,0.897,0.905,0.879,0.904,0.909,0.903,0.901,0.904,0.888,0.916,0.915,0.916,0.914,0.905,0.876,0.915,0.919,0.915,0.917,0.92,0.915,0.913,0.922,0.919,0.908,0.875,0.918,0.915,0.915,0.898,0.922,0.919,0.908,0.925,0.923,0.922,0.917,0.913,0.923,0.92,0.922,0.916,0.917,0.907,0.922,0.917,0.905,0.921,0.904,0.921,0.917,0.921,0.915,0.915,0.916,0.921,0.922,0.915,0.923,0.919,0.923,0.923,0.914,0.91,0.921,0.921,0.925,0.915,0.922,0.92,0.921,0.917,0.918,0.911,0.913,0.922,0.909,0.921,0.919,0.903,0.92,0.918,0.924,0.921]
        },
        {
            'n_graph_layers' : 2,
            'n_FC_layers' : 2,
            'embedding_size' : 128,
            'epoch' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61],
            'train' : [0.807,0.825,0.851,0.855,0.822,0.881,0.871,0.877,0.873,0.885,0.9,0.88,0.904,0.905,0.887,0.902,0.908,0.907,0.899,0.909,0.918,0.922,0.913,0.928,0.912,0.933,0.927,0.915,0.919,0.924,0.942,0.931,0.928,0.92,0.942,0.94,0.945,0.951,0.948,0.954,0.95,0.945,0.956,0.943,0.953,0.956,0.956,0.946,0.958,0.963,0.959,0.96,0.952,0.962,0.963,0.966,0.961,0.967,0.97,0.963,0.964],
            'test' :  [0.832,0.84,0.868,0.87,0.829,0.883,0.878,0.88,0.874,0.886,0.893,0.875,0.902,0.893,0.877,0.894,0.891,0.893,0.887,0.891,0.9,0.904,0.898,0.905,0.896,0.911,0.898,0.895,0.889,0.899,0.91,0.902,0.898,0.894,0.912,0.907,0.912,0.909,0.912,0.912,0.91,0.905,0.91,0.905,0.911,0.911,0.913,0.9,0.909,0.91,0.906,0.907,0.901,0.908,0.91,0.907,0.908,0.91,0.909,0.906,0.903],
            'val' :   [0.797,0.799,0.833,0.837,0.79,0.857,0.85,0.851,0.854,0.86,0.872,0.861,0.878,0.871,0.862,0.873,0.868,0.876,0.866,0.876,0.887,0.887,0.879,0.889,0.879,0.892,0.885,0.868,0.869,0.884,0.892,0.881,0.878,0.876,0.894,0.888,0.896,0.893,0.889,0.894,0.889,0.886,0.894,0.881,0.89,0.896,0.894,0.875,0.889,0.894,0.889,0.889,0.882,0.888,0.889,0.888,0.886,0.891,0.889,0.889,0.882]
        },
        {
            'n_graph_layers' : 3,
            'n_FC_layers' : 3,
            'embedding_size' : 64,
            'epoch' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40],
            'train' : [0.796,0.853,0.874,0.886,0.883,0.895,0.904,0.903,0.898,0.907,0.911,0.91,0.908,0.917,0.924,0.921,0.918,0.917,0.918,0.923,0.926,0.935,0.933,0.935,0.933,0.935,0.932,0.928,0.943,0.935,0.938,0.942,0.939,0.945,0.934,0.949,0.95,0.949,0.939,0.942],
            'test' :  [0.833,0.865,0.883,0.89,0.881,0.896,0.902,0.904,0.896,0.907,0.908,0.91,0.912,0.912,0.919,0.911,0.91,0.912,0.908,0.917,0.916,0.923,0.918,0.92,0.919,0.915,0.915,0.911,0.924,0.913,0.917,0.922,0.916,0.92,0.907,0.921,0.922,0.923,0.908,0.917],
            'val' :   [0.801,0.837,0.862,0.867,0.86,0.875,0.887,0.884,0.876,0.882,0.886,0.891,0.891,0.895,0.899,0.892,0.897,0.894,0.89,0.898,0.899,0.899,0.904,0.902,0.9,0.9,0.893,0.9,0.906,0.893,0.902,0.906,0.9,0.902,0.89,0.903,0.908,0.908,0.886,0.901]
        },
        {
            'n_graph_layers' : 3,
            'n_FC_layers' : 2,
            'embedding_size' : 64,
            'epoch' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245],
            'train' : [0.819,0.848,0.855,0.877,0.885,0.875,0.894,0.898,0.905,0.91,0.906,0.915,0.914,0.913,0.919,0.902,0.92,0.922,0.928,0.924,0.929,0.935,0.936,0.934,0.932,0.937,0.937,0.939,0.938,0.941,0.94,0.942,0.943,0.949,0.941,0.945,0.938,0.947,0.946,0.952,0.948,0.951,0.952,0.945,0.957,0.955,0.959,0.955,0.958,0.961,0.952,0.96,0.957,0.953,0.962,0.958,0.957,0.966,0.964,0.962,0.962,0.961,0.963,0.962,0.967,0.969,0.968,0.964,0.963,0.968,0.967,0.968,0.967,0.971,0.967,0.967,0.968,0.967,0.971,0.968,0.969,0.96,0.975,0.969,0.977,0.971,0.97,0.974,0.976,0.975,0.974,0.967,0.976,0.977,0.973,0.977,0.976,0.979,0.974,0.977,0.972,0.977,0.98,0.98,0.972,0.98,0.979,0.976,0.979,0.976,0.982,0.979,0.981,0.978,0.98,0.981,0.977,0.982,0.976,0.983,0.983,0.984,0.982,0.975,0.982,0.977,0.985,0.98,0.98,0.984,0.982,0.985,0.98,0.984,0.979,0.983,0.982,0.973,0.983,0.985,0.983,0.985,0.985,0.985,0.98,0.982,0.984,0.982,0.984,0.984,0.984,0.983,0.985,0.984,0.988,0.987,0.984,0.983,0.987,0.984,0.987,0.987,0.987,0.982,0.987,0.984,0.987,0.986,0.986,0.986,0.988,0.986,0.986,0.987,0.986,0.985,0.989,0.987,0.988,0.989,0.986,0.989,0.99,0.983,0.986,0.989,0.988,0.989,0.987,0.987,0.987,0.988,0.988,0.991,0.989,0.989,0.987,0.988,0.987,0.988,0.987,0.988,0.988,0.988,0.99,0.989,0.989,0.988,0.991,0.989,0.989,0.991,0.99,0.99,0.988,0.991,0.991,0.991,0.992,0.988,0.991,0.99,0.99,0.991,0.991,0.987,0.991,0.992,0.989,0.989,0.987,0.993,0.99,0.99,0.99,0.99,0.988,0.991,0.989,0.991,0.991,0.987,0.991,0.992,0.99],
            'test' :  [0.841,0.859,0.87,0.888,0.893,0.887,0.897,0.902,0.906,0.909,0.909,0.913,0.911,0.914,0.915,0.894,0.912,0.912,0.917,0.911,0.918,0.924,0.926,0.922,0.917,0.924,0.922,0.922,0.921,0.924,0.924,0.923,0.926,0.928,0.92,0.921,0.916,0.922,0.922,0.926,0.926,0.928,0.927,0.923,0.931,0.931,0.928,0.925,0.933,0.933,0.924,0.933,0.924,0.922,0.928,0.927,0.924,0.932,0.93,0.926,0.925,0.927,0.926,0.923,0.931,0.93,0.931,0.928,0.919,0.924,0.926,0.928,0.924,0.931,0.925,0.926,0.927,0.925,0.926,0.92,0.922,0.916,0.928,0.922,0.927,0.922,0.924,0.928,0.931,0.927,0.923,0.919,0.925,0.927,0.923,0.923,0.925,0.927,0.917,0.927,0.92,0.923,0.925,0.925,0.916,0.926,0.928,0.923,0.918,0.919,0.926,0.921,0.922,0.923,0.924,0.931,0.924,0.927,0.921,0.926,0.928,0.922,0.924,0.92,0.923,0.923,0.924,0.924,0.92,0.923,0.92,0.928,0.925,0.924,0.92,0.922,0.921,0.911,0.921,0.924,0.925,0.922,0.925,0.923,0.923,0.919,0.924,0.918,0.925,0.923,0.924,0.926,0.92,0.923,0.923,0.924,0.923,0.919,0.927,0.918,0.922,0.922,0.924,0.919,0.925,0.921,0.922,0.924,0.924,0.921,0.929,0.922,0.923,0.92,0.922,0.922,0.926,0.918,0.924,0.924,0.919,0.928,0.922,0.919,0.912,0.924,0.921,0.921,0.919,0.923,0.92,0.924,0.921,0.923,0.926,0.921,0.924,0.926,0.923,0.919,0.923,0.922,0.923,0.923,0.924,0.924,0.92,0.92,0.922,0.923,0.919,0.924,0.923,0.922,0.921,0.921,0.919,0.922,0.924,0.921,0.923,0.923,0.918,0.92,0.923,0.921,0.921,0.922,0.918,0.917,0.919,0.925,0.924,0.924,0.922,0.92,0.919,0.921,0.917,0.919,0.921,0.92,0.918,0.925,0.917],
            'val' :   [0.805,0.831,0.847,0.864,0.876,0.865,0.882,0.88,0.887,0.888,0.884,0.889,0.893,0.895,0.895,0.871,0.896,0.891,0.895,0.895,0.9,0.909,0.91,0.907,0.903,0.908,0.907,0.908,0.91,0.911,0.907,0.908,0.914,0.915,0.906,0.906,0.898,0.911,0.906,0.904,0.911,0.911,0.915,0.906,0.919,0.913,0.914,0.91,0.915,0.92,0.915,0.921,0.911,0.907,0.912,0.915,0.915,0.918,0.918,0.91,0.915,0.915,0.917,0.911,0.921,0.916,0.915,0.908,0.902,0.915,0.915,0.917,0.913,0.916,0.91,0.911,0.91,0.908,0.912,0.91,0.911,0.904,0.915,0.909,0.916,0.906,0.913,0.917,0.917,0.915,0.913,0.908,0.913,0.918,0.912,0.916,0.918,0.918,0.906,0.918,0.907,0.911,0.913,0.913,0.906,0.916,0.915,0.912,0.911,0.903,0.917,0.912,0.917,0.913,0.913,0.916,0.908,0.912,0.907,0.917,0.917,0.915,0.916,0.909,0.914,0.909,0.913,0.914,0.912,0.911,0.911,0.917,0.911,0.91,0.91,0.91,0.909,0.905,0.909,0.912,0.913,0.914,0.916,0.917,0.906,0.908,0.914,0.91,0.913,0.916,0.914,0.916,0.912,0.911,0.911,0.91,0.914,0.91,0.918,0.913,0.914,0.909,0.917,0.909,0.911,0.912,0.908,0.913,0.912,0.913,0.915,0.911,0.913,0.909,0.916,0.911,0.914,0.906,0.913,0.909,0.91,0.914,0.911,0.905,0.909,0.914,0.914,0.915,0.909,0.913,0.91,0.916,0.909,0.915,0.914,0.914,0.912,0.913,0.912,0.908,0.915,0.91,0.914,0.91,0.91,0.914,0.91,0.914,0.914,0.913,0.911,0.916,0.914,0.916,0.915,0.916,0.913,0.911,0.917,0.91,0.915,0.914,0.914,0.91,0.915,0.911,0.914,0.915,0.914,0.909,0.906,0.913,0.913,0.917,0.91,0.908,0.909,0.91,0.905,0.916,0.912,0.911,0.91,0.915,0.912]
        },
        {
            'n_graph_layers' : 2,
            'n_FC_layers' : 2,
            'embedding_size' : 64,
            'epoch' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245],
            'train' : [0.819,0.848,0.855,0.877,0.885,0.875,0.894,0.898,0.905,0.91,0.906,0.915,0.914,0.913,0.919,0.902,0.92,0.922,0.928,0.924,0.929,0.935,0.936,0.934,0.932,0.937,0.937,0.939,0.938,0.941,0.94,0.942,0.943,0.949,0.941,0.945,0.938,0.947,0.946,0.952,0.948,0.951,0.952,0.945,0.957,0.955,0.959,0.955,0.958,0.961,0.952,0.96,0.957,0.953,0.962,0.958,0.957,0.966,0.964,0.962,0.962,0.961,0.963,0.962,0.967,0.969,0.968,0.964,0.963,0.968,0.967,0.968,0.967,0.971,0.967,0.967,0.968,0.967,0.971,0.968,0.969,0.96,0.975,0.969,0.977,0.971,0.97,0.974,0.976,0.975,0.974,0.967,0.976,0.977,0.973,0.977,0.976,0.979,0.974,0.977,0.972,0.977,0.98,0.98,0.972,0.98,0.979,0.976,0.979,0.976,0.982,0.979,0.981,0.978,0.98,0.981,0.977,0.982,0.976,0.983,0.983,0.984,0.982,0.975,0.982,0.977,0.985,0.98,0.98,0.984,0.982,0.985,0.98,0.984,0.979,0.983,0.982,0.973,0.983,0.985,0.983,0.985,0.985,0.985,0.98,0.982,0.984,0.982,0.984,0.984,0.984,0.983,0.985,0.984,0.988,0.987,0.984,0.983,0.987,0.984,0.987,0.987,0.987,0.982,0.987,0.984,0.987,0.986,0.986,0.986,0.988,0.986,0.986,0.987,0.986,0.985,0.989,0.987,0.988,0.989,0.986,0.989,0.99,0.983,0.986,0.989,0.988,0.989,0.987,0.987,0.987,0.988,0.988,0.991,0.989,0.989,0.987,0.988,0.987,0.988,0.987,0.988,0.988,0.988,0.99,0.989,0.989,0.988,0.991,0.989,0.989,0.991,0.99,0.99,0.988,0.991,0.991,0.991,0.992,0.988,0.991,0.99,0.99,0.991,0.991,0.987,0.991,0.992,0.989,0.989,0.987,0.993,0.99,0.99,0.99,0.99,0.988,0.991,0.989,0.991,0.991,0.987,0.991,0.992,0.99],
            'test' :  [0.841,0.859,0.87,0.888,0.893,0.887,0.897,0.902,0.906,0.909,0.909,0.913,0.911,0.914,0.915,0.894,0.912,0.912,0.917,0.911,0.918,0.924,0.926,0.922,0.917,0.924,0.922,0.922,0.921,0.924,0.924,0.923,0.926,0.928,0.92,0.921,0.916,0.922,0.922,0.926,0.926,0.928,0.927,0.923,0.931,0.931,0.928,0.925,0.933,0.933,0.924,0.933,0.924,0.922,0.928,0.927,0.924,0.932,0.93,0.926,0.925,0.927,0.926,0.923,0.931,0.93,0.931,0.928,0.919,0.924,0.926,0.928,0.924,0.931,0.925,0.926,0.927,0.925,0.926,0.92,0.922,0.916,0.928,0.922,0.927,0.922,0.924,0.928,0.931,0.927,0.923,0.919,0.925,0.927,0.923,0.923,0.925,0.927,0.917,0.927,0.92,0.923,0.925,0.925,0.916,0.926,0.928,0.923,0.918,0.919,0.926,0.921,0.922,0.923,0.924,0.931,0.924,0.927,0.921,0.926,0.928,0.922,0.924,0.92,0.923,0.923,0.924,0.924,0.92,0.923,0.92,0.928,0.925,0.924,0.92,0.922,0.921,0.911,0.921,0.924,0.925,0.922,0.925,0.923,0.923,0.919,0.924,0.918,0.925,0.923,0.924,0.926,0.92,0.923,0.923,0.924,0.923,0.919,0.927,0.918,0.922,0.922,0.924,0.919,0.925,0.921,0.922,0.924,0.924,0.921,0.929,0.922,0.923,0.92,0.922,0.922,0.926,0.918,0.924,0.924,0.919,0.928,0.922,0.919,0.912,0.924,0.921,0.921,0.919,0.923,0.92,0.924,0.921,0.923,0.926,0.921,0.924,0.926,0.923,0.919,0.923,0.922,0.923,0.923,0.924,0.924,0.92,0.92,0.922,0.923,0.919,0.924,0.923,0.922,0.921,0.921,0.919,0.922,0.924,0.921,0.923,0.923,0.918,0.92,0.923,0.921,0.921,0.922,0.918,0.917,0.919,0.925,0.924,0.924,0.922,0.92,0.919,0.921,0.917,0.919,0.921,0.92,0.918,0.925,0.917],
            'val' :   [0.805,0.831,0.847,0.864,0.876,0.865,0.882,0.88,0.887,0.888,0.884,0.889,0.893,0.895,0.895,0.871,0.896,0.891,0.895,0.895,0.9,0.909,0.91,0.907,0.903,0.908,0.907,0.908,0.91,0.911,0.907,0.908,0.914,0.915,0.906,0.906,0.898,0.911,0.906,0.904,0.911,0.911,0.915,0.906,0.919,0.913,0.914,0.91,0.915,0.92,0.915,0.921,0.911,0.907,0.912,0.915,0.915,0.918,0.918,0.91,0.915,0.915,0.917,0.911,0.921,0.916,0.915,0.908,0.902,0.915,0.915,0.917,0.913,0.916,0.91,0.911,0.91,0.908,0.912,0.91,0.911,0.904,0.915,0.909,0.916,0.906,0.913,0.917,0.917,0.915,0.913,0.908,0.913,0.918,0.912,0.916,0.918,0.918,0.906,0.918,0.907,0.911,0.913,0.913,0.906,0.916,0.915,0.912,0.911,0.903,0.917,0.912,0.917,0.913,0.913,0.916,0.908,0.912,0.907,0.917,0.917,0.915,0.916,0.909,0.914,0.909,0.913,0.914,0.912,0.911,0.911,0.917,0.911,0.91,0.91,0.91,0.909,0.905,0.909,0.912,0.913,0.914,0.916,0.917,0.906,0.908,0.914,0.91,0.913,0.916,0.914,0.916,0.912,0.911,0.911,0.91,0.914,0.91,0.918,0.913,0.914,0.909,0.917,0.909,0.911,0.912,0.908,0.913,0.912,0.913,0.915,0.911,0.913,0.909,0.916,0.911,0.914,0.906,0.913,0.909,0.91,0.914,0.911,0.905,0.909,0.914,0.914,0.915,0.909,0.913,0.91,0.916,0.909,0.915,0.914,0.914,0.912,0.913,0.912,0.908,0.915,0.91,0.914,0.91,0.91,0.914,0.91,0.914,0.914,0.913,0.911,0.916,0.914,0.916,0.915,0.916,0.913,0.911,0.917,0.91,0.915,0.914,0.914,0.91,0.915,0.911,0.914,0.915,0.914,0.909,0.906,0.913,0.913,0.917,0.91,0.908,0.909,0.91,0.905,0.916,0.912,0.911,0.91,0.915,0.912]
        },
        {
            'n_graph_layers' : 3,
            'n_FC_layers' : 3,
            'embedding_size' : 32,
            'epoch' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250],
            'train' : [0.772,0.835,0.844,0.86,0.87,0.875,0.881,0.891,0.899,0.891,0.9,0.901,0.905,0.907,0.915,0.916,0.917,0.914,0.921,0.921,0.918,0.92,0.92,0.921,0.931,0.931,0.929,0.931,0.929,0.937,0.934,0.934,0.939,0.942,0.94,0.931,0.943,0.94,0.94,0.944,0.946,0.945,0.936,0.945,0.948,0.944,0.95,0.95,0.938,0.945,0.951,0.951,0.947,0.949,0.953,0.947,0.955,0.952,0.954,0.941,0.95,0.955,0.953,0.956,0.952,0.954,0.958,0.958,0.957,0.959,0.951,0.956,0.955,0.958,0.961,0.959,0.962,0.96,0.96,0.957,0.963,0.96,0.964,0.963,0.962,0.963,0.962,0.964,0.963,0.953,0.964,0.966,0.962,0.968,0.966,0.966,0.966,0.965,0.957,0.967,0.965,0.966,0.968,0.967,0.968,0.964,0.963,0.955,0.973,0.965,0.97,0.968,0.968,0.972,0.968,0.971,0.972,0.972,0.97,0.972,0.971,0.972,0.971,0.961,0.973,0.972,0.971,0.973,0.973,0.975,0.972,0.971,0.974,0.972,0.976,0.974,0.974,0.972,0.967,0.955,0.968,0.971,0.978,0.973,0.975,0.975,0.965,0.976,0.974,0.974,0.974,0.973,0.974,0.976,0.976,0.978,0.974,0.978,0.976,0.979,0.976,0.976,0.975,0.977,0.979,0.977,0.97,0.976,0.977,0.978,0.976,0.974,0.979,0.979,0.98,0.979,0.975,0.978,0.979,0.973,0.979,0.972,0.979,0.964,0.975,0.982,0.98,0.978,0.981,0.981,0.979,0.98,0.979,0.982,0.979,0.971,0.98,0.975,0.978,0.982,0.982,0.979,0.98,0.982,0.977,0.981,0.978,0.983,0.98,0.98,0.982,0.978,0.98,0.976,0.983,0.979,0.982,0.981,0.981,0.981,0.982,0.977,0.979,0.983,0.974,0.983,0.984,0.983,0.981,0.982,0.981,0.98,0.983,0.984,0.982,0.984,0.98,0.983,0.983,0.982,0.984,0.984,0.981,0.983,0.983,0.984,0.985,0.983,0.984,0.982],
            'test' :  [0.792,0.856,0.862,0.864,0.879,0.884,0.888,0.897,0.902,0.897,0.902,0.906,0.905,0.906,0.911,0.911,0.91,0.91,0.911,0.914,0.909,0.907,0.909,0.908,0.914,0.919,0.921,0.919,0.919,0.924,0.924,0.923,0.924,0.925,0.925,0.922,0.925,0.928,0.922,0.927,0.929,0.926,0.919,0.926,0.927,0.927,0.93,0.93,0.918,0.922,0.929,0.931,0.924,0.927,0.927,0.925,0.933,0.928,0.928,0.92,0.929,0.93,0.929,0.927,0.926,0.923,0.928,0.929,0.93,0.928,0.924,0.927,0.925,0.925,0.93,0.926,0.93,0.93,0.931,0.924,0.928,0.926,0.931,0.928,0.928,0.928,0.927,0.927,0.928,0.918,0.927,0.931,0.928,0.931,0.928,0.93,0.929,0.925,0.919,0.928,0.928,0.928,0.93,0.924,0.926,0.923,0.922,0.92,0.928,0.925,0.929,0.928,0.927,0.929,0.927,0.931,0.929,0.929,0.928,0.928,0.926,0.928,0.927,0.922,0.929,0.927,0.927,0.927,0.926,0.929,0.925,0.925,0.928,0.924,0.927,0.927,0.928,0.928,0.923,0.911,0.922,0.923,0.933,0.924,0.927,0.927,0.918,0.927,0.928,0.926,0.927,0.923,0.925,0.927,0.925,0.927,0.925,0.927,0.926,0.927,0.926,0.927,0.924,0.925,0.926,0.924,0.919,0.926,0.924,0.927,0.927,0.927,0.928,0.926,0.923,0.926,0.923,0.927,0.925,0.924,0.924,0.918,0.924,0.916,0.923,0.926,0.925,0.923,0.927,0.928,0.926,0.926,0.924,0.924,0.925,0.916,0.926,0.921,0.922,0.925,0.927,0.925,0.926,0.926,0.925,0.926,0.926,0.927,0.922,0.925,0.923,0.921,0.926,0.922,0.927,0.923,0.928,0.923,0.924,0.927,0.927,0.923,0.925,0.928,0.921,0.926,0.929,0.928,0.927,0.927,0.923,0.923,0.924,0.928,0.927,0.924,0.926,0.925,0.923,0.923,0.927,0.927,0.924,0.923,0.922,0.927,0.924,0.927,0.928,0.924],
            'val' :   [0.769,0.833,0.836,0.847,0.861,0.866,0.867,0.873,0.885,0.878,0.889,0.89,0.892,0.889,0.892,0.893,0.894,0.893,0.893,0.896,0.894,0.89,0.892,0.891,0.903,0.903,0.905,0.903,0.907,0.911,0.909,0.903,0.911,0.91,0.91,0.907,0.909,0.914,0.909,0.912,0.914,0.912,0.904,0.912,0.913,0.912,0.916,0.917,0.905,0.907,0.917,0.918,0.913,0.915,0.915,0.913,0.92,0.918,0.916,0.908,0.917,0.915,0.914,0.916,0.915,0.909,0.918,0.921,0.921,0.916,0.907,0.915,0.91,0.908,0.916,0.914,0.918,0.917,0.917,0.908,0.917,0.915,0.92,0.914,0.915,0.917,0.916,0.921,0.916,0.9,0.912,0.917,0.913,0.919,0.917,0.917,0.915,0.915,0.909,0.917,0.914,0.918,0.918,0.912,0.914,0.913,0.91,0.904,0.919,0.917,0.917,0.917,0.914,0.915,0.916,0.918,0.918,0.915,0.92,0.916,0.913,0.917,0.909,0.91,0.914,0.917,0.913,0.916,0.916,0.915,0.914,0.913,0.92,0.916,0.919,0.915,0.917,0.918,0.913,0.894,0.914,0.91,0.921,0.915,0.919,0.915,0.905,0.915,0.915,0.917,0.917,0.911,0.915,0.916,0.916,0.918,0.919,0.918,0.916,0.917,0.914,0.919,0.912,0.916,0.916,0.912,0.902,0.915,0.917,0.918,0.915,0.914,0.917,0.922,0.916,0.922,0.912,0.915,0.916,0.913,0.914,0.908,0.913,0.904,0.914,0.919,0.916,0.909,0.913,0.919,0.914,0.918,0.917,0.916,0.914,0.91,0.918,0.912,0.913,0.919,0.918,0.913,0.918,0.914,0.914,0.915,0.915,0.918,0.914,0.911,0.915,0.912,0.914,0.91,0.918,0.913,0.918,0.913,0.913,0.915,0.919,0.909,0.914,0.912,0.909,0.913,0.916,0.916,0.916,0.918,0.914,0.914,0.913,0.915,0.914,0.918,0.918,0.913,0.914,0.915,0.915,0.916,0.917,0.916,0.915,0.915,0.914,0.914,0.919,0.914]
        },
        {
            'n_graph_layers': 3,
            'n_FC_layers': 2,
            'embedding_size': 32,
            'epoch': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250],
            'train': [0.824, 0.838, 0.852, 0.859, 0.838, 0.878, 0.882, 0.879, 0.886, 0.9, 0.892, 0.889, 0.906, 0.905, 0.902, 0.914, 0.911, 0.909, 0.919, 0.916, 0.924, 0.919, 0.919, 0.897, 0.924, 0.931, 0.925, 0.935, 0.935, 0.923, 0.932, 0.936, 0.938, 0.941, 0.927, 0.934, 0.94, 0.933, 0.937, 0.939, 0.94, 0.941, 0.943, 0.944, 0.943, 0.94, 0.946, 0.942, 0.943, 0.946, 0.947, 0.947, 0.931, 0.946, 0.944, 0.95, 0.95, 0.945, 0.951, 0.949, 0.946, 0.952, 0.953, 0.952, 0.954, 0.952, 0.942, 0.951, 0.958, 0.961, 0.957, 0.956, 0.951, 0.954, 0.96, 0.958, 0.961, 0.954, 0.962, 0.955, 0.961, 0.957, 0.944, 0.958, 0.965, 0.961, 0.955, 0.961, 0.964, 0.966, 0.962, 0.958, 0.967, 0.958, 0.961, 0.959, 0.962, 0.96, 0.962, 0.966, 0.96, 0.964, 0.961, 0.967, 0.963, 0.968, 0.967, 0.961, 0.966, 0.965, 0.967, 0.963, 0.967, 0.969, 0.964, 0.971, 0.966, 0.967, 0.957, 0.968, 0.97, 0.965, 0.97, 0.965, 0.972, 0.972, 0.963, 0.969, 0.971, 0.973, 0.971, 0.968, 0.97, 0.972, 0.969, 0.969, 0.966, 0.97, 0.97, 0.962, 0.97, 0.975, 0.971, 0.974, 0.973, 0.969, 0.974, 0.975, 0.974, 0.975, 0.976, 0.972, 0.977, 0.974, 0.974, 0.964, 0.976, 0.97, 0.975, 0.978, 0.976, 0.974, 0.975, 0.972, 0.971, 0.978, 0.977, 0.976, 0.978, 0.978, 0.979, 0.969, 0.971, 0.977, 0.976, 0.972, 0.974, 0.971, 0.98, 0.976, 0.977, 0.978, 0.977, 0.979, 0.97, 0.978, 0.975, 0.978, 0.977, 0.973, 0.979, 0.979, 0.979, 0.979, 0.98, 0.978, 0.978, 0.981, 0.978, 0.978, 0.977, 0.97, 0.976, 0.98, 0.98, 0.983, 0.98, 0.974, 0.978, 0.982, 0.977, 0.976, 0.983, 0.983, 0.981, 0.977, 0.977, 0.981, 0.974, 0.974, 0.984, 0.981, 0.978, 0.98, 0.981, 0.982, 0.98, 0.983, 0.983, 0.977, 0.979, 0.981, 0.981, 0.983, 0.982, 0.977, 0.982, 0.982, 0.982, 0.983, 0.985, 0.982, 0.985, 0.983, 0.981, 0.982, 0.983, 0.982, 0.983, 0.985],
            'test': [0.847, 0.859, 0.869, 0.874, 0.853, 0.889, 0.89, 0.88, 0.89, 0.899, 0.891, 0.885, 0.906, 0.904, 0.902, 0.908, 0.911, 0.909, 0.914, 0.909, 0.915, 0.913, 0.912, 0.898, 0.916, 0.92, 0.919, 0.923, 0.922, 0.908, 0.923, 0.922, 0.923, 0.923, 0.914, 0.922, 0.926, 0.918, 0.924, 0.926, 0.925, 0.925, 0.925, 0.925, 0.926, 0.926, 0.929, 0.925, 0.925, 0.923, 0.926, 0.931, 0.914, 0.925, 0.926, 0.925, 0.929, 0.923, 0.927, 0.931, 0.926, 0.925, 0.93, 0.931, 0.929, 0.932, 0.917, 0.925, 0.931, 0.933, 0.929, 0.929, 0.921, 0.928, 0.932, 0.93, 0.93, 0.93, 0.934, 0.928, 0.931, 0.929, 0.92, 0.931, 0.935, 0.929, 0.924, 0.931, 0.934, 0.935, 0.934, 0.926, 0.935, 0.928, 0.929, 0.925, 0.93, 0.927, 0.931, 0.933, 0.929, 0.932, 0.928, 0.932, 0.925, 0.933, 0.93, 0.926, 0.931, 0.932, 0.93, 0.926, 0.931, 0.929, 0.925, 0.935, 0.928, 0.926, 0.922, 0.929, 0.932, 0.924, 0.932, 0.929, 0.934, 0.931, 0.923, 0.93, 0.935, 0.932, 0.93, 0.929, 0.932, 0.934, 0.932, 0.931, 0.929, 0.931, 0.93, 0.926, 0.932, 0.932, 0.933, 0.933, 0.932, 0.927, 0.934, 0.931, 0.934, 0.933, 0.932, 0.93, 0.934, 0.931, 0.933, 0.921, 0.931, 0.929, 0.932, 0.932, 0.929, 0.931, 0.93, 0.93, 0.93, 0.932, 0.935, 0.933, 0.933, 0.931, 0.931, 0.925, 0.926, 0.932, 0.93, 0.931, 0.929, 0.923, 0.93, 0.928, 0.932, 0.932, 0.932, 0.935, 0.928, 0.93, 0.929, 0.931, 0.932, 0.926, 0.933, 0.931, 0.93, 0.93, 0.93, 0.931, 0.929, 0.932, 0.929, 0.93, 0.931, 0.925, 0.929, 0.933, 0.931, 0.932, 0.931, 0.927, 0.93, 0.931, 0.927, 0.927, 0.932, 0.933, 0.932, 0.927, 0.928, 0.927, 0.927, 0.929, 0.933, 0.931, 0.93, 0.931, 0.932, 0.932, 0.93, 0.932, 0.93, 0.923, 0.926, 0.926, 0.93, 0.93, 0.929, 0.924, 0.932, 0.932, 0.932, 0.931, 0.933, 0.93, 0.934, 0.931, 0.929, 0.931, 0.932, 0.929, 0.93, 0.932],
            'val': [0.821, 0.831, 0.845, 0.846, 0.818, 0.867, 0.865, 0.853, 0.865, 0.88, 0.872, 0.865, 0.884, 0.883, 0.884, 0.891, 0.893, 0.891, 0.894, 0.89, 0.9, 0.895, 0.895, 0.881, 0.902, 0.906, 0.906, 0.91, 0.908, 0.898, 0.908, 0.909, 0.911, 0.911, 0.903, 0.907, 0.914, 0.902, 0.908, 0.915, 0.911, 0.912, 0.914, 0.909, 0.912, 0.908, 0.915, 0.915, 0.914, 0.914, 0.91, 0.917, 0.898, 0.912, 0.908, 0.919, 0.915, 0.909, 0.915, 0.914, 0.916, 0.916, 0.918, 0.917, 0.917, 0.914, 0.903, 0.917, 0.92, 0.921, 0.914, 0.915, 0.91, 0.912, 0.921, 0.916, 0.919, 0.917, 0.921, 0.913, 0.915, 0.914, 0.904, 0.914, 0.919, 0.918, 0.908, 0.914, 0.92, 0.92, 0.918, 0.915, 0.923, 0.919, 0.917, 0.911, 0.916, 0.913, 0.918, 0.92, 0.916, 0.921, 0.913, 0.919, 0.915, 0.918, 0.921, 0.913, 0.918, 0.917, 0.912, 0.912, 0.912, 0.916, 0.915, 0.92, 0.917, 0.913, 0.904, 0.916, 0.918, 0.911, 0.917, 0.911, 0.922, 0.918, 0.911, 0.918, 0.919, 0.918, 0.916, 0.916, 0.918, 0.92, 0.916, 0.917, 0.917, 0.919, 0.915, 0.908, 0.913, 0.918, 0.921, 0.919, 0.92, 0.911, 0.919, 0.917, 0.916, 0.917, 0.921, 0.912, 0.918, 0.916, 0.919, 0.903, 0.917, 0.914, 0.918, 0.921, 0.919, 0.918, 0.915, 0.917, 0.915, 0.916, 0.92, 0.92, 0.92, 0.919, 0.917, 0.907, 0.911, 0.916, 0.914, 0.915, 0.917, 0.909, 0.919, 0.918, 0.918, 0.923, 0.918, 0.916, 0.913, 0.919, 0.914, 0.917, 0.915, 0.911, 0.92, 0.915, 0.916, 0.917, 0.917, 0.916, 0.916, 0.917, 0.913, 0.916, 0.916, 0.912, 0.912, 0.918, 0.918, 0.921, 0.919, 0.908, 0.918, 0.917, 0.912, 0.909, 0.921, 0.918, 0.917, 0.913, 0.909, 0.914, 0.912, 0.917, 0.921, 0.917, 0.916, 0.915, 0.916, 0.917, 0.917, 0.918, 0.919, 0.91, 0.912, 0.913, 0.918, 0.916, 0.916, 0.907, 0.915, 0.917, 0.917, 0.92, 0.92, 0.917, 0.92, 0.919, 0.916, 0.917, 0.917, 0.916, 0.915, 0.917]
        },
        {
            'n_graph_layers' : 2,
            'n_FC_layers' : 2,
            'embedding_size' : 32,
            'epoch' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250],
            'train' : [0.823,0.84,0.835,0.826,0.797,0.863,0.825,0.834,0.879,0.877,0.889,0.873,0.893,0.852,0.893,0.898,0.897,0.903,0.908,0.907,0.911,0.9,0.908,0.906,0.914,0.914,0.912,0.918,0.917,0.913,0.909,0.911,0.922,0.92,0.928,0.927,0.928,0.922,0.924,0.928,0.928,0.929,0.93,0.925,0.932,0.935,0.924,0.933,0.935,0.937,0.924,0.934,0.93,0.937,0.93,0.938,0.937,0.934,0.928,0.937,0.942,0.941,0.931,0.946,0.941,0.942,0.94,0.936,0.937,0.943,0.941,0.941,0.936,0.942,0.948,0.949,0.943,0.944,0.95,0.945,0.947,0.949,0.954,0.951,0.954,0.949,0.948,0.955,0.948,0.95,0.955,0.948,0.952,0.955,0.954,0.953,0.952,0.957,0.955,0.955,0.948,0.947,0.956,0.957,0.95,0.96,0.957,0.96,0.959,0.96,0.962,0.956,0.96,0.96,0.95,0.964,0.96,0.963,0.958,0.962,0.962,0.964,0.96,0.962,0.965,0.96,0.963,0.963,0.964,0.961,0.964,0.966,0.965,0.961,0.964,0.963,0.966,0.966,0.968,0.966,0.965,0.964,0.963,0.965,0.966,0.966,0.965,0.966,0.967,0.964,0.968,0.962,0.964,0.968,0.964,0.971,0.968,0.97,0.966,0.968,0.971,0.969,0.969,0.971,0.968,0.973,0.969,0.971,0.971,0.972,0.973,0.972,0.973,0.972,0.965,0.966,0.969,0.969,0.973,0.973,0.972,0.974,0.973,0.975,0.97,0.974,0.975,0.971,0.975,0.972,0.975,0.974,0.972,0.971,0.973,0.973,0.972,0.975,0.972,0.976,0.975,0.974,0.973,0.976,0.973,0.977,0.974,0.974,0.974,0.976,0.974,0.974,0.976,0.978,0.977,0.977,0.975,0.976,0.977,0.976,0.978,0.979,0.978,0.978,0.978,0.977,0.978,0.978,0.977,0.979,0.979,0.978,0.977,0.98,0.979,0.975,0.976,0.979,0.98,0.979,0.977,0.978,0.979,0.979,0.973,0.981,0.98,0.976,0.98,0.977],
            'test' :  [0.841,0.853,0.851,0.839,0.818,0.872,0.844,0.845,0.885,0.879,0.89,0.879,0.893,0.866,0.893,0.892,0.891,0.9,0.9,0.901,0.906,0.893,0.898,0.898,0.905,0.907,0.901,0.908,0.905,0.896,0.894,0.889,0.903,0.901,0.912,0.91,0.913,0.905,0.907,0.907,0.91,0.912,0.909,0.906,0.912,0.912,0.909,0.912,0.913,0.911,0.902,0.91,0.904,0.913,0.908,0.915,0.911,0.91,0.909,0.909,0.914,0.911,0.905,0.916,0.917,0.912,0.915,0.907,0.91,0.911,0.915,0.912,0.906,0.911,0.915,0.914,0.913,0.914,0.913,0.914,0.913,0.916,0.915,0.913,0.914,0.914,0.91,0.918,0.916,0.916,0.915,0.912,0.917,0.917,0.915,0.915,0.914,0.917,0.915,0.917,0.91,0.91,0.916,0.912,0.91,0.918,0.915,0.918,0.917,0.915,0.92,0.914,0.918,0.916,0.911,0.919,0.917,0.92,0.912,0.916,0.92,0.918,0.915,0.915,0.918,0.911,0.918,0.915,0.917,0.915,0.919,0.918,0.916,0.914,0.916,0.917,0.92,0.917,0.917,0.918,0.916,0.916,0.913,0.916,0.916,0.916,0.917,0.912,0.916,0.916,0.918,0.911,0.919,0.916,0.915,0.921,0.915,0.918,0.92,0.919,0.917,0.915,0.919,0.919,0.916,0.921,0.916,0.92,0.918,0.918,0.92,0.917,0.919,0.917,0.914,0.915,0.914,0.918,0.918,0.915,0.918,0.921,0.917,0.92,0.918,0.918,0.919,0.916,0.919,0.919,0.917,0.92,0.914,0.914,0.916,0.92,0.916,0.92,0.914,0.921,0.92,0.919,0.917,0.92,0.918,0.921,0.918,0.916,0.918,0.918,0.915,0.916,0.919,0.917,0.922,0.917,0.915,0.92,0.915,0.917,0.919,0.918,0.915,0.916,0.916,0.919,0.918,0.919,0.918,0.92,0.919,0.92,0.918,0.919,0.919,0.915,0.918,0.919,0.923,0.919,0.915,0.918,0.919,0.919,0.916,0.917,0.921,0.911,0.916,0.915],
            'val' :   [0.806,0.817,0.809,0.802,0.787,0.836,0.812,0.819,0.851,0.85,0.858,0.841,0.865,0.842,0.863,0.87,0.862,0.875,0.878,0.879,0.879,0.87,0.88,0.875,0.885,0.885,0.882,0.889,0.884,0.876,0.871,0.869,0.881,0.882,0.892,0.89,0.894,0.886,0.889,0.889,0.892,0.893,0.891,0.891,0.895,0.892,0.887,0.894,0.892,0.886,0.878,0.887,0.887,0.894,0.89,0.896,0.89,0.892,0.894,0.892,0.898,0.896,0.884,0.897,0.898,0.896,0.896,0.889,0.888,0.896,0.896,0.894,0.889,0.896,0.9,0.893,0.895,0.9,0.898,0.899,0.896,0.897,0.899,0.898,0.897,0.901,0.899,0.903,0.895,0.899,0.901,0.895,0.901,0.899,0.895,0.9,0.901,0.899,0.897,0.9,0.89,0.892,0.899,0.898,0.896,0.902,0.898,0.898,0.895,0.899,0.902,0.902,0.896,0.897,0.891,0.901,0.899,0.904,0.895,0.896,0.901,0.902,0.898,0.901,0.901,0.899,0.901,0.9,0.901,0.898,0.898,0.903,0.897,0.891,0.9,0.897,0.898,0.898,0.897,0.9,0.9,0.898,0.895,0.897,0.9,0.899,0.899,0.897,0.902,0.896,0.898,0.893,0.9,0.9,0.9,0.901,0.9,0.902,0.899,0.897,0.899,0.898,0.903,0.9,0.897,0.902,0.898,0.9,0.901,0.902,0.903,0.903,0.903,0.898,0.899,0.895,0.892,0.9,0.904,0.896,0.902,0.903,0.9,0.902,0.904,0.9,0.903,0.904,0.9,0.902,0.901,0.906,0.896,0.898,0.9,0.902,0.896,0.899,0.891,0.902,0.901,0.903,0.901,0.903,0.904,0.904,0.899,0.898,0.899,0.897,0.897,0.899,0.901,0.902,0.907,0.902,0.899,0.904,0.897,0.899,0.902,0.902,0.902,0.9,0.897,0.898,0.901,0.9,0.902,0.904,0.899,0.899,0.899,0.901,0.9,0.901,0.903,0.9,0.906,0.904,0.894,0.902,0.902,0.899,0.897,0.9,0.902,0.892,0.903,0.897]
        }
    ]

    data2 = [
        {
            'n_graph_layers': 3,
            'n_FC_layers': 2,
            'embedding_size': 128,
            'epoch': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126],
            'train': [0.828,0.853,0.869,0.858,0.883,0.882,0.889,0.877,0.898,0.897,0.895,0.896,0.902,0.915,0.912,0.885,0.868,0.895,0.922,0.927,0.921,0.933,0.929,0.932,0.937,0.937,0.925,0.937,0.894,0.937,0.944,0.926,0.945,0.944,0.943,0.94,0.949,0.945,0.943,0.893,0.95,0.946,0.953,0.947,0.956,0.956,0.951,0.953,0.95,0.956,0.954,0.953,0.959,0.954,0.958,0.944,0.959,0.96,0.952,0.956,0.959,0.964,0.958,0.961,0.962,0.964,0.963,0.961,0.963,0.965,0.966,0.954,0.965,0.955,0.967,0.964,0.961,0.96,0.968,0.966,0.968,0.964,0.965,0.967,0.971,0.959,0.969,0.97,0.965,0.965,0.963,0.968,0.972,0.97,0.968,0.973,0.967,0.969,0.975,0.973,0.973,0.962,0.974,0.971,0.971,0.973,0.974,0.969,0.978,0.972,0.976,0.961,0.971,0.956,0.979,0.977,0.978,0.974,0.977,0.98,0.979,0.978,0.972,0.98,0.981,0.978],
            'test':  [0.85,0.867,0.883,0.868,0.892,0.889,0.895,0.888,0.903,0.903,0.9,0.903,0.905,0.916,0.912,0.895,0.878,0.905,0.922,0.929,0.918,0.93,0.928,0.931,0.935,0.932,0.922,0.93,0.898,0.927,0.934,0.924,0.935,0.934,0.934,0.929,0.938,0.934,0.932,0.89,0.933,0.931,0.94,0.934,0.94,0.939,0.938,0.938,0.931,0.937,0.937,0.938,0.939,0.932,0.938,0.927,0.933,0.939,0.933,0.933,0.937,0.94,0.938,0.938,0.94,0.937,0.939,0.936,0.936,0.939,0.938,0.932,0.937,0.936,0.939,0.939,0.937,0.935,0.939,0.934,0.938,0.937,0.938,0.937,0.938,0.929,0.936,0.939,0.934,0.934,0.935,0.938,0.94,0.938,0.937,0.938,0.935,0.936,0.938,0.938,0.938,0.927,0.936,0.937,0.937,0.941,0.936,0.935,0.94,0.936,0.933,0.92,0.93,0.919,0.939,0.938,0.94,0.934,0.935,0.938,0.937,0.937,0.933,0.937,0.94,0.936],
            'val':   [0.811,0.836,0.856,0.849,0.871,0.869,0.876,0.87,0.889,0.885,0.868,0.881,0.894,0.899,0.895,0.875,0.854,0.89,0.913,0.914,0.9,0.919,0.916,0.914,0.92,0.918,0.915,0.916,0.873,0.913,0.922,0.913,0.923,0.921,0.922,0.918,0.928,0.921,0.921,0.875,0.927,0.924,0.928,0.922,0.928,0.928,0.927,0.926,0.923,0.931,0.926,0.925,0.932,0.925,0.929,0.919,0.928,0.929,0.925,0.928,0.93,0.929,0.93,0.926,0.933,0.926,0.927,0.923,0.926,0.931,0.928,0.921,0.929,0.927,0.929,0.928,0.926,0.924,0.929,0.927,0.929,0.924,0.923,0.929,0.928,0.921,0.929,0.928,0.925,0.926,0.923,0.925,0.93,0.927,0.922,0.93,0.924,0.928,0.927,0.927,0.925,0.917,0.928,0.926,0.926,0.929,0.923,0.921,0.928,0.926,0.925,0.905,0.918,0.907,0.924,0.928,0.928,0.925,0.925,0.925,0.927,0.924,0.923,0.928,0.929,0.925]
        }
    ]

    def plot(dict):

        plt.rcParams.update({
            'lines.markersize': 3,  # Dot size
            'axes.grid': True,
            'axes.axisbelow': True,
            'grid.color': 'gainsboro'
        })

        dict['train'] = [i * 100 for i in dict['train']]
        dict['test'] =  [i * 100 for i in dict['test']]
        dict['val'] =   [i * 100 for i in dict['val']]

        plt.clf()
        plt.scatter(dict['epoch'], dict['train'], color='blue')
        plt.scatter(dict['epoch'], dict['val'], color='green')
        plt.scatter(dict['epoch'], dict['test'], color='darkorange')
        plt.ylim(70, 101)
        plt.xlim(-5, 131)
        plt.title("reduced hybridization" +
                  "\nn_graph_layers " + str(dict['n_graph_layers']) +
                  "\nn_FC_layers " + str(dict['n_FC_layers']) +
                  "\nembedding_size " + str(dict['embedding_size']))
        plt.xlabel("epoch")
        plt.ylabel("Accuracy (%)")
        plt.tight_layout()

        plt.savefig('GNN_accuracy_' +
                    str(dict['embedding_size']) + '_' +
                    str(dict['n_graph_layers']) + '_' +
                    str(dict['n_FC_layers']) + '.png')
    #plot(data[7])
    for dict in data2:
        plot(dict)

    plt.rcParams.update({
        'axes.titlesize': 10,
        'lines.markersize': 1,  # Dot size
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': 'gainsboro',
    })
    plt.clf()
    def subplot(ax, dict):

        dict['train'] = [i * 100 for i in dict['train']]
        dict['test'] =  [i * 100 for i in dict['test']]
        dict['val'] =   [i * 100 for i in dict['val']]

        ax.scatter(dict['epoch'], dict['train'], color='blue')
        ax.scatter(dict['epoch'], dict['val'], color='green')
        ax.scatter(dict['epoch'], dict['test'], color='darkorange')
        ax.set_ylim(70, 101)
        ax.set_title("n_graph_layers " + str(dict['n_graph_layers']) +
                     "\nn_FC_layers " + str(dict['n_FC_layers']) +
                     "\nembedding_size " + str(dict['embedding_size']))
        #ax.xlabel("epoch")
        #ax.ylabel("Accuracy (%)")

    fig, axs = plt.subplots(nrows=5, ncols=3, sharex=False, sharey=True, figsize=(8.5, 11))
    for i, ax in enumerate(axs.flatten()):
        subplot(ax, data[i])

    # for ax in axs.flat:
    #     ax.set(xlabel='epoch', ylabel='Accuracy (%)')
    # for ax in fig.get_axes():
    #     ax.label_outer()
    
    # Ensure x-tick values are shown on all subplots
    for ax in axs.flat:
        ax.tick_params(axis='x', labelbottom=True)
    for ax in axs[:, 0]:  # Only the first column of subplots
        ax.set_ylabel('Accuracy (%)')
    # Set the xlabel only on the bottom row subplots
    for ax in axs[-1, :]:  # Only the last row of subplots
        ax.set_xlabel('epoch')

    # add legend
    labels = ['Train set', 'Validation set', 'Test set']
    fig.legend(labels=labels, loc='lower center', bbox_to_anchor=(0.5, 0.04), ncol=3, frameon=False, bbox_transform=fig.transFigure, fontsize=11, markerscale=10)

    plt.tight_layout(rect=[0, 0.07, 1, 1])

    #plt.tight_layout()

    fig.savefig('GNN_accuracy_subplots.png')


def plot_GNN_prediction(args):

    # python3 main.py --mode plotstuff --list_trainset list_trainset_clean.txt > GNN_accuracy.out

    set = []  # What data is used to plot the graph (either test, train or both)

    if os.path.isfile('PDBBind_GNN_pred_10/' + args.list_trainset) is True:
        print('| Set selected from %-86s |' % args.list_trainset)
        with open('PDBBind_GNN_pred_10/' + args.list_trainset, 'rb') as list_f:
            for line in list_f:
                set.append(line.strip().decode('UTF-8'))

    accuracies = pd.DataFrame(columns=['RMSD threshold', 'Lowest RMSD', 'Random', 'Predicted'])
    accuracies['RMSD threshold'] = np.arange(0, 5.25, 0.25)
    accuracies['Lowest RMSD'] = 0
    accuracies['Random'] = 0
    accuracies['Predicted'] = 0

    total = 0

    for pdb_id in set:

        input_file = "Predict_" + pdb_id + ".out"

        with open('PDBBind_GNN_pred_10/' + input_file, "r") as f:
            lines = f.readlines()

        PDBdata = pd.DataFrame(columns=['RMSD', 'Label_ML'])
        inside_table = False
        counter = 0

        for line in lines:


            if "| Pose # | score    | RMSD" in line:  # Detect table start
                inside_table = True
                total += 1
                with open("PDBs_that_ran.txt", "a") as file:
                    file.write(pdb_id + "\n")

                continue
            # if "|-------------------------------------------------------------------------------------------------------------------|" in line:  # Detect table end
            #     if counter == 10:
            #         inside_table = False
            #         continue

            if inside_table:
                if counter == 11:
                    inside_table = False
                    #print(line)
                    print(PDBdata)
                    random = PDBdata.sample()
                    for index, value in accuracies['RMSD threshold'].items():

                        # Lowest RMSD
                        if PDBdata['RMSD'].min() <= value:
                            accuracies.at[index, 'Lowest RMSD'] += 1

                        # Random
                        if random['RMSD'].iloc[0] <= value:
                            accuracies.at[index, 'Random'] += 1

                        # Predicted
                        if PDBdata.loc[PDBdata['Label_ML'].idxmax(), 'RMSD'] <= value:  # For the GNN, the max prediction indicated the best pose
                            accuracies.at[index, 'Predicted'] += 1
                    #print(line)
                elif counter == 0:
                    counter += 1
                else:
                    #print(line)
                    counter += 1
                    parts = [x.strip() for x in line.split("|") if x.strip()]
                    if len(parts) >= 3:  # Ensure it's a valid row
                        score = parts[1]
                        rmsd = parts[2]
                        PDBdata.loc[len(PDBdata)] = [float(rmsd), float(score)]



    f.close()

    accuracies['Lowest RMSD'] = (accuracies['Lowest RMSD'] / total) * 100
    accuracies['Random'] = (accuracies['Random'] / total) * 100
    accuracies['Predicted'] = (accuracies['Predicted'] / total) * 100

    print('total PDBs: ' + str(total) + '/' + str(len(set)))
    print('Accuracies %:')
    print(accuracies)

    # plt.plot(accuracies['RMSD threshold'], accuracies['Lowest RMSD'], label="Lowest RMSD", color='red')
    # plt.plot(accuracies['RMSD threshold'], accuracies['Random'], label="Random", color='orange')
    # plt.plot(accuracies['RMSD threshold'], accuracies['Predicted'], label="Predicted", color='green')

    # ax = plt.gca()
    #
    # font1 = {'color': 'black', 'size': 14}
    # font2 = {'color': 'black', 'size': 12}
    #
    # ax.tick_params(axis='both', which='major', labelsize=12)
    # ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    #
    # # Set the x axis label of the current axis.
    # plt.xlabel('RMSD (Å)', fontdict=font1)
    # # Set the y axis label of the current axis.
    # plt.ylabel('Accuracy (%)', fontdict=font1)
    # # show a legend on the plot
    # plt.legend(prop={'size': 12})
    # # show gridlines
    # plt.grid()
    # # Set a title of the current axes.
    # plt.title(args.model + ' ' + args.set, fontdict=font1)
    # # Add extra white space at the bottom so that the label is not cut off
    # plt.subplots_adjust(bottom=0.13)
    #
    # plt.savefig(cwd + '/graphs/' + args.model + '_' + args.set + '.png')

def plot_GNN_accuracies_2x1():

    data2x1 = [
        {
            'title': '\nembedding_size 128\nn_graph_layers 3\nn_FC_layers 2',
            'description': '',
            'n_graph_layers': 3,
            'n_FC_layers': 2,
            'embedding_size': 128,
            'epoch': [    1,     2,     3,     4,     5,    6,    7,      8,      9,   10,    11,    12,   13,   14,   15,  16,   17,  18,    19,    20,    21,   22,    23,      24,     25,   26,    27,    28,    29,    30,    31,    32,   33,   34,   35,   36,    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98],
            'train': [0.797, 0.844, 0.864, 0.88, 0.881,  0.878, 0.893, 0.891, 0.901, 0.9, 0.892,   0.908, 0.902, 0.922, 0.912, 0.921, 0.901, 0.914, 0.927, 0.925, 0.924, 0.921, 0.905, 0.936, 0.934, 0.938, 0.935, 0.93, 0.908, 0.941, 0.944, 0.935, 0.94, 0.943, 0.936, 0.942, 0.945, 0.947, 0.935, 0.905, 0.947, 0.944, 0.942, 0.915, 0.953, 0.95, 0.941, 0.954, 0.953, 0.959, 0.956, 0.952, 0.96, 0.955, 0.956, 0.952, 0.952, 0.949, 0.96, 0.954, 0.943, 0.963, 0.935, 0.963, 0.96, 0.963, 0.963, 0.957, 0.96, 0.964, 0.966, 0.959, 0.966, 0.957, 0.963, 0.969, 0.957, 0.954, 0.965, 0.97, 0.972, 0.955, 0.971, 0.968, 0.971, 0.965, 0.972, 0.96, 0.966, 0.972, 0.962, 0.972, 0.969, 0.956, 0.97, 0.974, 0.976, 0.972],
            'test':  [0.818, 0.861, 0.881, 0.891, 0.892, 0.885, 0.901, 0.894, 0.905, 0.905, 0.901, 0.908, 0.906, 0.922, 0.912, 0.922, 0.901, 0.919, 0.927, 0.92, 0.92,   0.921, 0.905, 0.929, 0.93, 0.929, 0.929, 0.915, 0.903, 0.93, 0.935, 0.929, 0.932, 0.933, 0.93, 0.929, 0.932, 0.936, 0.926, 0.894, 0.933, 0.93, 0.925, 0.905, 0.936, 0.931, 0.925, 0.936, 0.934, 0.94, 0.933, 0.929, 0.938, 0.934, 0.938, 0.928, 0.931, 0.925, 0.937, 0.932, 0.92, 0.937, 0.918, 0.935, 0.933, 0.937, 0.932, 0.929, 0.936, 0.938, 0.938, 0.929, 0.935, 0.93, 0.936, 0.936, 0.925, 0.924, 0.933, 0.936, 0.939, 0.928, 0.937, 0.936, 0.938, 0.931, 0.935, 0.924, 0.928, 0.935, 0.927, 0.937, 0.934, 0.92, 0.934, 0.935, 0.938, 0.933],
            'val':   [0.785, 0.834, 0.854, 0.867, 0.868, 0.866, 0.881, 0.877, 0.883, 0.883, 0.873, 0.889, 0.884, 0.905, 0.897, 0.905, 0.879, 0.904, 0.909, 0.903, 0.901, 0.904, 0.888, 0.916, 0.915, 0.916, 0.914, 0.905, 0.876, 0.915, 0.919, 0.915, 0.917, 0.92, 0.915, 0.913, 0.922, 0.919, 0.908, 0.875, 0.918, 0.915, 0.915, 0.898, 0.922, 0.919, 0.908, 0.925, 0.923, 0.922, 0.917, 0.913, 0.923, 0.92, 0.922, 0.916, 0.917, 0.907, 0.922, 0.917, 0.905, 0.921, 0.904, 0.921, 0.917, 0.921, 0.915, 0.915, 0.916, 0.921, 0.922, 0.915, 0.923, 0.919, 0.923, 0.923, 0.914, 0.91, 0.921, 0.921, 0.925, 0.915, 0.922, 0.92, 0.921, 0.917, 0.918, 0.911, 0.913, 0.922, 0.909, 0.921, 0.919, 0.903, 0.92, 0.918, 0.924, 0.921]
        },
        {
            'title': 'Coarse featurization\nembedding_size 128\nn_graph_layers 3\nn_FC_layers 2',
            'description': 'Coarse featurization',
            'n_graph_layers': 3,
            'n_FC_layers': 2,
            'embedding_size': 128,
            'epoch': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98],
            'train': [0.828, 0.853, 0.869, 0.858, 0.883, 0.882, 0.889, 0.877, 0.898, 0.897, 0.895, 0.896, 0.902, 0.915, 0.912, 0.885, 0.868, 0.895, 0.922, 0.927, 0.921, 0.933, 0.929, 0.932, 0.937, 0.937, 0.925, 0.937, 0.894, 0.937, 0.944, 0.926, 0.945, 0.944, 0.943, 0.94, 0.949, 0.945, 0.943, 0.893, 0.95, 0.946, 0.953, 0.947, 0.956, 0.956, 0.951, 0.953, 0.95, 0.956, 0.954, 0.953, 0.959, 0.954, 0.958, 0.944, 0.959, 0.96, 0.952, 0.956, 0.959, 0.964, 0.958, 0.961, 0.962, 0.964, 0.963, 0.961, 0.963, 0.965, 0.966, 0.954, 0.965, 0.955, 0.967, 0.964, 0.961, 0.96, 0.968, 0.966, 0.968, 0.964, 0.965, 0.967, 0.971, 0.959, 0.969, 0.97, 0.965, 0.965, 0.963, 0.968, 0.972, 0.97, 0.968, 0.973, 0.967, 0.969],
            'test':  [0.85, 0.867, 0.883, 0.868, 0.892, 0.889, 0.895, 0.888, 0.903, 0.903, 0.9, 0.903, 0.905, 0.916, 0.912, 0.895, 0.878, 0.905, 0.922, 0.929, 0.918, 0.93, 0.928, 0.931, 0.935, 0.932, 0.922, 0.93, 0.898, 0.927, 0.934, 0.924, 0.935, 0.934, 0.934, 0.929, 0.938, 0.934, 0.932, 0.89, 0.933, 0.931, 0.94, 0.934, 0.94, 0.939, 0.938, 0.938, 0.931, 0.937, 0.937, 0.938, 0.939, 0.932, 0.938, 0.927, 0.933, 0.939, 0.933, 0.933, 0.937, 0.94, 0.938, 0.938, 0.94, 0.937, 0.939, 0.936, 0.936, 0.939, 0.938, 0.932, 0.937, 0.936, 0.939, 0.939, 0.937, 0.935, 0.939, 0.934, 0.938, 0.937, 0.938, 0.937, 0.938, 0.929, 0.936, 0.939, 0.934, 0.934, 0.935, 0.938, 0.94, 0.938, 0.937, 0.938, 0.935, 0.936],
            'val':   [0.811, 0.836, 0.856, 0.849, 0.871, 0.869, 0.876, 0.87, 0.889, 0.885, 0.868, 0.881, 0.894, 0.899, 0.895, 0.875, 0.854, 0.89, 0.913, 0.914, 0.9, 0.919, 0.916, 0.914, 0.92, 0.918, 0.915, 0.916, 0.873, 0.913, 0.922, 0.913, 0.923, 0.921, 0.922, 0.918, 0.928, 0.921, 0.921, 0.875, 0.927, 0.924, 0.928, 0.922, 0.928, 0.928, 0.927, 0.926, 0.923, 0.931, 0.926, 0.925, 0.932, 0.925, 0.929, 0.919, 0.928, 0.929, 0.925, 0.928, 0.93, 0.929, 0.93, 0.926, 0.933, 0.926, 0.927, 0.923, 0.926, 0.931, 0.928, 0.921, 0.929, 0.927, 0.929, 0.928, 0.926, 0.924, 0.929, 0.927, 0.929, 0.924, 0.923, 0.929, 0.928, 0.921, 0.929, 0.928, 0.925, 0.926, 0.923, 0.925, 0.93, 0.927, 0.922, 0.93, 0.924, 0.928]
        }
        # {
        #     'title': 'Coarse featurization\nembedding_size 128\nn_graph_layers 3\nn_FC_layers 2',
        #     'description': 'Coarse featurization',
        #     'n_graph_layers': 3,
        #     'n_FC_layers': 2,
        #     'embedding_size': 128,
        #     'epoch': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126],
        #     'train': [0.828, 0.853, 0.869, 0.858, 0.883, 0.882, 0.889, 0.877, 0.898, 0.897, 0.895, 0.896, 0.902, 0.915, 0.912, 0.885, 0.868, 0.895, 0.922, 0.927, 0.921, 0.933, 0.929, 0.932, 0.937, 0.937, 0.925, 0.937, 0.894, 0.937, 0.944, 0.926, 0.945, 0.944, 0.943, 0.94, 0.949, 0.945, 0.943, 0.893, 0.95, 0.946, 0.953, 0.947, 0.956, 0.956, 0.951, 0.953, 0.95, 0.956, 0.954, 0.953, 0.959, 0.954, 0.958, 0.944, 0.959, 0.96, 0.952, 0.956, 0.959, 0.964, 0.958, 0.961, 0.962, 0.964, 0.963, 0.961, 0.963, 0.965, 0.966, 0.954, 0.965, 0.955, 0.967, 0.964, 0.961, 0.96, 0.968, 0.966, 0.968, 0.964, 0.965, 0.967, 0.971, 0.959, 0.969, 0.97, 0.965, 0.965, 0.963, 0.968, 0.972, 0.97, 0.968, 0.973, 0.967, 0.969, 0.975, 0.973, 0.973, 0.962, 0.974, 0.971, 0.971, 0.973, 0.974, 0.969, 0.978, 0.972, 0.976, 0.961, 0.971, 0.956, 0.979, 0.977, 0.978, 0.974, 0.977, 0.98, 0.979, 0.978, 0.972, 0.98, 0.981, 0.978],
        #     'test':  [0.85, 0.867, 0.883, 0.868, 0.892, 0.889, 0.895, 0.888, 0.903, 0.903, 0.9, 0.903, 0.905, 0.916, 0.912, 0.895, 0.878, 0.905, 0.922, 0.929, 0.918, 0.93, 0.928, 0.931, 0.935, 0.932, 0.922, 0.93, 0.898, 0.927, 0.934, 0.924, 0.935, 0.934, 0.934, 0.929, 0.938, 0.934, 0.932, 0.89, 0.933, 0.931, 0.94, 0.934, 0.94, 0.939, 0.938, 0.938, 0.931, 0.937, 0.937, 0.938, 0.939, 0.932, 0.938, 0.927, 0.933, 0.939, 0.933, 0.933, 0.937, 0.94, 0.938, 0.938, 0.94, 0.937, 0.939, 0.936, 0.936, 0.939, 0.938, 0.932, 0.937, 0.936, 0.939, 0.939, 0.937, 0.935, 0.939, 0.934, 0.938, 0.937, 0.938, 0.937, 0.938, 0.929, 0.936, 0.939, 0.934, 0.934, 0.935, 0.938, 0.94, 0.938, 0.937, 0.938, 0.935, 0.936, 0.938, 0.938, 0.938, 0.927, 0.936, 0.937, 0.937, 0.941, 0.936, 0.935, 0.94, 0.936, 0.933, 0.92, 0.93, 0.919, 0.939, 0.938, 0.94, 0.934, 0.935, 0.938, 0.937, 0.937, 0.933, 0.937, 0.94, 0.936],
        #     'val':   [0.811, 0.836, 0.856, 0.849, 0.871, 0.869, 0.876, 0.87, 0.889, 0.885, 0.868, 0.881, 0.894, 0.899, 0.895, 0.875, 0.854, 0.89, 0.913, 0.914, 0.9, 0.919, 0.916, 0.914, 0.92, 0.918, 0.915, 0.916, 0.873, 0.913, 0.922, 0.913, 0.923, 0.921, 0.922, 0.918, 0.928, 0.921, 0.921, 0.875, 0.927, 0.924, 0.928, 0.922, 0.928, 0.928, 0.927, 0.926, 0.923, 0.931, 0.926, 0.925, 0.932, 0.925, 0.929, 0.919, 0.928, 0.929, 0.925, 0.928, 0.93, 0.929, 0.93, 0.926, 0.933, 0.926, 0.927, 0.923, 0.926, 0.931, 0.928, 0.921, 0.929, 0.927, 0.929, 0.928, 0.926, 0.924, 0.929, 0.927, 0.929, 0.924, 0.923, 0.929, 0.928, 0.921, 0.929, 0.928, 0.925, 0.926, 0.923, 0.925, 0.93, 0.927, 0.922, 0.93, 0.924, 0.928, 0.927, 0.927, 0.925, 0.917, 0.928, 0.926, 0.926, 0.929, 0.923, 0.921, 0.928, 0.926, 0.925, 0.905, 0.918, 0.907, 0.924, 0.928, 0.928, 0.925, 0.925, 0.925, 0.927, 0.924, 0.923, 0.928, 0.929, 0.925]
        # }
    ]
    data1x1 = [
        {
            'title': 'Forecaster graphs\nn_graph_layers 3\nn_FC_layers 2\nembedding_size 128',
            'description': 'Forecaster graphs',
            'n_graph_layers': 3,
            'n_FC_layers': 2,
            'embedding_size': 128,
            'epoch': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
            'train': [0.872, 0.897, 0.912, 0.917, 0.89, 0.923, 0.926, 0.929, 0.93, 0.922, 0.94, 0.921, 0.936, 0.944, 0.947, 0.951, 0.947, 0.958, 0.931, 0.957, 0.949, 0.938, 0.948, 0.959, 0.957, 0.961, 0.957, 0.962, 0.968, 0.947, 0.955, 0.962, 0.951, 0.965, 0.957],
            'test': [0.876, 0.893, 0.908, 0.911, 0.882, 0.914, 0.918, 0.916, 0.918, 0.91, 0.925, 0.908, 0.922, 0.926, 0.927, 0.932, 0.924, 0.934, 0.91, 0.926, 0.921, 0.914, 0.916, 0.925, 0.924, 0.927, 0.919, 0.928, 0.926, 0.906, 0.916, 0.918, 0.912, 0.917, 0.905],
            'val': [0.89, 0.903, 0.92, 0.92, 0.897, 0.922, 0.924, 0.921, 0.924, 0.92, 0.93, 0.913, 0.921, 0.925, 0.932, 0.933, 0.929, 0.936, 0.917, 0.93, 0.924, 0.917, 0.916, 0.927, 0.924, 0.928, 0.919, 0.925, 0.93, 0.908, 0.908, 0.921, 0.914, 0.923, 0.911]

        }
    ]

    def plot(dict):

        plt.rcParams.update({
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'lines.markersize': 3,  # Dot size
            'axes.grid': True,
            'axes.axisbelow': True,
            'grid.color': 'gainsboro',
            'legend.fontsize': 11,  # Legend text size
            'text.color': 'black',  # All text in black
            'axes.labelcolor': 'black',  # Axes labels in black
            'xtick.color': 'black',  # X-axis ticks in black
            'ytick.color': 'black',  # Y-axis ticks in black
            #'figure.figsize': (4, 4),
        })

        dict['train'] = [i * 100 for i in dict['train']]
        dict['test'] =  [i * 100 for i in dict['test']]
        dict['val'] =   [i * 100 for i in dict['val']]

        plt.clf()
        #fig = plt.figure(figsize=(4.25, 4))
        fig = plt.figure(figsize=(5, 4.44))
        plt.scatter(dict['epoch'], dict['train'], color='blue')
        plt.scatter(dict['epoch'], dict['val'], color='green')
        plt.scatter(dict['epoch'], dict['test'], color='darkorange')
        plt.ylim(70, 101)
        plt.title(dict['description'] +
                     "\nembedding_size " + str(dict['embedding_size']) +
                     "\nn_graph_layers " + str(dict['n_graph_layers']) +
                     "\nn_FC_layers " + str(dict['n_FC_layers']))
        plt.xlabel("epoch")
        plt.ylabel("Accuracy (%)")

        # add legend
        labels = ['Train set', 'Validation set', 'Test set']

        #plt.legend(labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=3, frameon=False, fontsize=11, markerscale=2) #bbox_transform=plt.transFigure,
        #plt.subplots_adjust(bottom=0.2, top=0.9)
        # plt.tight_layout(rect=[0, 0.1, 1, 1]) # [left, bottom, right, top]

        fig.legend(labels=labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=False, bbox_transform=fig.transFigure, fontsize=12, markerscale=3)
        plt.tight_layout(rect=[0.04, 0.1, 0.94, 1]) # [left, bottom, right, top]
        #plt.tight_layout()

        plt.savefig('GNN_accuracy_' +
                    dict['description'] +
                    str(dict['embedding_size']) + '_' +
                    str(dict['n_graph_layers']) + '_' +
                    str(dict['n_FC_layers']) + '.png')
    #plot(data[7])
    for dict in data1x1:
        plot(dict)

    plt.clf()
    plt.rcParams.update({
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'lines.markersize': 3,  # Dot size
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': 'gainsboro',
        'legend.fontsize': 11,  # Legend text size
        'text.color': 'black',  # All text in black
        'axes.labelcolor': 'black',  # Axes labels in black
        'xtick.color': 'black',  # X-axis ticks in black
        'ytick.color': 'black',  # Y-axis ticks in black
    })
    plt.clf()

    def subplot(ax, dict):

        dict['train'] = [i * 100 for i in dict['train']]
        dict['test'] =  [i * 100 for i in dict['test']]
        dict['val'] =   [i * 100 for i in dict['val']]

        ax.scatter(dict['epoch'], dict['train'], color='blue')
        ax.scatter(dict['epoch'], dict['val'], color='green')
        ax.scatter(dict['epoch'], dict['test'], color='darkorange')
        ax.set_ylim(70, 101)
        ax.set_title(dict['description'] +
                     "\nembedding size: " + str(dict['embedding_size']) +
                     "\nno. graph layers: " + str(dict['n_graph_layers']) +
                     "\nno. FC layers: " + str(dict['n_FC_layers']))

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(8.5, 4))
    for i, ax in enumerate(axs.flatten()):
        subplot(ax, data2x1[i])

    # for ax in axs.flat:
    #     ax.set(xlabel='epoch', ylabel='Accuracy (%)')
    # for ax in fig.get_axes():
    #     ax.label_outer()

    # Ensure x-tick values are shown on all subplots
    for ax in axs.flat:
        ax.tick_params(axis='x', labelbottom=True)
    #for ax in axs[:, 0]:  # Only the first column of subplots
    #    ax.set_ylabel('Accuracy (%)')
    axs[0].set_ylabel('Accuracy (%)')
    # Set the xlabel only on the bottom row subplots
    for ax in axs:  # Only the last row of subplots
        ax.set_xlabel('epoch')

    # # add legend
    # labels = ['Train set', 'Validation set', 'Test set']
    # fig.legend(labels=labels, loc='lower center', bbox_to_anchor=(0.5, 0.04), ncol=3, frameon=False, bbox_transform=fig.transFigure, fontsize=11, markerscale=10)
    # plt.tight_layout(rect=[0, 0.07, 1, 1])

    plt.tight_layout()

    fig.savefig('Prediction_accuracy_2x1.png')
def plot_prediction_accuracies_4x4():
    
    # python3 main.py --mode plotstuff
    
    data = [
        {
            'set': 'PDBBind Set (train)',
            'RMSD':              [0.00, 0.25,  0.50,  0.75,  1.00,  1.25,  1.50,  1.75,  2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy':  [0.00, 0.95, 15.34, 28.19, 38.04, 46.36, 52.70, 57.73, 61.40, 64.72, 67.67, 70.03, 72.11, 73.97, 75.61, 77.39, 78.86, 80.43, 81.60, 82.62, 83.94],
            'sampling_accuracy': [0.00, 3.04, 27.31, 45.61, 56.90, 65.50, 71.19, 75.58, 78.47, 81.39, 83.53, 85.12, 86.63, 88.21, 89.62, 90.70, 91.67, 92.78, 93.61, 94.24, 94.87],
            'random':            [0.00, 0.96, 12.60, 23.03, 31.70, 39.10, 45.74, 51.00, 55.17, 59.16, 62.79, 65.42, 67.89, 69.97, 72.07, 74.08, 76.03, 77.65, 79.08, 80.64, 82.04],
            'XGBc':              [0.00, 1.30, 16.12, 30.27, 41.04, 50.11, 56.87, 62.00, 65.90, 69.25, 72.02, 74.18, 76.08, 77.88, 79.63, 81.11, 82.47, 83.75, 84.71, 85.64, 86.80],
            'LRc':               [0.00, 1.33, 16.33, 28.95, 39.74, 48.28, 55.37, 60.54, 64.96, 68.18, 71.09, 73.52, 75.38, 77.12, 78.50, 80.10, 81.36, 82.86, 83.88, 84.99, 86.09],
            'GNN - RMSD threshold': [0,     0.25,       0.5,      0.75,         1,      1.25,       1.5,      1.75,         2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5],
            'GNN - Lowest RMSD':    [0,  2.90874, 28.701406, 47.463468, 59.057072, 67.383513, 72.870141, 76.978219, 79.955886, 82.657844, 84.739454, 86.55914, 88.075545, 89.61952, 90.805073, 91.797629, 92.624759, 93.672457, 94.347946, 95.037221, 95.368073],
            'GNN - Random':         [0, 0.896057, 12.848084,  24.24869, 33.291977, 40.887786,   47.2429,  52.08161, 56.437827, 60.215054, 63.330576, 65.756824, 67.769506, 70.002757, 71.918941, 74.262476, 76.137304, 77.901847, 79.500965, 80.989799, 82.395919],
            'GNN - Predicted':      [0, 1.240695, 18.183071, 34.063965, 45.960849, 55.913978, 63.027295, 68.637993, 72.911497, 76.30273, 78.770334, 80.424593, 82.27185, 83.52633, 85.042735, 86.200717, 87.331128, 88.420182, 89.385167, 90.29501, 91.039427]

    },
        {
            'set': 'PDBBind Set (test)',
            'RMSD':              [0.00, 0.25,  0.50,  0.75,  1.00,  1.25, 1.50,  1.75,  2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy':  [0.00, 1.17, 17.98, 32.12, 42.80, 52.16, 58.54, 64.27, 68.18, 71.12, 73.53, 75.76, 78.12, 80.10, 81.52, 83.09, 84.46, 85.78, 87.17, 88.64, 89.91],
            'sampling_accuracy': [0.00, 3.47, 32.76, 52.69, 64.25, 72.92, 78.42, 81.92, 85.02, 87.37, 89.35, 91.15, 92.49, 93.41, 94.22, 94.90, 95.66, 96.07, 96.63, 97.06, 97.41],
            'random':            [0.00, 1.06, 15.39, 27.84, 37.55, 47.11, 54.08, 59.41, 63.89, 67.82, 70.66, 73.48, 75.89, 78.25, 79.87, 81.47, 82.96, 84.53, 85.98, 87.20, 88.41],
            'XGBc':              [0.00, 1.04, 18.97, 34.74, 45.66, 55.78, 63.51, 69.14, 72.72, 76.04, 78.32, 80.93, 82.91, 84.79, 86.16, 87.27, 88.34, 89.25, 90.16, 91.13, 91.99],
            'LRc':               [0.00, 0.91, 17.22, 31.39, 42.01, 51.98, 58.72, 65.06, 69.07, 72.92, 75.76, 78.52, 80.53, 82.83, 84.36, 85.62, 86.69, 87.73, 89.02, 90.34, 91.20],
            'GNN - RMSD threshold': [0,     0.25,       0.5,      0.75,         1,      1.25,       1.5,      1.75,         2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5],
            'GNN - Lowest RMSD':    [0, 4.846025, 35.736453, 56.433516, 68.106947, 76.223442, 81.260444, 84.148962, 86.607782, 88.732394, 90.403438, 91.931249, 92.957746, 93.888756, 94.772022, 95.368823, 96.132729, 96.514681, 97.063738, 97.541179, 97.756028],
            'GNN - Random':         [0, 1.312963, 17.307233, 30.412986, 40.033421, 49.773216, 56.815469, 62.043447, 66.149439, 69.706374, 72.451659, 75.02984, 77.512533, 79.207448, 81.2127, 82.883743, 84.196706, 85.820005, 87.228455, 88.350442, 89.4963],
            'GNN - Predicted':      [0, 1.718787, 22.797804, 39.770828, 52.733349, 64.168059, 71.425161, 76.390547, 80.377178, 82.955359, 85.270948, 87.562664, 88.756266, 89.61566, 90.642158, 91.47768, 92.480306, 93.148723, 93.912628, 94.390069, 94.891382]
        },
        {
            'set': 'Fitted Set',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy': [0.00, 1.49, 32.71, 46.47, 56.51, 65.80, 72.12, 75.46, 78.07, 80.30, 82.16, 84.01, 86.25, 88.10, 89.22, 89.96, 90.33, 90.71, 91.08, 92.19, 92.57],
            'sampling_accuracy': [0.00, 5.58, 46.47, 66.54, 76.21, 84.76, 87.36, 90.33, 91.08, 92.19, 94.05, 94.05, 94.80, 94.80, 94.80, 95.91, 96.28, 97.03, 97.03, 97.03, 97.40],
            'random': [0.00, 1.12, 24.91, 38.29, 49.44, 57.62, 63.20, 67.29, 72.49, 75.46, 78.07, 81.04, 82.53, 83.64, 84.39, 85.13, 85.87, 86.99, 87.73, 88.10, 88.85],
            'XGBc': [0.00, 2.97, 28.62, 47.58, 57.99, 65.43, 71.00, 75.84, 79.55, 83.27, 86.25, 87.73, 89.22, 90.33, 90.71, 92.19, 92.19, 93.31, 93.68, 94.42, 94.42],
            'LRc': [0.00, 3.72, 32.71, 47.58, 57.99, 65.80, 70.26, 75.84, 79.55, 83.27, 85.13, 87.36, 89.22, 90.33, 90.33, 91.45, 91.82, 92.19, 92.57, 93.68, 94.42],
            'GNN - Predicted': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]
        },
        {
            'set': 'Astex Set',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy': [0.00, 2.90, 18.84, 31.88, 40.58, 49.64, 58.70, 66.67, 72.46, 73.55, 75.72, 77.17, 78.99, 81.52, 82.25, 84.06, 84.78, 86.96, 88.41, 89.86, 90.58],
            'sampling_accuracy': [0.00, 6.88, 28.99, 46.01, 59.06, 69.20, 76.09, 78.99, 81.16, 83.33, 86.23, 87.32, 88.41, 90.22, 90.58, 91.67, 93.48, 94.93, 96.01, 96.01, 96.01],
            'random': [0.00, 2.54, 17.03, 28.99, 38.41, 49.28, 56.52, 59.78, 61.59, 65.58, 69.93, 72.46, 73.55, 75.36, 76.81, 78.26, 78.99, 81.52, 83.33, 85.87, 86.59],
            'XGBc': [0.00, 3.62, 21.38, 32.61, 42.75, 56.16, 64.86, 69.20, 71.74, 76.45, 78.99, 79.71, 81.16, 82.25, 83.33, 84.42, 84.78, 86.96, 88.04, 89.86, 90.22],
            'LRc': [0.00, 1.81, 20.29, 32.97, 43.48, 53.99, 63.04, 70.29, 72.83, 75.72, 77.17, 78.99, 79.71, 81.88, 82.97, 84.78, 85.51, 86.23, 88.77, 90.22, 90.94],
            'GNN - Predicted': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]
        }
    ]

    data2 = [
        {
            'set': 'PDBBind Set (full)',
            'RMSD':              [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy':  [0.000000, 1.309614, 18.455715, 32.528388, 42.581378, 50.968963, 57.259652, 61.688115, 65.352006, 68.334595, 71.150643, 73.323240, 75.382286, 76.926571, 78.607116, 80.242241, 81.680545, 82.967449, 84.239213, 85.518547, 86.691900],
            'sampling_accuracy': [0.000000, 3.830431, 31.574565, 51.052233, 62.588948, 70.802422, 76.230129, 79.947010, 82.619228, 85.049205, 86.994701, 88.599546, 89.954580, 91.067373, 92.089326, 92.922029, 93.701741, 94.451173, 95.094625, 95.753217, 96.146858],
            'random':            [0.000000, 1.143073, 15.155185, 26.971991, 36.063588, 44.473883, 50.968963, 56.033308, 59.803179, 63.580621, 66.691900, 69.326268, 71.483724, 73.580621, 75.541257, 77.191522, 78.894777, 80.355791, 81.801665, 83.361090, 84.663134],
            'XGBc':              [0.000000, 1.427115,17.473666,31.489976,41.938498,51.333673,58.724091,63.948352,68.119266,71.398233,74.065579,76.257221,78.109072,80.071356,81.481481,82.883112,84.123344,85.270133,86.264016,87.325858,88.328236],
            'LRc':               [0.00, 1.33, 16.33, 28.95, 39.74, 48.28, 55.37, 60.54, 64.96, 68.18, 71.09, 73.52, 75.38, 77.12, 78.50, 80.10, 81.36, 82.86, 83.88, 84.99, 86.09],
            'GNN - RMSD threshold': [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5],
            'GNN - Lowest RMSD':    [0.000000, 3.827355, 31.648996, 51.267013, 62.891947, 71.200842, 76.517031, 80.036093, 82.750583, 85.171817, 87.081735, 88.751034, 90.021806, 91.217385, 92.262576, 93.044590, 93.819084, 94.586059, 95.202647, 95.819235, 96.165125],
            'GNN - Random':         [0.000000, 1.120385, 14.580044, 26.746372, 36.160614, 44.800361, 51.402361, 56.462892, 60.718851, 64.373261, 67.403564, 69.832318, 71.862546, 73.825100, 75.562072, 77.509587, 79.148808, 80.697797, 82.118956, 83.540116, 84.810888],
            'GNN - Predicted':      [0.000000, 1.488834, 20.219565, 36.491466, 48.898413, 59.515753, 66.764418, 71.997895, 75.990676, 79.013460, 81.442214, 83.322054, 84.856004, 85.946312, 87.232123, 88.209640, 89.239792, 90.112038, 90.969246, 91.721182, 92.420483]

        },
        {
            'set': 'PDBBind Set (val)',
            'RMSD':              [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy':  [0.000000, 1.495845, 20.110803, 34.515235, 45.041551, 54.459834, 60.443213, 64.986150, 68.919668, 72.299169, 75.401662, 78.060942, 79.501385, 80.886427, 81.883657, 82.936288, 83.822715, 85.096953, 85.650970, 86.481994, 87.977839],
            'sampling_accuracy': [0.000000, 5.041551, 33.240997, 52.354571, 65.207756, 73.795014, 79.168975, 82.880886, 85.373961, 87.423823, 89.141274, 90.415512, 91.301939, 91.855956, 92.299169, 92.742382, 93.296399, 93.905817, 94.515235, 94.958449, 95.678670],
            'random':            [0.000000, 1.772853, 15.900277, 28.531856, 38.947368, 49.307479, 56.011080, 60.221607, 64.709141, 67.756233, 70.803324, 73.573407, 75.290859, 76.897507, 78.282548, 79.722992, 80.997230, 82.548476, 83.767313, 84.653740, 85.817175],
            'XGBc':              [0.000000, 2.216066, 20.387812, 34.34903, 46.204986, 56.34349, 63.98892, 68.254848, 71.800554, 74.293629, 77.67313, 79.722992, 81.440443, 82.493075, 83.268698, 84.376731, 85.096953, 86.426593, 87.257618, 88.254848, 88.975069],
            'LRc':               [0.000000, 1.994460,19.944598,33.684211,45.373961,56.288089,63.656510,68.753463,72.908587,75.900277,78.947368,80.941828,82.437673,83.545706,84.376731,85.373961,86.204986,87.423823,88.144044,89.252078,90.083102],
            'GNN - RMSD threshold': [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5],
            'GNN - Lowest RMSD':    [0.000000, 5.118534, 33.943966, 54.471983, 66.109914, 74.784483, 80.064655, 82.704741, 84.967672, 86.961207, 88.739224, 90.140086, 91.002155, 91.433190, 92.295259, 92.672414, 93.265086, 93.803879, 94.342672, 94.989224, 95.689655],
            'GNN - Random':         [0.000000, 1.346983, 18.372845, 30.818966, 40.463362, 49.568966, 56.196121, 61.260776, 64.439655, 68.426724, 71.336207, 73.976293, 75.808190, 77.424569, 78.717672, 80.010776, 81.196121, 82.058190, 82.920259, 84.267241, 85.560345],
            'GNN - Predicted':      [0.000000, 1.939655, 22.359914, 38.577586, 51.724138, 63.092672, 70.851293, 75.215517, 78.125000, 80.711207, 83.243534, 85.075431, 86.153017, 87.122845, 88.092672, 88.685345, 89.385776, 89.870690, 90.517241, 91.271552, 92.241379]
        },
        {
            'set': 'Fitted Set',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy': [0.00, 1.49, 32.71, 46.47, 56.51, 65.80, 72.12, 75.46, 78.07, 80.30, 82.16, 84.01, 86.25, 88.10, 89.22, 89.96, 90.33, 90.71, 91.08, 92.19, 92.57],
            'sampling_accuracy': [0.00, 5.58, 46.47, 66.54, 76.21, 84.76, 87.36, 90.33, 91.08, 92.19, 94.05, 94.05, 94.80, 94.80, 94.80, 95.91, 96.28, 97.03, 97.03, 97.03, 97.40],
            'random': [0.00, 1.12, 24.91, 38.29, 49.44, 57.62, 63.20, 67.29, 72.49, 75.46, 78.07, 81.04, 82.53, 83.64, 84.39, 85.13, 85.87, 86.99, 87.73, 88.10, 88.85],
            'XGBc': [0.00, 2.97, 28.62, 47.58, 57.99, 65.43, 71.00, 75.84, 79.55, 83.27, 86.25, 87.73, 89.22, 90.33, 90.71, 92.19, 92.19, 93.31, 93.68, 94.42, 94.42],
            'LRc': [0.00, 3.72, 32.71, 47.58, 57.99, 65.80, 70.26, 75.84, 79.55, 83.27, 85.13, 87.36, 89.22, 90.33, 90.33, 91.45, 91.82, 92.19, 92.57, 93.68, 94.42],
            'GNN - Predicted': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]
        },
        {
            'set': 'Astex Set',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy': [0.00, 2.90, 18.84, 31.88, 40.58, 49.64, 58.70, 66.67, 72.46, 73.55, 75.72, 77.17, 78.99, 81.52, 82.25, 84.06, 84.78, 86.96, 88.41, 89.86, 90.58],
            'sampling_accuracy': [0.00, 6.88, 28.99, 46.01, 59.06, 69.20, 76.09, 78.99, 81.16, 83.33, 86.23, 87.32, 88.41, 90.22, 90.58, 91.67, 93.48, 94.93, 96.01, 96.01, 96.01],
            'random': [0.00, 2.54, 17.03, 28.99, 38.41, 49.28, 56.52, 59.78, 61.59, 65.58, 69.93, 72.46, 73.55, 75.36, 76.81, 78.26, 78.99, 81.52, 83.33, 85.87, 86.59],
            'XGBc': [0.00, 3.62, 21.38, 32.61, 42.75, 56.16, 64.86, 69.20, 71.74, 76.45, 78.99, 79.71, 81.16, 82.25, 83.33, 84.42, 84.78, 86.96, 88.04, 89.86, 90.22],
            'LRc': [0.00, 1.81, 20.29, 32.97, 43.48, 53.99, 63.04, 70.29, 72.83, 75.72, 77.17, 78.99, 79.71, 81.88, 82.97, 84.78, 85.51, 86.23, 88.77, 90.22, 90.94],
            'GNN - Predicted': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]
        }
    ]

    def plot(dict):

        plt.rcParams.update({
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            # 'lines.markersize': 1,  # Dot size
            'axes.grid': True,
            'axes.axisbelow': True,
            'grid.color': 'gainsboro',
            'legend.fontsize': 11,  # Legend text size
            'text.color': 'black',  # All text in black
            'axes.labelcolor': 'black',  # Axes labels in black
            'xtick.color': 'black',  # X-axis ticks in black
            'ytick.color': 'black',  # Y-axis ticks in black
        })

        #dict['train'] = [i * 100 for i in dict['train']]
        #dict['test'] = [i * 100 for i in dict['test']]
        #dict['val'] = [i * 100 for i in dict['val']]

        plt.clf()
        ax.plot(dict['RMSD'], dict['scoring_accuracy'], color='blue')
        ax.plot(dict['RMSD'], dict['sampling_accuracy'], color='red')
        ax.plot(dict['RMSD'], dict['random'], color='darkorange')
        ax.plot(dict['RMSD'], dict['XGBc'], color='green')
        ax.plot(dict['RMSD'], dict['LRc'], color='green', linestyle='dashed')
        #if dict['set'] == 'PDBBind Set (train)' or dict['set'] == 'PDBBind Set (test)':
        #ax.plot(dict['RMSD'], dict['GNN - Predicted'], color='green', linestyle='dotted')

        plt.title(dict['set'])
        plt.xlabel("RMSD (Å)")
        plt.ylabel("Accuracy (%)")
        plt.xticks(np.arange(0, 5.1, 0.5))
        plt.yticks(np.arange(0, 101, 10))
        plt.xlim(-0.1, 5.1)
        plt.ylim(-2.5, 102.5)

        #plt.savefig('Prediction accuracy top 100' + dict['n_graph_layers'] + '.png')

    # plot(data[7])
    #for dict in data2:
    #    plot(dict)

    plt.rcParams.update({
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        #'lines.markersize': 1,  # Dot size
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': 'gainsboro',
        'legend.fontsize': 11,         # Legend text size
        'text.color': 'black',        # All text in black
        'axes.labelcolor': 'black',   # Axes labels in black
        'xtick.color': 'black',       # X-axis ticks in black
        'ytick.color': 'black',       # Y-axis ticks in black
    })
    plt.clf()

    def subplot(ax, dict):

        #dict['train'] = [i * 100 for i in dict['train']]
        #dict['test'] = [i * 100 for i in dict['test']]
        #dict['val'] = [i * 100 for i in dict['val']]

        ax.plot(dict['RMSD'], dict['scoring_accuracy'], color='blue')
        ax.plot(dict['RMSD'], dict['sampling_accuracy'], color='red')
        ax.plot(dict['RMSD'], dict['random'], color='darkorange')
        ax.plot(dict['RMSD'], dict['XGBc'], color='green')
        ax.plot(dict['RMSD'], dict['LRc'], color='green', linestyle='dashed')
        if dict['set'] == 'PDBBind Set (full)' or dict['set'] == 'PDBBind Set (val)':
            ax.plot(dict['RMSD'], dict['GNN - Predicted'], color='green', linestyle='dotted')

        ax.set_xticks(np.arange(0, 5.1, 1))
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xlim(-0.1, 5.1)
        ax.set_ylim(-2.5, 102.5)
        #ax.xlabel("RMSD (Å)")
        #ax.ylabel("Accuracy (%)")
        ax.set_title(dict['set'], fontsize=12)

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, figsize=(8.5, 7))
    for i, ax in enumerate(axs.flatten()):
        subplot(ax, data2[i])

    # for ax in axs.flat:
    #     ax.set(xlabel='epoch', ylabel='Accuracy (%)')
    # for ax in fig.get_axes():
    #     ax.label_outer()

    # Ensure x-tick values are shown on all subplots
    for ax in axs.flat:
        ax.tick_params(axis='x', labelbottom=True)
    for ax in axs[:, 0]:  # Only the first column of subplots
        ax.set_ylabel('Accuracy (%)')
    # Set the xlabel only on the bottom row subplots
    for ax in axs[-1, :]:  # Only the last row of subplots
        ax.set_xlabel('RMSD (Å)')

    # add legend
    #labels1 = ['Scoring accuracy', 'Sampling accuracy', 'Random accuracy']
    labels1 = ['Best energy', 'Best RMSD', 'Random']
    labels2 = ['_nolegend_','_nolegend_','_nolegend_','XGBc', 'LRc', 'GNN']
    fig.legend(labels=labels1, loc='lower center', bbox_to_anchor=(0.5, 0.03), ncol=3, frameon=False, bbox_transform=fig.transFigure)
    fig.legend(labels=labels2, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=False, bbox_transform=fig.transFigure)

    legend_handles = [
        Line2D([0], [0], color='b', lw=1, label='Line'),

        Patch(facecolor='blue', label='Scoring accuracy'),
        Patch(facecolor='red', label='Sampling accuracy'),
        Patch(facecolor='darkorange', label='Random'),
        Patch(facecolor='green', label='XGBc'),
        Patch(facecolor='green', label='LRc'),
    ]
    #fig.legend(handles=legend_handles, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)  # , , )#, title='Label',
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    #plt.tight_layout()

    fig.savefig('Prediction_accuracy_with_GNN.png')

def plot_prediction_accuracies_2x1():
    regression = [
        {
            'set': 'Fitted Set',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy': [0.0, 2.5547445255474455, 33.57664233576642, 47.08029197080292, 56.934306569343065, 66.42335766423358, 72.62773722627738, 75.91240875912409, 78.46715328467154, 80.65693430656934, 82.48175182481752, 84.30656934306569, 86.4963503649635, 88.32116788321167, 89.41605839416059, 90.14598540145985, 90.51094890510949, 90.87591240875912, 91.24087591240875, 92.33576642335767, 92.7007299270073],
            'sampling_accuracy': [0.0, 6.569343065693431, 47.08029197080292, 66.78832116788321, 76.27737226277372, 85.03649635036497, 87.5912408759124, 90.51094890510949, 91.24087591240875, 92.33576642335767, 94.16058394160584, 94.16058394160584, 94.8905109489051, 94.8905109489051, 94.8905109489051, 95.98540145985402, 96.35036496350365, 97.08029197080292, 97.08029197080292, 97.08029197080292, 97.44525547445255],
            'random': [0.0, 2.18978102189781, 28.467153284671532, 43.065693430656935, 53.284671532846716, 62.40875912408759, 67.88321167883211, 70.07299270072993, 73.35766423357664, 78.10218978102189, 80.65693430656934, 82.11678832116789, 83.57664233576642, 85.03649635036497, 85.4014598540146, 86.4963503649635, 86.86131386861314, 88.68613138686132, 90.14598540145985, 90.51094890510949, 91.60583941605839],
            'XGBc': [0.0, 3.6496350364963503, 27.37226277372263, 45.25547445255474, 60.583941605839414, 73.35766423357664, 81.75182481751825, 88.32116788321167, 91.24087591240875, 92.33576642335767, 93.06569343065694, 94.16058394160584, 94.52554744525547, 94.52554744525547, 94.52554744525547, 94.8905109489051, 95.25547445255475, 96.35036496350365, 96.71532846715328, 96.71532846715328, 96.71532846715328],
            'LRc': [0.0, 2.9197080291970803, 25.912408759124087, 39.416058394160586, 51.09489051094891, 59.85401459854015, 63.86861313868613, 68.61313868613139, 72.26277372262774, 76.64233576642336, 79.92700729927007, 82.11678832116789, 83.57664233576642, 84.67153284671532, 85.76642335766424, 86.86131386861314, 87.5912408759124, 88.68613138686132, 90.14598540145985, 90.87591240875912, 91.97080291970804]
        },
        {
            'set': 'Astex Set',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy': [0.0, 2.8776978417266186, 18.705035971223023, 31.654676258992804, 40.28776978417266, 49.280575539568346, 58.273381294964025, 66.54676258992805, 72.3021582733813, 73.38129496402878, 75.53956834532374, 76.97841726618705, 78.77697841726619, 81.29496402877697, 82.01438848920863, 83.81294964028777, 84.53237410071942, 87.05035971223022, 88.48920863309353, 89.92805755395683, 90.64748201438849],
            'sampling_accuracy': [0.0, 6.83453237410072, 28.776978417266186, 45.68345323741007, 58.63309352517986, 69.06474820143885, 75.89928057553956, 79.13669064748201, 81.29496402877697, 83.45323741007195, 86.33093525179856, 87.41007194244604, 88.48920863309353, 90.28776978417267, 90.64748201438849, 91.72661870503597, 93.5251798561151, 94.96402877697842, 96.0431654676259, 96.0431654676259, 96.0431654676259],
            'random': [0.0, 1.7985611510791366, 15.467625899280575, 26.97841726618705, 36.69064748201439, 43.884892086330936, 52.15827338129496, 56.115107913669064, 60.07194244604317, 64.02877697841727, 67.98561151079137, 69.7841726618705, 71.22302158273381, 72.66187050359713, 75.53956834532374, 76.61870503597122, 78.77697841726619, 81.65467625899281, 83.81294964028777, 84.53237410071942, 86.33093525179856],
            'XGBc': [0.0, 2.158273381294964, 19.424460431654676, 29.496402877697843, 37.76978417266187, 48.92086330935252, 55.75539568345324, 60.07194244604317, 63.669064748201436, 66.54676258992805, 69.42446043165468, 72.66187050359713, 73.38129496402878, 73.7410071942446, 75.53956834532374, 76.61870503597122, 77.6978417266187, 80.93525179856115, 82.01438848920863, 83.09352517985612, 83.45323741007195],
            'LRc': [0.0, 2.158273381294964, 17.26618705035971, 29.85611510791367, 37.76978417266187, 46.402877697841724, 52.15827338129496, 56.47482014388489, 59.71223021582734, 64.02877697841727, 68.34532374100719, 69.06474820143885, 71.58273381294964, 73.7410071942446, 74.46043165467626, 76.61870503597122, 78.41726618705036, 79.85611510791367, 81.29496402877697, 83.45323741007195, 83.81294964028777]
        }
    ]

    classifier = [
        {
            'set': 'Fitted Set',
            'RMSD':              [0.00, 0.25,  0.50,  0.75,  1.00,  1.25,  1.50,  1.75,  2.00,  2.25,  2.50,  2.75,  3.00,  3.25,  3.50,  3.75,  4.00,  4.25,  4.50,  4.75,  5.00],
            'scoring_accuracy':  [0.00, 2.55, 33.58, 47.08, 56.93, 66.42, 72.63, 75.91, 78.47, 80.66, 82.48, 84.31, 86.50, 88.32, 89.42, 90.15, 90.51, 90.88, 91.24, 92.34, 92.70],
            'sampling_accuracy': [0.00, 6.57, 47.08, 66.79, 76.28, 85.04, 87.59, 90.51, 91.24, 92.34, 94.16, 94.16, 94.89, 94.89, 94.89, 95.99, 96.35, 97.08, 97.08, 97.08, 97.45],
            'random':            [0.00, 2.92, 23.72, 37.59, 46.72, 56.20, 63.87, 68.98, 72.26, 74.82, 78.47, 79.56, 82.48, 84.67, 85.40, 86.86, 87.23, 89.05, 90.15, 90.88, 91.24],
            'XGBc':              [0.00, 2.92, 25.91, 39.78, 51.09, 59.85, 63.87, 67.88, 72.26, 76.64, 79.93, 81.75, 83.21, 84.67, 85.77, 86.86, 87.59, 88.69, 90.51, 90.88, 91.97],
            'LRc':               [0.00, 2.00, 16.00, 20.00, 34.50, 43.50, 51.00, 55.00, 57.00, 63.00, 67.00, 68.00, 69.50, 72.00, 73.00, 74.00, 76.00, 77.00, 79.50, 81.50, 82.00]
        },
        {
            'set': 'Astex Set',
            'RMSD':              [0.00, 0.25,  0.50,  0.75,  1.00,  1.25,  1.50,  1.75,  2.00,  2.25,  2.50,  2.75,  3.00,  3.25,  3.50,  3.75,  4.00,  4.25,  4.50,  4.75,  5.00],
            'scoring_accuracy':  [0.00, 2.88, 18.71, 31.65, 40.29, 49.28, 58.27, 66.55, 72.30, 73.38, 75.54, 76.98, 78.78, 81.29, 82.01, 83.81, 84.53, 87.05, 88.49, 89.93, 90.65],
            'sampling_accuracy': [0.00, 6.83, 28.78, 45.68, 58.63, 69.06, 75.90, 79.14, 81.29, 83.45, 86.33, 87.41, 88.49, 90.29, 90.65, 91.73, 93.53, 94.96, 96.04, 96.04, 96.04],
            'random':            [0.00, 2.52, 17.27, 26.62, 34.53, 43.53, 49.64, 55.76, 60.43, 63.67, 67.99, 68.71, 69.42, 71.58, 73.74, 75.18, 76.98, 79.14, 80.58, 82.73, 83.45],
            'XGBc':              [0.00, 2.16, 17.27, 29.86, 38.13, 46.76, 52.52, 56.47, 59.71, 63.67, 67.99, 68.71, 70.86, 73.38, 74.46, 76.26, 78.06, 79.50, 80.94, 83.09, 83.45],
            'LRc':               [0.00, 2.00, 16.00, 20.00, 34.50, 43.50, 51.00, 55.00, 57.00, 63.00, 67.00, 68.00, 69.50, 72.00, 73.00, 74.00, 76.00, 77.00, 79.50, 81.50, 82.00]
        }
    ]

    classifier_PDBBind = [
        # {
        #     'set': 'PDBBind Set (train)',
        #     'RMSD':              [0.00, 0.25,  0.50,  0.75,  1.00,  1.25,  1.50,  1.75,  2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
        #     'scoring_accuracy':  [0.00, 0.95, 15.34, 28.19, 38.04, 46.36, 52.70, 57.73, 61.40, 64.72, 67.67, 70.03, 72.11, 73.97, 75.61, 77.39, 78.86, 80.43, 81.60, 82.62, 83.94],
        #     'sampling_accuracy': [0.00, 3.04, 27.31, 45.61, 56.90, 65.50, 71.19, 75.58, 78.47, 81.39, 83.53, 85.12, 86.63, 88.21, 89.62, 90.70, 91.67, 92.78, 93.61, 94.24, 94.87],
        #     'random':            [0.00, 0.96, 12.60, 23.03, 31.70, 39.10, 45.74, 51.00, 55.17, 59.16, 62.79, 65.42, 67.89, 69.97, 72.07, 74.08, 76.03, 77.65, 79.08, 80.64, 82.04],
        #     'XGBc':              [0.00, 1.30, 16.12, 30.27, 41.04, 50.11, 56.87, 62.00, 65.90, 69.25, 72.02, 74.18, 76.08, 77.88, 79.63, 81.11, 82.47, 83.75, 84.71, 85.64, 86.80],
        #     'LRc':               [0.00, 1.33, 16.33, 28.95, 39.74, 48.28, 55.37, 60.54, 64.96, 68.18, 71.09, 73.52, 75.38, 77.12, 78.50, 80.10, 81.36, 82.86, 83.88, 84.99, 86.09],
        #     'GNN - RMSD threshold': [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5],
        #     'GNN - Lowest RMSD':   [0, 2.90874, 28.701406, 47.463468, 59.057072, 67.383513, 72.870141, 76.978219, 79.955886, 82.657844, 84.739454, 86.55914, 88.075545, 89.61952, 90.805073, 91.797629, 92.624759, 93.672457, 94.347946, 95.037221, 95.368073],
        #     'GNN - Random':      [0, 0.896057, 12.848084, 24.24869, 33.291977, 40.887786, 47.2429, 52.08161, 56.437827, 60.215054, 63.330576, 65.756824, 67.769506, 70.002757, 71.918941, 74.262476, 76.137304, 77.901847, 79.500965, 80.989799, 82.395919],
        #     'GNN - Predicted':     [0, 1.240695, 18.183071, 34.063965, 45.960849, 55.913978, 63.027295, 68.637993, 72.911497, 76.30273, 78.770334, 80.424593, 82.27185, 83.52633, 85.042735, 86.200717, 87.331128, 88.420182, 89.385167, 90.29501, 91.039427]
        # 
        # },
        # {
        #     'set': 'PDBBind Set (test)',
        #     'RMSD':              [0.00, 0.25,  0.50,  0.75,  1.00,  1.25,  1.50,  1.75,  2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
        #     'scoring_accuracy':  [0.00, 1.17, 17.98, 32.12, 42.80, 52.16, 58.54, 64.27, 68.18, 71.12, 73.53, 75.76, 78.12, 80.10, 81.52, 83.09, 84.46, 85.78, 87.17, 88.64, 89.91],
        #     'sampling_accuracy': [0.00, 3.47, 32.76, 52.69, 64.25, 72.92, 78.42, 81.92, 85.02, 87.37, 89.35, 91.15, 92.49, 93.41, 94.22, 94.90, 95.66, 96.07, 96.63, 97.06, 97.41],
        #     'random':            [0.00, 1.06, 15.39, 27.84, 37.55, 47.11, 54.08, 59.41, 63.89, 67.82, 70.66, 73.48, 75.89, 78.25, 79.87, 81.47, 82.96, 84.53, 85.98, 87.20, 88.41],
        #     'XGBc':              [0.00, 1.04, 18.97, 34.74, 45.66, 55.78, 63.51, 69.14, 72.72, 76.04, 78.32, 80.93, 82.91, 84.79, 86.16, 87.27, 88.34, 89.25, 90.16, 91.13, 91.99],
        #     'LRc':               [0.00, 0.91, 17.22, 31.39, 42.01, 51.98, 58.72, 65.06, 69.07, 72.92, 75.76, 78.52, 80.53, 82.83, 84.36, 85.62, 86.69, 87.73, 89.02, 90.34, 91.20],
        #     'GNN - RMSD threshold': [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5],
        #     'GNN - Lowest RMSD': [0, 4.846025, 35.736453, 56.433516, 68.106947, 76.223442, 81.260444, 84.148962, 86.607782, 88.732394, 90.403438, 91.931249, 92.957746, 93.888756, 94.772022, 95.368823, 96.132729, 96.514681, 97.063738, 97.541179, 97.756028],
        #     'GNN - Random': [0, 1.312963, 17.307233, 30.412986, 40.033421, 49.773216, 56.815469, 62.043447, 66.149439, 69.706374, 72.451659, 75.02984, 77.512533, 79.207448, 81.2127, 82.883743, 84.196706, 85.820005, 87.228455, 88.350442, 89.4963],
        #     'GNN - Predicted': [0, 1.718787, 22.797804, 39.770828, 52.733349, 64.168059, 71.425161, 76.390547, 80.377178, 82.955359, 85.270948, 87.562664, 88.756266, 89.61566, 90.642158, 91.47768, 92.480306, 93.148723, 93.912628, 94.390069, 94.891382]
        # }
        {
            'set': 'Fitted Set',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy': [0.00, 1.49, 32.71, 46.47, 56.51, 65.80, 72.12, 75.46, 78.07, 80.30, 82.16, 84.01, 86.25, 88.10, 89.22, 89.96, 90.33, 90.71, 91.08, 92.19, 92.57],
            'sampling_accuracy': [0.00, 5.58, 46.47, 66.54, 76.21, 84.76, 87.36, 90.33, 91.08, 92.19, 94.05, 94.05, 94.80, 94.80, 94.80, 95.91, 96.28, 97.03, 97.03, 97.03, 97.40],
            'random': [0.00, 1.12, 24.91, 38.29, 49.44, 57.62, 63.20, 67.29, 72.49, 75.46, 78.07, 81.04, 82.53, 83.64, 84.39, 85.13, 85.87, 86.99, 87.73, 88.10, 88.85],
            'XGBc': [0.00, 2.97, 28.62, 47.58, 57.99, 65.43, 71.00, 75.84, 79.55, 83.27, 86.25, 87.73, 89.22, 90.33, 90.71, 92.19, 92.19, 93.31, 93.68, 94.42, 94.42],
            'LRc': [0.00, 3.72, 32.71, 47.58, 57.99, 65.80, 70.26, 75.84, 79.55, 83.27, 85.13, 87.36, 89.22, 90.33, 90.33, 91.45, 91.82, 92.19, 92.57, 93.68, 94.42],
            'GNN - Predicted': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]
        },
        {
            'set': 'Astex Set',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy': [0.00, 2.90, 18.84, 31.88, 40.58, 49.64, 58.70, 66.67, 72.46, 73.55, 75.72, 77.17, 78.99, 81.52, 82.25, 84.06, 84.78, 86.96, 88.41, 89.86, 90.58],
            'sampling_accuracy': [0.00, 6.88, 28.99, 46.01, 59.06, 69.20, 76.09, 78.99, 81.16, 83.33, 86.23, 87.32, 88.41, 90.22, 90.58, 91.67, 93.48, 94.93, 96.01, 96.01, 96.01],
            'random': [0.00, 2.54, 17.03, 28.99, 38.41, 49.28, 56.52, 59.78, 61.59, 65.58, 69.93, 72.46, 73.55, 75.36, 76.81, 78.26, 78.99, 81.52, 83.33, 85.87, 86.59],
            'XGBc': [0.00, 3.62, 21.38, 32.61, 42.75, 56.16, 64.86, 69.20, 71.74, 76.45, 78.99, 79.71, 81.16, 82.25, 83.33, 84.42, 84.78, 86.96, 88.04, 89.86, 90.22],
            'LRc': [0.00, 1.81, 20.29, 32.97, 43.48, 53.99, 63.04, 70.29, 72.83, 75.72, 77.17, 78.99, 79.71, 81.88, 82.97, 84.78, 85.51, 86.23, 88.77, 90.22, 90.94],
            'GNN - Predicted': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]
        }
    ]

    def plot(dict):

        plt.rcParams.update({
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            # 'lines.markersize': 1,  # Dot size
            'axes.grid': True,
            'axes.axisbelow': True,
            'grid.color': 'gainsboro',
            'legend.fontsize': 11,  # Legend text size
            'text.color': 'black',  # All text in black
            'axes.labelcolor': 'black',  # Axes labels in black
            'xtick.color': 'black',  # X-axis ticks in black
            'ytick.color': 'black',  # Y-axis ticks in black
        })

        #dict['train'] = [i * 100 for i in dict['train']]
        #dict['test'] = [i * 100 for i in dict['test']]
        #dict['val'] = [i * 100 for i in dict['val']]

        plt.clf()
        ax.plot(dict['RMSD'], dict['scoring_accuracy'], color='blue')
        ax.plot(dict['RMSD'], dict['sampling_accuracy'], color='red')
        ax.plot(dict['RMSD'], dict['random'], color='darkorange')
        ax.plot(dict['RMSD'], dict['XGBc'], color='green')
        ax.plot(dict['RMSD'], dict['LRc'], color='green', linestyle='dashed')

        plt.title(dict['set'])
        plt.xlabel("RMSD (Å)")
        plt.ylabel("Accuracy (%)")
        plt.xticks(np.arange(0, 5.1, 0.5))
        plt.yticks(np.arange(0, 101, 10))
        plt.xlim(-0.1, 5.1)
        plt.ylim(-2.5, 102.5)

        plt.savefig('Prediction accuracy top 100' + dict['n_graph_layers'] + '.png')

    # plot(data[7])
    #for dict in data2:
    #    plot(dict)

    plt.rcParams.update({
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        #'lines.markersize': 1,  # Dot size
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': 'gainsboro',
        'legend.fontsize': 11,         # Legend text size
        'text.color': 'black',        # All text in black
        'axes.labelcolor': 'black',   # Axes labels in black
        'xtick.color': 'black',       # X-axis ticks in black
        'ytick.color': 'black',       # Y-axis ticks in black
    })
    plt.clf()

    def subplot(ax, dict):

        #dict['train'] = [i * 100 for i in dict['train']]
        #dict['test'] = [i * 100 for i in dict['test']]
        #dict['val'] = [i * 100 for i in dict['val']]

        ax.plot(dict['RMSD'], dict['scoring_accuracy'], color='blue')
        ax.plot(dict['RMSD'], dict['sampling_accuracy'], color='red')
        ax.plot(dict['RMSD'], dict['random'], color='darkorange')
        ax.plot(dict['RMSD'], dict['XGBc'], color='green')
        ax.plot(dict['RMSD'], dict['LRc'], color='green', linestyle='dashed')
        # if dict['set'] == 'PDBBind Set (train)' or dict['set'] == 'PDBBind Set (test)':
        #     ax.plot(dict['RMSD'], dict['GNN - Predicted'], color='green', linestyle='dotted')

        ax.set_xticks(np.arange(0, 5.1, 1))
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xlim(-0.1, 5.1)
        ax.set_ylim(-2.5, 102.5)
        #ax.xlabel("RMSD (Å)")
        #ax.ylabel("Accuracy (%)")
        ax.set_title(dict['set'], fontsize=12)

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(8.5, 4))
    for i, ax in enumerate(axs.flatten()):
        subplot(ax, regression[i])
        #subplot(ax, classifier[i])
        #subplot(ax, classifier_PDBBind[i])

    # for ax in axs.flat:
    #     ax.set(xlabel='epoch', ylabel='Accuracy (%)')
    # for ax in fig.get_axes():
    #     ax.label_outer()

    # Ensure x-tick values are shown on all subplots
    for ax in axs.flat:
        ax.tick_params(axis='x', labelbottom=True)
    #for ax in axs[:, 0]:  # Only the first column of subplots
    #    ax.set_ylabel('Accuracy (%)')
    axs[0].set_ylabel('Accuracy (%)')
    # Set the xlabel only on the bottom row subplots
    for ax in axs:  # Only the last row of subplots
        ax.set_xlabel('RMSD (Å)')

    # add legend
    #labels1 = ['Scoring accuracy', 'Sampling accuracy', 'Random accuracy']
    labels1 = ['Best energy', 'Best RMSD', 'Random']
    labels2 = ['_nolegend_','_nolegend_','_nolegend_','XGBr accuracy', 'LRr accuracy']
    #labels2 = ['_nolegend_', '_nolegend_', '_nolegend_', 'XGBc', 'LRc', 'GNN']
    fig.legend(labels=labels1, loc='lower center', bbox_to_anchor=(0.5, 0.06), ncol=3, frameon=False, bbox_transform=fig.transFigure)
    fig.legend(labels=labels2, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=False, bbox_transform=fig.transFigure)

    legend_handles = [
        Line2D([0], [0], color='b', lw=1, label='Line'),

        Patch(facecolor='blue', label='Scoring accuracy'),
        Patch(facecolor='red', label='Sampling accuracy'),
        Patch(facecolor='darkorange', label='Random'),
        Patch(facecolor='green', label='XGBc'),
        Patch(facecolor='green', label='LRc'),
    ]
    #fig.legend(handles=legend_handles, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)  # , , )#, title='Label',
    plt.tight_layout(rect=[0, 0.13, 1, 1])

    #plt.tight_layout()

    fig.savefig('Prediction_accuracy_classifier.png')


def plot_docking_accuracies_with_scaling_4x4():
    # python3 main.py --mode plotstuff

    data = [
        {
            'set': 'vdW of hydrophobic residues SC\nPDBBind set (train)',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy_1x':  [0.00, 1.19, 16.94, 30.13, 39.90, 47.63, 53.86, 58.25, 61.78, 65.10, 68.16, 70.43, 72.53, 74.31, 76.10, 77.97, 79.41, 80.72, 81.82, 83.16, 84.58],
            'sampling_accuracy_1x': [0.00, 2.95, 28.84, 47.48, 59.08, 67.09, 72.67, 76.68, 79.67, 82.48, 84.62, 86.38, 87.96, 89.47, 90.67, 91.72, 92.51, 93.48, 94.17, 94.87, 95.25],
            '0.5x':                 [0.00, 1.04, 15.92, 28.91, 38.65, 46.39, 52.58, 56.88, 60.73, 63.92, 66.82, 69.19, 71.37, 73.37, 75.12, 76.83, 78.23, 79.69, 80.98, 82.34, 83.86],
            '1.5x':                 [0.00, 1.31, 17.45, 30.90, 40.76, 48.65, 54.85, 59.31, 62.88, 66.23, 69.07, 71.23, 73.33, 75.05, 76.81, 78.52, 79.84, 81.37, 82.55, 83.90, 85.29],
            '2.0x':                 [0.00, 1.25, 17.85, 31.36, 41.33, 49.39, 55.47, 60.03, 63.48, 66.78, 69.63, 71.88, 73.90, 75.66, 77.51, 79.01, 80.29, 81.81, 82.98, 84.41, 85.68],
            '3.0x':                 [0.00, 1.26, 17.77, 30.95, 40.98, 49.24, 55.52, 60.04, 63.45, 66.66, 69.49, 71.71, 73.65, 75.48, 77.45, 78.97, 80.37, 81.84, 83.12, 84.37, 85.66],
            '4.0x':                 [0.00, 1.37, 17.57, 30.62, 40.50, 49.00, 54.97, 59.38, 62.89, 66.24, 69.08, 71.25, 73.30, 75.12, 77.01, 78.56, 80.02, 81.52, 82.80, 84.03, 85.42],
            '5.0x':                 [0.00, 1.31, 17.36, 30.05, 39.82, 48.49, 54.45, 58.85, 62.42, 65.84, 68.76, 70.88, 72.98, 74.88, 76.85, 78.35, 79.90, 81.31, 82.60, 83.84, 85.31],
            '6.0x':                 [0.00, 1.31, 17.29, 29.79, 39.53, 48.13, 54.26, 58.61, 62.19, 65.63, 68.66, 70.86, 72.79, 74.64, 76.65, 78.07, 79.58, 80.99, 82.35, 83.69, 85.03]
        },
        {
            'set': 'vdW of hydrophobic residues SC\nPDBBind set (test)',
            'RMSD':                 [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75,     2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy_1x':  [0.00, 1.61, 19.84, 34.43, 44.78, 53.48, 59.60, 64.58, 68.21, 70.89, 73.28, 75.19, 77.40, 78.61, 79.98, 81.73, 83.08, 84.01, 85.29, 86.59, 87.50],
            'sampling_accuracy_1x': [0.00, 4.49, 34.76, 54.69, 66.07, 73.79, 78.86, 81.66, 84.17, 86.24, 87.89, 89.33, 90.45, 91.29, 92.15, 92.68, 93.43, 93.85, 94.36, 94.78, 95.06],
            '0.5x':                 [0.00, 1.42, 18.87, 33.10, 43.46, 52.02, 57.88, 62.98, 66.60, 69.26, 71.72, 73.65, 75.65, 76.89, 78.24, 80.00, 81.40, 82.59, 83.87, 85.29, 86.33],
            '1.5x':                 [0.00, 1.77, 20.75, 35.15, 45.16, 53.93, 60.02, 64.91, 68.72, 71.33, 73.75, 75.75, 78.05, 79.28, 80.82, 82.42, 83.70, 84.73, 85.75, 86.89, 87.77],
            '2.0x':                 [0.00, 1.91, 20.77, 35.66, 45.53, 54.48, 60.79, 65.49, 69.30, 72.19, 74.61, 76.70, 78.91, 80.00, 81.52, 82.91, 84.24, 85.24, 86.29, 87.38, 88.33],
            '3.0x':                 [0.00, 1.88, 20.73, 35.52, 45.69, 54.39, 60.95, 65.58, 69.61, 72.51, 75.00, 77.26, 79.49, 80.87, 82.24, 83.56, 84.94, 86.05, 86.96, 88.05, 88.73],
            '4.0x':                 [0.00, 1.88, 21.03, 35.57, 45.53, 53.97, 60.53, 65.60, 69.65, 72.70, 75.07, 77.47, 79.80, 81.14, 82.68, 83.96, 85.24, 86.45, 87.33, 88.22, 88.98],
            '5.0x':                 [0.00, 1.79, 20.96, 35.24, 45.11, 53.81, 60.42, 65.49, 69.61, 72.54, 75.05, 77.63, 79.84, 81.26, 82.89, 84.26, 85.63, 86.82, 87.66, 88.45, 89.05],
            '6.0x':                 [0.00, 1.77, 20.89, 34.97, 44.88, 53.60, 60.32, 65.28, 69.40, 72.44, 74.98, 77.65, 79.96, 81.35, 82.87, 84.22, 85.49, 86.80, 87.59, 88.40, 89.03]
        },
        {
            'set': 'H-Bond of hydrophilic residues SC\nPDBBind set (train)',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy_1x':  [0.00, 1.19, 16.94, 30.13, 39.90, 47.63, 53.86, 58.25, 61.78, 65.10, 68.16, 70.43, 72.53, 74.31, 76.10, 77.97, 79.41, 80.72, 81.82, 83.16, 84.58],
            'sampling_accuracy_1x': [0.00, 2.95, 28.84, 47.48, 59.08, 67.09, 72.67, 76.68, 79.67, 82.48, 84.62, 86.38, 87.96, 89.47, 90.67, 91.72, 92.51, 93.48, 94.17, 94.87, 95.25],
            '0.5x':                 [0.00, 1.07, 16.06, 29.44, 39.19, 47.07, 53.24, 57.64, 61.20, 64.43, 67.30, 69.70, 71.74, 73.63, 75.48, 77.31, 78.92, 80.24, 81.63, 83.04, 84.45],
            '1.5x':                 [0.00, 1.28, 17.07, 29.94, 40.01, 47.67, 53.89, 58.31, 61.95, 65.13, 68.05, 70.16, 72.24, 74.01, 75.94, 77.79, 79.23, 80.63, 81.83, 83.09, 84.41],
            '2.0x':                 [0.00, 1.18, 16.72, 29.37, 39.20, 46.70, 52.79, 57.36, 61.14, 64.39, 67.24, 69.49, 71.52, 73.29, 75.16, 77.07, 78.57, 80.16, 81.45, 82.81, 84.08],
            '3.0x':                 [0.00, 1.15, 15.92, 27.93, 37.74, 45.32, 51.34, 55.62, 59.59, 63.10, 65.78, 68.22, 70.39, 72.27, 74.11, 75.98, 77.66, 79.31, 80.53, 81.94, 83.20]
        },
        {
            'set': 'H-Bond of hydrophilic residues SC\nPDBBind set (test)',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy_1x':  [0.00, 1.61, 19.84, 34.43, 44.78, 53.48, 59.60, 64.58, 68.21, 70.89, 73.28, 75.19, 77.40, 78.61, 79.98, 81.73, 83.08, 84.01, 85.29, 86.59, 87.50],
            'sampling_accuracy_1x': [0.00, 4.49, 34.76, 54.69, 66.07, 73.79, 78.86, 81.66, 84.17, 86.24, 87.89, 89.33, 90.45, 91.29, 92.15, 92.68, 93.43, 93.85, 94.36, 94.78, 95.06],
            '0.5x':                 [0.00, 1.51, 19.40, 33.85, 43.99, 52.88, 59.00, 63.95, 67.93, 70.51, 73.19, 75.24, 77.35, 78.61, 80.05, 81.73, 83.03, 84.24, 85.47, 86.66, 87.54],
            '1.5x':                 [0.00, 1.56, 19.98, 34.13, 44.13, 53.02, 59.16, 64.12, 67.72, 70.56, 72.96, 74.91, 76.86, 77.98, 79.49, 81.17, 82.49, 83.52, 84.82, 86.19, 87.12],
            '2.0x':                 [0.00, 1.33, 18.98, 33.08, 43.36, 52.27, 58.23, 63.23, 66.95, 69.72, 72.00, 74.28, 76.31, 77.47, 79.10, 80.70, 82.05, 83.12, 84.31, 85.70, 86.68],
            '3.0x':                 [0.00, 1.30, 18.10, 31.71, 41.87, 50.51, 56.76, 61.79, 65.46, 68.42, 70.91, 73.30, 75.37, 76.58, 78.17, 79.77, 81.21, 82.24, 83.45, 84.91, 85.91]
        },
        {
            'set': 'ES of hydrophilic residues BB\nPDBBind set (train)',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy_1x': [0.00, 1.19, 16.94, 30.13, 39.90, 47.63, 53.86, 58.25, 61.78, 65.10, 68.16, 70.43, 72.53, 74.31, 76.10, 77.97, 79.41, 80.72, 81.82, 83.16, 84.58],
            'sampling_accuracy_1x': [0.00, 2.95, 28.84, 47.48, 59.08, 67.09, 72.67, 76.68, 79.67, 82.48, 84.62, 86.38, 87.96, 89.47, 90.67, 91.72, 92.51, 93.48, 94.17, 94.87, 95.25],
            '0.5x': [0.00, 1.16, 16.84, 30.01, 39.80, 47.33, 53.74, 58.09, 61.66, 64.94, 68.01, 70.17, 72.32, 74.13, 75.86, 77.70, 79.14, 80.49, 81.64, 83.08, 84.46],
            '1.5x': [0.00, 1.16, 16.93, 30.46, 40.29, 47.92, 54.11, 58.55, 62.23, 65.65, 68.69, 70.92, 73.08, 74.81, 76.62, 78.33, 79.80, 81.05, 82.24, 83.54, 84.99],
            '2.0x': [0.00, 1.12, 16.85, 30.37, 40.12, 47.99, 54.16, 58.55, 62.17, 65.69, 68.80, 71.04, 73.09, 74.86, 76.78, 78.56, 79.97, 81.31, 82.51, 83.85, 85.27],
            '3.0x': [0.00, 1.04, 16.85, 30.30, 39.89, 48.10, 54.26, 58.81, 62.48, 65.90, 69.06, 71.40, 73.38, 75.08, 77.03, 78.79, 80.20, 81.57, 82.87, 84.23, 85.53]
        },
        {
            'set': 'ES of hydrophilic residues BB\nPDBBind set (test)',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy_1x': [0.00, 1.61, 19.84, 34.43, 44.78, 53.48, 59.60, 64.58, 68.21, 70.89, 73.28, 75.19, 77.40, 78.61, 79.98, 81.73, 83.08, 84.01, 85.29, 86.59, 87.50],
            'sampling_accuracy_1x': [0.00, 4.49, 34.76, 54.69, 66.07, 73.79, 78.86, 81.66, 84.17, 86.24, 87.89, 89.33, 90.45, 91.29, 92.15, 92.68, 93.43, 93.85, 94.36, 94.78, 95.06],
            '0.5x': [0.00, 1.54, 20.15, 34.17, 44.32, 53.20, 59.30, 64.30, 67.93, 70.54, 73.03, 75.05, 77.10, 78.35, 79.82, 81.52, 82.84, 83.80, 84.98, 86.33, 87.22],
            '1.5x': [0.00, 1.56, 19.89, 34.78, 45.02, 53.76, 59.81, 64.79, 68.37, 70.98, 73.35, 75.37, 77.52, 78.89, 80.40, 82.12, 83.38, 84.40, 85.59, 86.80, 87.70],
            '2.0x': [0.00, 1.49, 19.91, 34.57, 44.64, 53.83, 60.02, 64.95, 68.63, 71.30, 73.63, 75.68, 77.93, 79.28, 80.80, 82.45, 83.68, 84.63, 85.80, 87.15, 88.10],
            '3.0x': [0.00, 1.54, 20.03, 34.62, 44.90, 53.83, 59.90, 65.00, 68.40, 71.19, 73.58, 75.63, 77.86, 79.17, 80.73, 82.42, 83.66, 84.70, 85.82, 87.17, 88.08 ]
        }
            
            # },
            # {
            #     'set': 'ES (Hydrophobic residues BB)',
            #     'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            #     'scoring_accuracy_1x': [],
            #     'sampling_accuracy_1x': [],
            #     '0.5x': [],
            #     '1.5x': [],
            #     '2.0x': [],
            #     '3.0x': []


    ]

    data2 = [
        {
            'set': 'vdW of hydrophobic residues SC\nPDBBind set (full)',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy_1x':  [0.00, 1.44, 18.74, 32.59, 42.74, 51.10, 57.26, 61.96, 65.48, 68.58, 71.40, 73.57, 75.66, 77.22, 78.79, 80.52, 81.86, 83.07, 84.18, 85.46, 86.70],
            'sampling_accuracy_1x': [0.00, 3.74, 31.74, 51.23, 62.85, 70.92, 76.32, 79.75, 82.53, 85.03, 86.96, 88.60, 89.93, 91.09, 92.13, 92.96, 93.72, 94.44, 95.06, 95.67, 96.07],
            '0.5x': [0.00, 1.28, 17.83, 31.38, 41.55, 49.80, 55.88, 60.56, 64.25, 67.26, 70.05, 72.21, 74.28, 76.00, 77.56, 79.21, 80.56, 81.92, 83.17, 84.51, 85.82],
            '1.5x': [0.00, 1.54, 19.35, 33.41, 43.52, 51.94, 58.07, 62.78, 66.36, 69.46, 72.16, 74.31, 76.40, 77.93, 79.52, 81.12, 82.40, 83.74, 84.80, 86.05, 87.27],
            '2.0x': [0.00, 1.53, 19.63, 33.85, 43.91, 52.54, 58.66, 63.31, 66.87, 70.02, 72.75, 74.98, 76.96, 78.52, 80.14, 81.55, 82.83, 84.14, 85.24, 86.54, 87.68],
            '3.0x': [0.00, 1.53, 19.62, 33.67, 43.79, 52.43, 58.79, 63.44, 67.03, 70.13, 72.86, 75.12, 77.08, 78.75, 80.38, 81.81, 83.18, 84.51, 85.61, 86.77, 87.84],
            '4.0x': [0.00, 1.59, 19.53, 33.39, 43.44, 52.18, 58.36, 63.06, 66.74, 69.97, 72.62, 74.93, 77.00, 78.66, 80.31, 81.74, 83.09, 84.44, 85.55, 86.67, 87.82],
            '5.0x': [0.00, 1.49, 19.31, 32.97, 42.84, 51.71, 57.97, 62.63, 66.39, 69.57, 72.34, 74.69, 76.77, 78.47, 80.22, 81.65, 83.11, 84.39, 85.47, 86.57, 87.73],
            '6.0x': [0.00, 1.50, 19.23, 32.64, 42.52, 51.36, 57.73, 62.35, 66.11, 69.35, 72.19, 74.63, 76.65, 78.31, 80.06, 81.44, 82.84, 84.18, 85.30, 86.47, 87.56]
        },
        {
            'set': 'vdW of hydrophobic residues SC\nPDBBind set (val)',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy_1x':  [0.00, 1.93, 22.00, 35.89, 46.41, 55.85, 61.48, 66.42, 69.42, 72.48, 75.16, 77.41, 79.02, 80.47, 81.55, 82.56, 83.48, 84.82, 85.52, 86.43, 87.66],
            'sampling_accuracy_1x': [0.00, 4.83, 34.01, 54.56, 66.09, 74.73, 79.83, 82.30, 84.66, 86.80, 88.41, 89.97, 90.77, 91.20, 92.01, 92.60, 93.19, 93.62, 94.21, 94.85, 95.55],
            '0.5x': [0.00, 1.82, 21.67, 34.98, 45.76, 54.77, 60.52, 65.40, 68.40, 71.35, 74.25, 76.07, 77.74, 79.40, 80.58, 81.65, 82.56, 83.85, 84.82, 85.78, 86.86],
            '1.5x': [0.00, 1.82, 22.26, 37.02, 47.69, 56.81, 62.39, 67.33, 70.23, 73.28, 75.91, 78.22, 79.67, 81.06, 82.03, 83.10, 84.12, 85.41, 86.00, 87.07, 88.30],
            '2.0x': [0.00, 1.66, 22.64, 37.18, 47.42, 56.92, 62.39, 67.01, 70.17, 73.12, 75.97, 78.33, 79.51, 81.22, 82.14, 83.10, 84.23, 85.35, 86.21, 87.39, 88.41],
            '3.0x': [0.00, 1.61, 22.96, 37.77, 47.48, 56.97, 62.77, 67.65, 70.76, 73.66, 76.34, 78.65, 79.94, 81.55, 82.40, 83.64, 84.82, 86.00, 86.75, 87.66, 88.68],
            '4.0x': [0.00, 1.66, 22.42, 37.02, 47.32, 57.14, 62.88, 67.54, 70.76, 73.77, 76.13, 78.65, 80.10, 81.71, 82.62, 83.85, 84.87, 85.89, 86.75, 87.88, 88.95],
            '5.0x': [0.00, 1.39, 21.83, 36.96, 46.62, 56.12, 62.34, 66.74, 70.17, 72.85, 75.43, 78.00, 79.56, 81.06, 82.14, 83.32, 84.55, 85.46, 86.21, 87.39, 88.57],
            '6.0x': [0.00, 1.50, 21.73, 36.27, 46.03, 55.47, 61.59, 66.20, 69.64, 72.32, 74.95, 77.63, 79.24, 80.63, 81.81, 83.05, 84.17, 85.25, 86.16, 87.39, 88.52]
        },
        {
            'set': 'H-Bond of hydrophilic residues SC\nPDBBind set (full)',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy_1x':  [0.00, 1.44, 18.74, 32.59, 42.74, 51.10, 57.26, 61.96, 65.48, 68.58, 71.40, 73.57, 75.66, 77.22, 78.79, 80.52, 81.86, 83.07, 84.18, 85.46, 86.70],
            'sampling_accuracy_1x': [0.00, 3.74, 31.74, 51.23, 62.85, 70.92, 76.32, 79.75, 82.53, 85.03, 86.96, 88.60, 89.93, 91.09, 92.13, 92.96, 93.72, 94.44, 95.06, 95.67, 96.07],
            '0.5x': [0.00, 1.29, 18.00, 32.04, 42.12, 50.69, 56.82, 61.51, 65.17, 68.21, 71.03, 73.23, 75.26, 76.88, 78.50, 80.16, 81.60, 82.91, 84.14, 85.42, 86.64],
            '1.5x': [0.00, 1.47, 18.82, 32.42, 42.58, 50.91, 57.09, 61.77, 65.39, 68.49, 71.25, 73.35, 75.32, 76.83, 78.56, 80.23, 81.61, 82.92, 84.11, 85.37, 86.54],
            '2.0x': [0.00, 1.33, 18.23, 31.61, 41.74, 49.97, 56.09, 60.88, 64.63, 67.79, 70.48, 72.77, 74.74, 76.25, 77.99, 79.69, 81.09, 82.53, 83.68, 85.03, 86.20],
            '3.0x': [0.00, 1.28, 17.42, 30.20, 40.29, 48.42, 54.61, 59.28, 63.14, 66.44, 69.06, 71.49, 73.58, 75.16, 76.93, 78.62, 80.16, 81.56, 82.75, 84.12, 85.30]
        },
        {
            'set': 'H-Bond of hydrophilic residues SC\nPDBBind set (val)',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy_1x': [0.00, 1.93, 22.00, 35.89, 46.41, 55.85, 61.48, 66.42, 69.42, 72.48, 75.16, 77.41, 79.02, 80.47, 81.55, 82.56, 83.48, 84.82, 85.52, 86.43, 87.66],
            'sampling_accuracy_1x': [0.00, 4.83, 34.01, 54.56, 66.09, 74.73, 79.83, 82.30, 84.66, 86.80, 88.41, 89.97, 90.77, 91.20, 92.01, 92.60, 93.19, 93.62, 94.21, 94.85, 95.55],            
            '0.5x': [0.00, 1.66, 22.00, 35.84, 46.62, 56.28, 61.91, 66.58, 69.47, 72.32, 75.05, 77.25, 78.92, 80.31, 81.38, 82.40, 83.48, 84.76, 85.52, 86.53, 87.66],
            '1.5x': [0.00, 1.93, 21.73, 36.05, 46.24, 55.42, 61.11, 65.88, 69.21, 72.42, 75.21, 77.52, 78.92, 80.31, 81.60, 82.46, 83.64, 85.19, 86.00, 86.91, 88.04],
            '2.0x': [0.00, 1.82, 21.19, 34.87, 45.17, 54.13, 60.41, 65.29, 68.72, 72.21, 75.11, 77.41, 78.92, 80.10, 81.49, 82.46, 83.53, 85.14, 85.62, 86.75, 87.88],
            '3.0x': [0.00, 1.66, 20.55, 33.64, 43.94, 52.52, 58.85, 63.95, 67.54, 70.60, 73.18, 75.48, 77.15, 78.38, 80.15, 81.22, 82.40, 83.58, 84.50, 85.46, 86.64]
        },
        {
            'set': 'ES of hydrophilic residues BB\nPDBBind set (full)',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy_1x':  [0.00, 1.44, 18.74, 32.59, 42.74, 51.10, 57.26, 61.96, 65.48, 68.58, 71.40, 73.57, 75.66, 77.22, 78.79, 80.52, 81.86, 83.07, 84.18, 85.46, 86.70],
            'sampling_accuracy_1x': [0.00, 3.74, 31.74, 51.23, 62.85, 70.92, 76.32, 79.75, 82.53, 85.03, 86.96, 88.60, 89.93, 91.09, 92.13, 92.96, 93.72, 94.44, 95.06, 95.67, 96.07],
            '0.5x': [0.00, 1.36, 18.78, 32.44, 42.57, 50.90, 57.16, 61.81, 65.33, 68.36, 71.22, 73.36, 75.43, 77.02, 78.58, 80.28, 81.64, 82.87, 83.98, 85.35, 86.55],
            '1.5x': [0.00, 1.43, 18.80, 32.91, 43.11, 51.43, 57.52, 62.20, 65.81, 68.95, 71.73, 73.93, 76.04, 77.63, 79.24, 80.86, 82.21, 83.40, 84.53, 85.78, 87.02],
            '2.0x': [0.00, 1.36, 18.75, 32.82, 42.88, 51.53, 57.69, 62.32, 65.90, 69.13, 71.95, 74.14, 76.23, 77.85, 79.54, 81.17, 82.47, 83.68, 84.81, 86.12, 87.34],
            '3.0x': [0.00, 1.28, 18.84, 32.97, 42.91, 51.62, 57.76, 62.56, 66.02, 69.20, 72.11, 74.35, 76.38, 77.94, 79.64, 81.32, 82.61, 83.87, 85.02, 86.35, 87.51]
        },
        {
            'set': 'ES of hydrophilic residues BB\nPDBBind set (val)',
            'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            'scoring_accuracy_1x':  [0.00, 1.93, 22.00, 35.89, 46.41, 55.85, 61.48, 66.42, 69.42, 72.48, 75.16, 77.41, 79.02, 80.47, 81.55, 82.56, 83.48, 84.82, 85.52, 86.43, 87.66],
            'sampling_accuracy_1x': [0.00, 4.83, 34.01, 54.56, 66.09, 74.73, 79.83, 82.30, 84.66, 86.80, 88.41, 89.97, 90.77, 91.20, 92.01, 92.60, 93.19, 93.62, 94.21, 94.85, 95.55],            
            '0.5x': [0.00, 1.66, 22.00, 35.84, 46.62, 56.28, 61.91, 66.58, 69.47, 72.32, 75.05, 77.25, 78.92, 80.31, 81.38, 82.40, 83.48, 84.76, 85.52, 86.53, 87.66],
            '1.5x': [0.00, 2.09, 22.37, 36.05, 46.94, 56.44, 61.86, 66.52, 69.69, 72.80, 75.32, 77.63, 79.35, 80.79, 81.76, 82.73, 83.74, 85.03, 85.73, 86.75, 87.88],
            '2.0x': [0.00, 1.93, 22.26, 36.27, 46.83, 56.71, 62.39, 67.01, 69.96, 73.12, 75.75, 78.00, 79.72, 81.28, 82.35, 83.26, 84.23, 85.46, 86.16, 87.12, 88.14],
            '3.0x': [0.00, 1.56, 22.64, 37.45, 47.37, 56.92, 62.77, 67.54, 70.12, 73.12, 76.02, 78.17, 79.83, 81.33, 82.30, 83.53, 84.33, 85.62, 86.16, 87.29, 88.41]
        }
    ]

    def plot(dict):

        plt.rcParams.update({
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            # 'lines.markersize': 1,  # Dot size
            'axes.grid': True,
            'axes.axisbelow': True,
            'grid.color': 'gainsboro',
            'legend.fontsize': 11,  # Legend text size
            'text.color': 'black',  # All text in black
            'axes.labelcolor': 'black',  # Axes labels in black
            'xtick.color': 'black',  # X-axis ticks in black
            'ytick.color': 'black',  # Y-axis ticks in black
        })

        # dict['train'] = [i * 100 for i in dict['train']]
        # dict['test'] = [i * 100 for i in dict['test']]
        # dict['val'] = [i * 100 for i in dict['val']]

        plt.clf()
        ax.plot(dict['RMSD'], dict['scoring_accuracy'], color='blue')
        ax.plot(dict['RMSD'], dict['sampling_accuracy'], color='red')
        ax.plot(dict['RMSD'], dict['random'], color='darkorange')
        ax.plot(dict['RMSD'], dict['XGBc'], color='green')
        ax.plot(dict['RMSD'], dict['LRc'], color='green', linestyle='dashed')
        # if dict['set'] == 'PDBBind Set (train)' or dict['set'] == 'PDBBind Set (test)':
        # ax.plot(dict['RMSD'], dict['GNN - Predicted'], color='green', linestyle='dotted')

        plt.title(dict['set'])
        plt.xlabel("RMSD (Å)")
        plt.ylabel("Accuracy (%)")
        plt.xticks(np.arange(0, 5.1, 0.5))
        plt.yticks(np.arange(0, 101, 10))
        plt.xlim(-0.1, 5.1)
        plt.ylim(-2.5, 102.5)

        # plt.savefig('Prediction accuracy top 100' + dict['n_graph_layers'] + '.png')

    # plot(data[7])
    # for dict in data2:
    #    plot(dict)

    plt.rcParams.update({
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        # 'lines.markersize': 1,  # Dot size
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': 'gainsboro',
        'legend.fontsize': 11,  # Legend text size
        'text.color': 'black',  # All text in black
        'axes.labelcolor': 'black',  # Axes labels in black
        'xtick.color': 'black',  # X-axis ticks in black
        'ytick.color': 'black',  # Y-axis ticks in black
    })
    plt.clf()

    def subplot(ax, dict):

        # dict['train'] = [i * 100 for i in dict['train']]
        # dict['test'] = [i * 100 for i in dict['test']]
        # dict['val'] = [i * 100 for i in dict['val']]

        ax.plot(dict['RMSD'], dict['scoring_accuracy_1x'], zorder=1, color='blue') #zorder=1 makes the curve go to the back
        ax.plot(dict['RMSD'], dict['sampling_accuracy_1x'], color='red')
        ax.plot(dict['RMSD'], dict['0.5x'], color='yellowgreen', linestyle='dashed')
        ax.plot(dict['RMSD'], dict['1.5x'], color='turquoise', linestyle='dashed')
        ax.plot(dict['RMSD'], dict['2.0x'], color='coral', linestyle='dashed')
        ax.plot(dict['RMSD'], dict['3.0x'], color='violet', linestyle='dashed')
        if dict['set'] == 'vdW of hydrophobic residues SC\nPDBBind set (test)':
            ax.plot(dict['RMSD'], dict['4.0x'], color='olivedrab', zorder=2, linestyle='dashed')
            #ax.plot(dict['RMSD'], dict['5.0x'], color='blueviolet', linestyle='dashed')
        if dict['set'] == 'vdW of hydrophobic residues SC\nPDBBind set (train)':
            ax.plot(dict['RMSD'], dict['4.0x'], color='olivedrab', zorder=2, linestyle='dashed')
            #ax.plot(dict['RMSD'], dict['5.0x'], color='blueviolet', linestyle='dashed')
        if dict['set'] == 'vdW of hydrophobic residues SC\nPDBBind set (full)':
            ax.plot(dict['RMSD'], dict['4.0x'], color='olivedrab', zorder=2, linestyle='dashed')
            #ax.plot(dict['RMSD'], dict['5.0x'], color='blueviolet', linestyle='dashed')
        if dict['set'] == 'vdW of hydrophobic residues SC\nPDBBind set (val)':
            ax.plot(dict['RMSD'], dict['4.0x'], color='olivedrab', zorder=2, linestyle='dashed')
            #ax.plot(dict['RMSD'], dict['5.0x'], color='blueviolet', linestyle='dashed')

        ax.set_xticks(np.arange(0, 5.1, 1))
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xlim(-0.1, 5.1)
        ax.set_ylim(-2.5, 102.5)
        # ax.xlabel("RMSD (Å)")
        # ax.ylabel("Accuracy (%)")
        ax.set_title(dict['set'], fontsize=12)
        
        # Create inset axes
        ax_inset = inset_axes(ax, width="30%", height="48%",
                              loc="lower right",  # Places it in lower right by default
                              bbox_to_anchor=(-0.08, 0.15, 1.0, 1.0), # (x, y, width, height)
                              bbox_transform=ax.transAxes, 
                              borderpad=0)
        ax_inset.plot(dict['RMSD'], dict['scoring_accuracy_1x'], color='blue')
        ax_inset.plot(dict['RMSD'], dict['sampling_accuracy_1x'], color='red')
        ax_inset.plot(dict['RMSD'], dict['0.5x'], color='yellowgreen', linestyle='dashed')
        ax_inset.plot(dict['RMSD'], dict['1.5x'], color='turquoise', linestyle='dashed')
        ax_inset.plot(dict['RMSD'], dict['2.0x'], color='coral', linestyle='dashed')
        ax_inset.plot(dict['RMSD'], dict['3.0x'], color='violet', linestyle='dashed')
        if dict['set'] == 'vdW of hydrophobic residues SC\nPDBBind set (test)':
            ax_inset.plot(dict['RMSD'], dict['4.0x'], color='olivedrab', zorder=2, linestyle='dashed')
            #ax_inset.plot(dict['RMSD'], dict['5.0x'], color='blueviolet', linestyle='dashed')
        if dict['set'] == 'vdW of hydrophobic residues SC\nPDBBind set (train)':
            ax_inset.plot(dict['RMSD'], dict['4.0x'], color='olivedrab', zorder=2, linestyle='dashed')
            #ax_inset.plot(dict['RMSD'], dict['5.0x'], color='blueviolet', linestyle='dashed')
        if dict['set'] == 'vdW of hydrophobic residues SC\nPDBBind set (full)':
            ax_inset.plot(dict['RMSD'], dict['4.0x'], color='olivedrab', zorder=2, linestyle='dashed')
            # ax_inset.plot(dict['RMSD'], dict['5.0x'], color='blueviolet', linestyle='dashed')
        if dict['set'] == 'vdW of hydrophobic residues SC\nPDBBind set (val)':
            ax_inset.plot(dict['RMSD'], dict['4.0x'], color='olivedrab', zorder=2, linestyle='dashed')
            # ax_inset.plot(dict['RMSD'], dict['5.0x'], color='blueviolet', linestyle='dashed')

        # Zoomed-in limits
        ax_inset.set_xlim(1.9, 2.1)
        ax_inset.set_xticks([2.0])
        if dict['set'] == 'vdW of hydrophobic residues SC\nPDBBind set (train)':
            ax_inset.set_ylim(59, 66)
            ax_inset.set_yticks([60, 65])
        if dict['set'] == 'vdW of hydrophobic residues SC\nPDBBind set (test)':
            ax_inset.set_ylim(64, 71)
            ax_inset.set_yticks([65, 70])
        if dict['set'] == 'H-Bond of hydrophilic residues SC\nPDBBind set (train)':
            ax_inset.set_ylim(59, 66)
            ax_inset.set_yticks([60, 65])
        if dict['set'] == 'H-Bond of hydrophilic residues SC\nPDBBind set (test)':
            ax_inset.set_ylim(64, 71)
            ax_inset.set_yticks([65, 70])
        if dict['set'] == 'ES of hydrophilic residues BB\nPDBBind set (train)':
            ax_inset.set_ylim(59, 66)
            ax_inset.set_yticks([60, 65])
        if dict['set'] == 'ES of hydrophilic residues BB\nPDBBind set (test)':
            ax_inset.set_ylim(64, 71)
            ax_inset.set_yticks([65, 70])
            
        if dict['set'] == 'vdW of hydrophobic residues SC\nPDBBind set (full)':
            ax_inset.set_ylim(62, 69)
            ax_inset.set_yticks([63, 68])
        if dict['set'] == 'vdW of hydrophobic residues SC\nPDBBind set (val)':
            ax_inset.set_ylim(66, 73)
            ax_inset.set_yticks([67, 72])
        if dict['set'] == 'H-Bond of hydrophilic residues SC\nPDBBind set (full)':
            ax_inset.set_ylim(62, 69)
            ax_inset.set_yticks([63, 68])
        if dict['set'] == 'H-Bond of hydrophilic residues SC\nPDBBind set (val)':
            ax_inset.set_ylim(66, 73)
            ax_inset.set_yticks([67, 72])
        if dict['set'] == 'ES of hydrophilic residues BB\nPDBBind set (full)':
            ax_inset.set_ylim(62, 69)
            ax_inset.set_yticks([63, 68])
        if dict['set'] == 'ES of hydrophilic residues BB\nPDBBind set (val)':
            ax_inset.set_ylim(66, 73)
            ax_inset.set_yticks([67, 72])

        # Mark inset area
        #mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="black", linestyle="dotted")

    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=True, figsize=(8.5, 10))
    for i, ax in enumerate(axs.flatten()):
        subplot(ax, data2[i])

    # for ax in axs.flat:
    #     ax.set(xlabel='epoch', ylabel='Accuracy (%)')
    # for ax in fig.get_axes():
    #     ax.label_outer()

    # Ensure x-tick values are shown on all subplots
    for ax in axs.flat:
        ax.tick_params(axis='x', labelbottom=True)
    for ax in axs[:, 0]:  # Only the first column of subplots
        ax.set_ylabel('Accuracy (%)')
    # Set the xlabel only on the bottom row subplots
    for ax in axs[-1, :]:  # Only the last row of subplots
        ax.set_xlabel('RMSD (Å)')

    # add legend
    # labels1 = ['Scoring accuracy', 'Sampling accuracy', 'Random accuracy']
    labels1 = ['Best energy (no correction)', 'Best RMSD (no correction)']
    labels2 = ['_nolegend_','_nolegend_','0.5x', '1.5x', '2.0x', '3.0x', '4.0x (vdW only)']
    fig.legend(labels=labels1, loc='lower center', bbox_to_anchor=(0.5, 0.03), ncol=2, frameon=False, bbox_transform=fig.transFigure)
    fig.legend(labels=labels2, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=5, frameon=False, bbox_transform=fig.transFigure)

    legend_handles = [
        Line2D([0], [0], color='b', lw=1, label='Line'),

        Patch(facecolor='blue', label='Scoring accuracy'),
        Patch(facecolor='red', label='Sampling accuracy'),
        Patch(facecolor='darkorange', label='Random'),
        Patch(facecolor='green', label='XGBc'),
        Patch(facecolor='green', label='LRc'),
    ]
    # fig.legend(handles=legend_handles, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)  # , , )#, title='Label',
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    # plt.tight_layout()

    fig.savefig('Docking_accuracy_with_scaling_2x2_full_val.png')
def plot_docking_accuracies_with_scaling_2x1():
    data = [
        {
            'set': 'H-Bond of hydrophilic residues SC',
            'RMSD':                 [0.00, 0.25,  0.50,  0.75,  1.00,  1.25,  1.50,  1.75,  2.00,  2.25,  2.50,  2.75,  3.00,  3.25,  3.50,  3.75,  4.00,  4.25,  4.50,  4.75,  5.00],
            'scoring_accuracy_1x':  [0.00, 1.45, 18.80, 32.67, 42.86, 51.16, 57.35, 62.04, 65.55, 68.66, 71.50, 73.66, 75.73, 77.29, 78.86, 80.58, 81.92, 83.13, 84.24, 85.52, 86.77],
            'sampling_accuracy_1x': [0.00, 3.73, 31.78, 51.30, 62.88, 70.97, 76.35, 79.80, 82.56, 85.06, 87.00, 88.64, 89.97, 91.11, 92.16, 92.98, 93.74, 94.47, 95.08, 95.68, 96.08],
            '0.5x':                 [0.00, 1.30, 17.77, 31.30, 41.51, 49.78, 55.92, 60.64, 64.30, 67.29, 70.09, 72.22, 74.24, 75.93, 77.50, 79.16, 80.50, 81.83, 83.08, 84.45, 85.77],
            '1.5x':                 [0.00, 1.58, 19.40, 33.43, 43.57, 52.01, 58.24, 62.95, 66.50, 69.55, 72.29, 74.39, 76.48, 77.96, 79.56, 81.17, 82.44, 83.74, 84.78, 86.05, 87.25],
            '2.0x':                 [0.00, 1.53, 19.64, 33.81, 43.91, 52.58, 58.81, 63.45, 66.99, 70.11, 72.85, 75.05, 77.03, 78.56, 80.19, 81.63, 82.90, 84.19, 85.28, 86.58, 87.69],
            '3.0x':                 [0.00, 1.50, 19.63, 33.64, 43.75, 52.47, 58.92, 63.56, 67.17, 70.25, 72.99, 75.21, 77.20, 78.86, 80.49, 81.94, 83.28, 84.57, 85.65, 86.81, 87.87],
            '4.0x':                 [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
            '5.0x':                 [0.00, 0.25, 1.50, 1.75, 3.00, 3.25, 4.50, 5.75, 6.00, 7.25, 8.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]
        },
        {
            'set': 'vdW of hydrophobic residues SC',
            'RMSD':                 [0.00, 0.25,  0.50,  0.75,  1.00,  1.25,  1.50,  1.75,  2.00,  2.25,  2.50,  2.75,  3.00,  3.25,  3.50,  3.75,  4.00,  4.25,  4.50,  4.75, 5.00],
            'scoring_accuracy_1x':  [0.00, 1.45, 18.80, 32.67, 42.86, 51.16, 57.35, 62.04, 65.55, 68.66, 71.50, 73.66, 75.73, 77.29, 78.86, 80.58, 81.92, 83.13, 84.24, 85.52, 86.77],
            'sampling_accuracy_1x': [0.00, 3.73, 31.78, 51.30, 62.88, 70.97, 76.35, 79.80, 82.56, 85.06, 87.00, 88.64, 89.97, 91.11, 92.16, 92.98, 93.74, 94.47, 95.08, 95.68, 96.08],
            '0.5x':                 [0.00, 1.31, 18.03, 32.01, 42.12, 50.81, 57.02, 61.73, 65.38, 68.42, 71.22, 73.37, 75.37, 76.93, 78.54, 80.22, 81.65, 82.95, 84.14, 85.44, 86.66],
            '1.5x':                 [0.00, 1.50, 18.79, 32.43, 42.66, 51.04, 57.28, 61.98, 65.52, 68.61, 71.39, 73.44, 75.38, 76.87, 78.60, 80.30, 81.66, 82.94, 84.12, 85.42, 86.58],
            '2.0x':                 [0.00, 1.32, 18.18, 31.59, 41.72, 50.01, 56.20, 61.02, 64.74, 67.89, 70.62, 72.88, 74.83, 76.33, 78.07, 79.82, 81.19, 82.61, 83.72, 85.09, 86.25],
            '3.0x':                 [0.00, 1.28, 17.36, 30.15, 40.21, 48.37, 54.64, 59.33, 63.16, 66.46, 69.12, 71.53, 73.59, 75.19, 76.96, 78.68, 80.21, 81.58, 82.73, 84.13, 85.31]    
        # },
        # {
        #     'set': 'ES (Hydrophobic residues BB)',
        #     'RMSD': [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
        #     'scoring_accuracy_1x': [],
        #     'sampling_accuracy_1x': [],
        #     '0.5x': [],
        #     '1.5x': [],
        #     '2.0x': [],
        #     '3.0x': []

        }
        
    ]

    def plot(dict):

        plt.rcParams.update({
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            # 'lines.markersize': 1,  # Dot size
            'axes.grid': True,
            'axes.axisbelow': True,
            'grid.color': 'gainsboro',
            'legend.fontsize': 11,  # Legend text size
            'text.color': 'black',  # All text in black
            'axes.labelcolor': 'black',  # Axes labels in black
            'xtick.color': 'black',  # X-axis ticks in black
            'ytick.color': 'black',  # Y-axis ticks in black
        })

        #dict['train'] = [i * 100 for i in dict['train']]
        #dict['test'] = [i * 100 for i in dict['test']]
        #dict['val'] = [i * 100 for i in dict['val']]

        plt.clf()
        ax.plot(dict['RMSD'], dict['scoring_accuracy'], color='blue')
        ax.plot(dict['RMSD'], dict['sampling_accuracy'], color='red')
        ax.plot(dict['RMSD'], dict['random'], color='darkorange')
        ax.plot(dict['RMSD'], dict['XGBc'], color='green')
        ax.plot(dict['RMSD'], dict['LRc'], color='green', linestyle='dashed')

        plt.title(dict['set'])
        plt.xlabel("RMSD (Å)")
        plt.ylabel("Accuracy (%)")
        plt.xticks(np.arange(0, 5.1, 0.5))
        plt.yticks(np.arange(0, 101, 10))
        plt.xlim(-0.1, 5.1)
        plt.ylim(-2.5, 102.5)

        plt.savefig('Prediction accuracy top 100' + dict['n_graph_layers'] + '.png')

    # plot(data[7])
    #for dict in data2:
    #    plot(dict)

    plt.rcParams.update({
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        #'lines.markersize': 1,  # Dot size
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': 'gainsboro',
        'legend.fontsize': 11,         # Legend text size
        'text.color': 'black',        # All text in black
        'axes.labelcolor': 'black',   # Axes labels in black
        'xtick.color': 'black',       # X-axis ticks in black
        'ytick.color': 'black',       # Y-axis ticks in black
    })
    plt.clf()

    def subplot(ax, dict):

        #dict['train'] = [i * 100 for i in dict['train']]
        #dict['test'] = [i * 100 for i in dict['test']]
        #dict['val'] = [i * 100 for i in dict['val']]

        ax.plot(dict['RMSD'], dict['scoring_accuracy_1x'], zorder=1, color='blue') #zorder=1 makes the curve go to the back
        ax.plot(dict['RMSD'], dict['sampling_accuracy_1x'], color='red')
        ax.plot(dict['RMSD'], dict['0.5x'], color='yellowgreen', linestyle='dashed')
        ax.plot(dict['RMSD'], dict['1.5x'], color='turquoise', linestyle='dashed')
        ax.plot(dict['RMSD'], dict['2.0x'], color='coral', linestyle='dashed')
        ax.plot(dict['RMSD'], dict['3.0x'], color='violet', linestyle='dashed')
        # if dict['set'] == 'H-Bond of hydrophilic residues SC':
        #     ax.plot(dict['RMSD'], dict['4.0x'], color='mediumvioletred', linestyle='dashed')
        # if dict['set'] == 'H-Bond of hydrophilic residues SC':
        #     ax.plot(dict['RMSD'], dict['5.0x'], color='coral', linestyle='dashed')

        ax.set_xticks(np.arange(0, 5.1, 1))
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xlim(-0.1, 5.1)
        ax.set_ylim(-2.5, 102.5)
        #ax.xlabel("RMSD (Å)")
        #ax.ylabel("Accuracy (%)")
        ax.set_title(dict['set'], fontsize=12)

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(8.5, 4))
    for i, ax in enumerate(axs.flatten()):
        subplot(ax, data[i])
        #subplot(ax, classifier[i])
        #subplot(ax, classifier_PDBBind[i])

    # for ax in axs.flat:
    #     ax.set(xlabel='epoch', ylabel='Accuracy (%)')
    # for ax in fig.get_axes():
    #     ax.label_outer()

    # Ensure x-tick values are shown on all subplots
    for ax in axs.flat:
        ax.tick_params(axis='x', labelbottom=True)
    #for ax in axs[:, 0]:  # Only the first column of subplots
    #    ax.set_ylabel('Accuracy (%)')
    axs[0].set_ylabel('Accuracy (%)')
    # Set the xlabel only on the bottom row subplots
    for ax in axs:  # Only the last row of subplots
        ax.set_xlabel('RMSD (Å)')

    # add legend
    #labels1 = ['Scoring accuracy', 'Sampling accuracy', 'Random accuracy']
    labels1 = ['Best energy (no correction)', 'Best RMSD (no correction)']
    labels2 = ['_nolegend_','_nolegend_','0.5x', '1.5x', '2.0x', '3.0x']#, '4.0x', '5.0x']
    #labels2 = ['_nolegend_', '_nolegend_', '_nolegend_', 'XGBc', 'LRc', 'GNN']
    fig.legend(labels=labels1, loc='lower center', bbox_to_anchor=(0.5, 0.06), ncol=2, frameon=False, bbox_transform=fig.transFigure)
    fig.legend(labels=labels2, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4, frameon=False, bbox_transform=fig.transFigure)

    legend_handles = [
        Line2D([0], [0], color='b', lw=1, label='Line'),

        Patch(facecolor='blue', label='Scoring accuracy'),
        Patch(facecolor='red', label='Sampling accuracy'),
        Patch(facecolor='darkorange', label='Random'),
        Patch(facecolor='green', label='XGBc'),
        Patch(facecolor='green', label='LRc'),
    ]
    #fig.legend(handles=legend_handles, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)  # , , )#, title='Label',
    plt.tight_layout(rect=[0, 0.13, 1, 1])

    #plt.tight_layout()

    fig.savefig('Docking_accuracy_with_scaling.png')

def plot_PDBBind_accuracy_hardcoded():

    xaxis = [0.00,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00,2.25,2.50,2.75,3.00,3.25,3.50,3.75,4.00,4.25,4.50,4.75,5.00]
    plt.clf()

    plt.rcParams.update({
        'axes.titlesize': 14,  # Title font size
        'axes.titleweight': 'bold',  # Title font weight
        'axes.labelsize': 12,  # X and Y axis labels font size
        'legend.fontsize': 12,  # Legend font sizee
        'xtick.labelsize': 12,  # X-axis tick labels font size
        'ytick.labelsize': 12,  # Y-axis tick labels font size
        'axes.grid': True,
        #'axes.grid.which': 'both',
        'axes.axisbelow': True,
        'grid.color': 'gainsboro',
    })
    plt.figure(figsize=(8, 5))

    # PDBBind set without water
    # lowest_energy = [0.0,1.2,17.1,30.7,40.8,49.6,56.0,61.2,64.8,68.1,70.8,73.2,75.1,77.0,78.5,80.0,81.4,82.8,83.9,85.1,86.6]
    # lowest_rmsd =   [0.0,3.6,30.4,49.2,61.1,69.4,74.9,78.8,81.6,84.2,86.2,87.9,89.2,90.5,91.5,92.3,93.2,94.0,94.7,95.2,95.8]
    # random =        [0.0,1.1,14.3,26.0,34.6,43.1,49.7,55.1,59.4,63.1,66.4,69.0,71.3,73.5,75.5,77.3,78.7,80.2,81.5,82.9,84.5]
    # XGBc_pred_10 =  [0.000000,0.752559,8.940397,16.897451,23.550070,29.530403,35.199679,39.955850,44.561509,48.575156,52.267710,55.508730,57.997190,60.646197,62.944010,65.101345,67.288782,69.466185,71.894441,73.921333,75.908087] # XGBc prediction on picking the right pose (based on subtracted pairs) out of 10 runs on the PDBBind set
    # LRc_pred_10 =   [0.000000,0.692354,9.241421,17.780454,24.924744,31.858318,37.738310,42.614891,47.371062,51.575356,54.996990,58.238009,60.887016,63.375477,65.723460,67.961068,69.997993,71.964680,74.061810,76.018463,77.894842] # LRc prediction on picking the right pose (based on subtracted pairs) out of 10 runs on the PDBBind set
    # XGBc_pred_10_flipped = [0.000000,1.334537,17.820590,32.831628,43.859121,53.401565,60.545856,66.084688,70.238812,73.640377,76.128838,78.496889,80.383303,82.189444,83.594220,84.657837,85.791692,86.754967,87.587799,88.380494,89.253462]
    # LRc_pred_10_flipped =  [0.000000,1.244230,17.098134,30.443508,40.949227,49.719045,56.492073,61.629540,65.171583,68.753763,71.523179,74.001605,75.767610,77.453341,78.908288,80.353201,81.617499,82.891832,84.126028,85.420429,86.524182]

    # lowest_energy =         [0.000000,1.104315,16.955488,30.479103,40.706762,49.541284,55.844376,61.034659,64.823310,68.025824,70.820591,73.182127,75.254842,77.081210,78.550799,80.147808,81.498471,82.934081,84.089365,85.227659,86.561332]
    # lowest_rmsd =           [0.000000,3.491335,30.045872,49.014611,60.635406,69.257560,74.838600,78.822630,81.719334,84.318722,86.340469,87.954468,89.313626,90.511383,91.573225,92.422698,93.255182,94.053687,94.758750,95.293918,95.846075]
    # random =                [0.000000,1.087326,13.829426,25.382263,34.616038,43.442066,49.940537,55.096840,59.522596,62.962963,66.156983,68.671424,70.769623,72.740401,74.711179,76.427115,77.973157,79.570166,81.107713,82.670744,83.843017]
    # XGBc_pred_10_all =      [0.000000,1.333673,17.889908,32.679239,43.637445,53.397893,60.788311,66.309888,70.268434,73.428474,76.078831,78.414883,80.241250,81.855250,83.248386,84.548080,85.694869,86.671764,87.521237,88.379205,89.203194]
    # XGBc_pred_10_test =     [0.000000,1.343813,19.371197,34.964503,45.360041,55.856998,63.717039,69.345842,73.047667,76.166329,78.549696,81.085193,83.062880,84.939148,86.561866,87.652130,88.666329,89.579108,90.415822,91.480730,92.241379]
    # XGBc_pred_10_trainval = [0.000000,1.341339,17.296883,31.361778,42.667348,52.069494,59.172202,64.716403,68.855391,71.934083,74.782831,77.018396,78.755749,80.314257,81.668370,83.022483,84.274400,85.283597,86.177823,86.905979,87.749106]
    # XGBc_pred_10_train =  [ 0.000000, 1.162212,16.237755,30.018263,40.976258,50.058111,57.380043,63.191101,67.358459,70.513033,73.252532,75.477337,77.287066,79.063590,80.624274,82.085340,83.529802,84.525984,85.505562,86.236095,87.165864]
    # LRc_pred_10_all =       [0.000000,1.112810,17.074414,30.615019,41.038056,49.915053,56.345566,61.748216,65.392457,68.934760,71.839959,74.142032,75.993884,77.726809,79.306830,80.827387,82.101597,83.358818,84.599049,85.796806,86.892627]
    # LRc_pred_10_test =      [0.000000,1.039554,18.204868,32.530426,43.204868,53.067951,59.381339,65.491886,68.940162,72.363083,74.949290,77.484787,79.665314,81.490872,83.012170,84.558824,85.649087,86.511156,87.905680,89.173428,90.035497]
    # LRc_pred_10_trainval =  [0.000000,1.149719,16.504854,29.649974,39.946346,48.326520,54.816045,59.862034,63.605008,67.207460,70.273378,72.457844,74.144098,75.830353,77.439959,78.947368,80.314257,81.770567,82.933061,84.095554,85.309147]

    # PDBBind set with water
    lowest_energy = [0.000000,1.309614,18.455715,32.528388,42.581378,50.968963,57.259652,61.688115,65.352006,68.334595,71.150643,73.323240,75.382286,76.926571,78.607116,80.242241,81.680545,82.967449,84.239213,85.518547,86.691900]
    lowest_rmsd =   [0.000000,3.830431,31.574565,51.052233,62.588948,70.802422,76.230129,79.947010,82.619228,85.049205,86.994701,88.599546,89.954580,91.067373,92.089326,92.922029,93.701741,94.451173,95.094625,95.753217,96.146858]
    random =        [0.000000,1.143073,15.155185,26.971991,36.063588,44.473883,50.968963,56.033308,59.803179,63.580621,66.691900,69.326268,71.483724,73.580621,75.541257,77.191522,78.894777,80.355791,81.801665,83.361090,84.663134]

    # Fitted Set
    # lowest_energy = [0.000000,1.486989,32.713755,46.468401,56.505576,65.799257,72.118959,75.464684,78.066914,80.297398,82.156134,84.014870,86.245353,88.104089,89.219331,89.962825,90.334572,90.706320,91.078067,92.193309,92.565056]
    # lowest_rmsd =   [0.000000,5.576208,46.468401,66.542751,76.208178,84.758364,87.360595,90.334572,91.078067,92.193309,94.052045,94.052045,94.795539,94.795539,94.795539,95.910781,96.282528,97.026022,97.026022,97.026022,97.397770]
    # random =        [0.000000,2.230483,26.765799,39.405204,47.955390,56.877323,62.825279,67.657993,72.118959,74.721190,78.810409,82.527881,84.758364,85.873606,86.617100,86.988848,88.104089,88.847584,89.591078,90.334572,91.449814]
    # XGBc =          [0.000000,1.486989,29.739777,47.211896,59.107807,68.773234,74.721190,78.066914,82.156134,83.643123,87.360595,88.475836,91.078067,92.193309,92.193309,92.936803,92.936803,94.052045,94.423792,95.910781,95.910781]
    # LRc =           [0.000000,1.858736,31.226766,48.698885,56.505576,66.542751,71.747212,75.464684,79.925651,82.527881,84.386617,85.501859,87.732342,88.475836,88.475836,89.591078,89.591078,89.962825,90.334572,91.821561,92.565056]

    # Astex Set
    # lowest_energy = [0.000000,2.898551,18.840580,31.884058,40.579710,49.637681,58.695652,66.666667,72.463768,73.550725,75.724638,77.173913,78.985507,81.521739,82.246377,84.057971,84.782609,86.956522,88.405797,89.855072,90.579710]
    # lowest_rmsd =   [0.000000,6.884058,28.985507,46.014493,59.057971,69.202899,76.086957,78.985507,81.159420,83.333333,86.231884,87.318841,88.405797,90.217391,90.579710,91.666667,93.478261,94.927536,96.014493,96.014493,96.014493]
    # random =        [0.000000,3.623188,18.840580,30.072464,39.130435,47.463768,52.536232,56.521739,60.507246,63.043478,66.666667,67.753623,68.840580,69.927536,72.463768,75.000000,77.536232,79.710145,81.521739,82.246377,83.695652]
    # XGBc =          [0.000000,1.811594,21.376812,34.782609,43.115942,52.173913,62.681159,69.565217,71.376812,74.275362,77.173913,78.985507,81.159420,82.246377,82.971014,83.695652,84.057971,86.594203,87.681159,88.768116,89.130435]
    # LRc =           [0.000000,2.173913,18.840580,31.159420,40.942029,52.173913,62.318841,68.478261,73.550725,75.362319,77.536232,78.623188,80.434783,82.246377,83.333333,84.420290,85.507246,86.956522,88.768116,89.855072,90.942029]

    # Fitted paper - Fitted Set
    # x  = [0.0, 0.25,  0.5, 0.75,  1.0, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.50, 4.00, 4.50, 5.00 ] #2.75, 4.5,
    # y1 = [0.0, 11.8, 48.9, 66.6, 76.4, 83.6, 88.5, 92.5, 93.4, 94.4, 95.1, 95.4, 96.4, 96.4, 97.7, 98.4, 98.7 ] # Best RMSD
    # y3 = [0.0,  1.6, 29.5, 42.0, 56.4, 62.6, 68.2, 73.1, 76.7, 81.0, 83.0, 84.3, 86.9, 89.5, 90.5, 91.8, 93.4 ] # Best energy

    # plt.plot(x, y1, label="Lowest energy", color='blue')
    # plt.plot(x, y3, label="Lowest RMSD", color='red')

    plt.plot(xaxis, lowest_energy, label="Scoring accuracy", color='blue')
    plt.plot(xaxis, lowest_rmsd, label="Sampling accuracy", color='red')
    #plt.plot(xaxis, random, label="Random", color='orange')
    # plt.plot(xaxis, XGBc, label="XGBc", color='green')
    # plt.plot(xaxis, LRc, label="LRc", color='green', linestyle='dashed')

    # Penalty scaling factor 1.5x on hydrophobic residues
    # lowest_energy = [0.0,1.2,17.1,30.7,40.8,49.6,56.0,61.2,64.8,68.1,70.8,73.2,75.1,77.0,78.5,80.0,81.4,82.8,83.9,85.1,86.6]
    # lowest_rmsd =   [0.0,3.6,30.4,49.2,61.1,69.4,74.9,78.8,81.6,84.2,86.2,87.9,89.2,90.5,91.5,92.3,93.2,94.0,94.7,95.2,95.8]
    # random =        [0.0,1.1,14.3,26.0,34.6,43.1,49.7,55.1,59.4,63.1,66.4,69.0,71.3,73.5,75.5,77.3,78.7,80.2,81.5,82.9,84.5]
    # plt.plot(xaxis, lowest_energy, label="Lowest energy (1.5x hydrophobic)", color='blue', linestyle='dashed')
    # plt.plot(xaxis, lowest_rmsd, label="Lowest RMSD (1.5x hydrophobic)", color='red', linestyle='dashed')
    # plt.plot(xaxis, random, label="Random (1.5x hydrophobic)", color='orange', linestyle='dashed')

    # Scaling factor 2.5x on hydrophobic residues (7)
    lowest_energy_vdW25x = [0.000000,1.404921,18.833536,32.533414,42.907047,51.344168,58.027035,62.659478,66.167983,69.190462,71.810450,74.172236,76.336574,78.128797,79.814702,81.158870,82.533414,83.961118,85.145808,86.353281,87.439247]
    lowest_rmsd_vdW25x =   [0.000000,4.002126,32.199271,51.291009,63.031592,71.240887,76.313791,79.890644,82.730863,85.039490,86.945626,88.585966,89.808627,90.993317,92.086877,92.891859,93.712029,94.311968,94.919502,95.428311,95.883961]
    random_vdW25x =        [0.000000,1.351762,15.871810,28.303463,37.705043,46.013062,52.559235,57.670109,61.960814,65.211118,68.241191,70.876367,73.200182,75.174666,77.020049,78.614824,80.126063,81.690462,83.042224,84.257290,85.487546]
    plt.plot(xaxis, lowest_energy_vdW25x, label="Scoring accuracy\n(hydrophobic\nvdW 2.5x)", color='blue', linestyle='dotted')
    #plt.plot(xaxis, lowest_rmsd_vdW25x, label="vdW 2.5x", color='red', linestyle='dotted')
    #plt.plot(xaxis, random_vdW25x, label="2.5x", color='orange', linestyle='dotted')

    # # Penalty scaling factor 3x on hydrophobic residues
    # lowest_energy = [0.0,1.2,17.1,30.7,40.8,49.6,56.0,61.2,64.8,68.1,70.8,73.2,75.1,77.0,78.5,80.0,81.4,82.8,83.9,85.1,86.6]
    # lowest_rmsd =   [0.0,3.6,30.4,49.2,61.1,69.4,74.9,78.8,81.6,84.2,86.2,87.9,89.2,90.5,91.5,92.3,93.2,94.0,94.7,95.2,95.8]
    # random =        [0.0,1.1,14.3,26.0,34.6,43.1,49.7,55.1,59.4,63.1,66.4,69.0,71.3,73.5,75.5,77.3,78.7,80.2,81.5,82.9,84.5]
    # plt.plot(xaxis, lowest_energy, label="Lowest energy (1.5x hydrophobic)", color='blue', linestyle='dashdot')
    # plt.plot(xaxis, lowest_rmsd, label="Lowest RMSD (1.5x hydrophobic)", color='red', linestyle='dashdot')
    # plt.plot(xaxis, random, label="Random (1.5x hydrophobic)", color='orange', linestyle='dashdot')

    plt.title("PDBBind Set")
    plt.xlabel("RMSD (Å)")
    plt.ylabel("Accuracy (%)")
    plt.xticks(np.arange(0, 5.1, 0.5))
    plt.yticks(np.arange(0, 101, 10))
    plt.xlim(-0.1, 5.1)
    plt.ylim(-2.5, 102.5)
    plt.legend(loc='lower right') #, ncol=2)
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    plt.savefig('Accuracy_vdW25x.png')

def plot_set_overlap():

    # Plot the overlap between PDBBind train, val, test, Fitted and Astex

    PDBBind_list = []
    trainval_list = []
    train_list = []
    test_list = []
    val_list = []
    Fitted_list = []
    Astex_list = []

    with open('List_set_PDBBind/list_all_clean.txt', 'rb') as list_f:
        for line in list_f:
            PDBBind_list.append(line.strip().decode('UTF-8'))
    with open('List_set_PDBBind/list_trainval_clean.txt', 'rb') as list_f:
        for line in list_f:
            trainval_list.append(line.strip().decode('UTF-8'))
    with open('List_set_PDBBind/list_trainset_clean.txt', 'rb') as list_f:
        for line in list_f:
            train_list.append(line.strip().decode('UTF-8'))
    with open('List_set_PDBBind/list_testset_clean.txt', 'rb') as list_f:
        for line in list_f:
            test_list.append(line.strip().decode('UTF-8'))
    with open('List_set_PDBBind/list_valset_clean.txt', 'rb') as list_f:
        for line in list_f:
            val_list.append(line.strip().decode('UTF-8'))
    with open('FittedSet_top10/Fitted_list_pdb_full.txt', 'rb') as list_f:
        for line in list_f:
            Fitted_list.append(line.strip().decode('UTF-8'))
    with open('AstexSet_top10/Astex_list_full_305.txt', 'rb') as list_f:
        for line in list_f:
            Astex_list.append(line.strip().decode('UTF-8'))

    c = set(PDBBind_list)    # PDBBind
    #c = set(trainval_list)  # train/val
    #a = set(train_list)        # train
    #b = set(test_list)          # test
    #c = set(val_list)            # val
    a = set(Fitted_list)      # Fitted
    b = set(Astex_list)        # Astex

    only_a = len(a - b - c)
    only_b = len(b - a - c)
    only_c = len(c - a - b)
    only_a_b = len(a & b - c)
    only_a_c = len(a & c - b)
    only_b_c = len(b & c - a)
    a_b_c = len(a & b & c)

    labels = ['Fitted Set', 'Astex Set', 'PDBBind Set']
    #labels = ['PDBBind\n(train/val set)', 'Fitted Set', 'PDBBind\n(test set)']
    #labels = ['train set', 'val set', 'test set']
    venn3(subsets=(only_a, only_b, only_a_b, only_c, only_a_c, only_b_c, a_b_c), set_labels=labels)
    venn3_circles(subsets=(only_a, only_b, only_a_b, only_c, only_a_c, only_b_c, a_b_c), linestyle='dashed', linewidth=1, color="grey")
    plt.savefig('Set_overlap_pdb_fitted_astex.png')

    plt.clf()
    sets = {
        'Astex Set': set(Astex_list),
        'PDBBind (train set)': set(train_list),
        'PDBBind (test set)': set(test_list),
        'PDBBind (val set)': set(val_list)}

    #venny4py(sets=sets)

def plot_stuff_for_ML1():

    def plot_sigmoid():
        # Sigmoid coefficients
        k = -0.1  # k is the slope
        a = 2.25 # a is the horizontal shift
        b = 1 # b is the vertical range

        def sigmoid(x):
            # sigmoid function
            # use k to adjust the slope
            s = b / (1 + np.exp(-(x - a) / k))
            return s

        def act_to_RMSD(y):
            RMSD = -k * np.log(b / y - 1) + a
            return RMSD

        plt.rcParams.update({
            'axes.titlesize': 12,
            'font.size': 10,
            'lines.linewidth': 2.0,
            'axes.grid': True,
            'axes.axisbelow': True,
            'grid.color': 'gainsboro',
        })
        plt.clf()
        fig, ax = plt.subplots()

        # Graph settings
        # plt.rcParams.update({
        #     "text.usetex": True,
        #     "font.family": "Helvetica",
        #     "font.size": "15"
        # })

        font1 = {'size': 20}
        font2 = {'size': 15}

        # Generate x values
        x = np.linspace(0, 4, 100)

        fig, ax = plt.subplots(figsize=(3.7, 3))

        plt.title("RMSD normalization curve")  # , fontdict = font1)
        plt.xlabel("RMSD (Å)", fontsize=12)  # , fontdict = font2)
        plt.ylabel("Normalized label", fontsize=12)  # , fontdict = font2)

        plt.plot(x, sigmoid(x))
        # plt.plot(x, act_to_RMSD(x))
        plt.grid(which='both')
        ax.xaxis.set_minor_locator(MultipleLocator(5))

        #ax.legend(loc='center right', frameon=False)
        #equation_text = r"$y = x^2$"
        #ax.text(1.05, 0.5, equation_text + ' (' + r" (\textbf{eq. 1})" + ')', fontsize=12, color="black", va='center')

        plt.tight_layout()
        plt.savefig('sigmoid.png')

    plot_sigmoid()
