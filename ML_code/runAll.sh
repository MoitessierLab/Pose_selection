# Filter by energy
#python3 main.py --mode filter_set --verbose 3 --CSV_data_dir CSV/ --energy_threshold 25 --list_trainset list_trainval_clean.txt --list_testset list_testset_clean.txt > split.out
#python3 main.py --mode filter_set --mismatch substract --verbose 3 --CSV_data_dir PDB_10/ --list_trainset list_trainval_clean.txt --list_testset list_testset_clean.txt > split_mismatch_substract.out
#python3 main.py --mode filter_set --verbose 3 --CSV_data_dir CSV_good_bad_poses/ --CSV_suffix _poses_output-results.txt --list_trainset list_trainval_clean.txt --list_testset list_testset_clean.txt #> substract_good_bad_poses.out

# Train linear regression
#python3 main.py --mode train_model --model LR --features Name,RMSD,Label --hyperparameter 0 --seed 345 --maxevals 50 --verbose 3 --CSV_data_dir CSV/ --list_trainset list_trainval_clean.txt --list_testset list_testset_clean.txt --normalize no > train_model_LR_v0001.out
#python3 main.py --mode train_model --model LRc --features Name,RMSD,Label,Energy --hyperparameter 0 --seed 345 --maxevals 50 --verbose 3 --CSV_data_dir CSV/ --list_trainset list_trainval_clean.txt --list_testset list_testset_clean.txt --normalize no > train_model_LR_v0002.out

# Train XGB
#python3 main.py --mode train_model --model XGBr --features Name,RMSD,Label --normalize yes --hyperparameter 10 --seed 345 --maxevals 50 --CSV_data_dir CSV/ --list_trainset list_trainval_clean.txt --list_testset list_testset_clean.txt --ngpu 1 --verbose 3 > train_model_XGBr_v0001.out
#python3 main.py --mode train_model --model XGBr --features Name,RMSD,Label,Energy --normalize yes --hyperparameter full --seed 345 --maxevals 50 --CSV_data_dir CSV/ --list_trainset list_trainval_clean.txt --list_testset list_testset_clean.txt --ngpu 1 --verbose 3 > train_model_XGBr_v0002.out

#python3 main.py --mode train_model --model LRc --features Name,RMSD,Label --name LRc_subs_E_NoNorm_ --normalize no --hyperparameter 0 --seed 345 --maxevals 50 --CSV_data_dir CSV_good_bad_poses/ --CSV_suffix _good_bad_substract.csv --list_trainset list_trainval_clean.txt --list_valset list_valset_clean.txt --list_testset list_testset_clean.txt --ngpu 1 --verbose 3 > train_model_LRc_substract_E_NoNorm.out
#python3 main.py --mode train_model --model LRc --features Name,RMSD,Label,Energy --name LRc_subs_noE_NoNorm_ --normalize no --hyperparameter 0 --seed 345 --maxevals 50 --CSV_data_dir CSV_good_bad_poses/ --CSV_suffix _good_bad_substract.csv --list_trainset list_trainval_clean.txt --list_valset list_valset_clean.txt --list_testset list_testset_clean.txt --ngpu 1 --verbose 3 > train_model_LRc_substract_noE_NoNorm.out
#python3 main.py --mode train_model --model LRc --features Name,RMSD,Label --normalize yes --name LRc_subs_E_Norm_ --hyperparameter 0 --seed 345 --maxevals 50 --CSV_data_dir CSV_good_bad_poses/ --CSV_suffix _good_bad_substract.csv --list_trainset list_trainval_clean.txt --list_valset list_valset_clean.txt --list_testset list_testset_clean.txt --ngpu 1 --verbose 3 > train_model_LRc_substract_E_Norm.out
#python3 main.py --mode train_model --model LRc --features Name,RMSD,Label,Energy --name LRc_subs_noE_Norm_ --normalize yes --hyperparameter 0 --seed 345 --maxevals 50 --CSV_data_dir CSV_good_bad_poses/ --CSV_suffix _good_bad_substract.csv --list_trainset list_trainval_clean.txt --list_valset list_valset_clean.txt --list_testset list_testset_clean.txt --ngpu 1 --verbose 3 > train_model_LRc_substract_noE_Norm.out


#python3 main.py --mode train_model --model XGBc --features Name,RMSD,Label --name XGBc_subs_E_NoNorm_ --normalize no --hyperparameter 100 --seed 345 --maxevals 50 --CSV_data_dir CSV_good_bad_poses/ --CSV_suffix _good_bad_substract.csv --list_trainset list_trainval_clean.txt --list_valset list_valset_clean.txt --list_testset list_testset_clean.txt --ngpu 1 --verbose 3 > train_model_XGBc_substract_E_NoNorm.out
#python3 main.py --mode train_model --model XGBc --features Name,RMSD,Label,Energy --name XGBc_subs_noE_NoNorm_ --normalize no --hyperparameter 100 --seed 345 --maxevals 50 --CSV_data_dir CSV_good_bad_poses/ --CSV_suffix _good_bad_substract.csv --list_trainset list_trainval_clean.txt --list_valset list_valset_clean.txt --list_testset list_testset_clean.txt --ngpu 1 --verbose 3 > train_model_XGBc_substract_noE_NoNorm.out
#python3 main.py --mode train_model --model XGBc --features Name,RMSD,Label --name XGBc_subs_E_Norm_ --normalize yes --hyperparameter 100 --seed 345 --maxevals 50 --CSV_data_dir CSV_good_bad_poses/ --CSV_suffix _good_bad_substract.csv --list_trainset list_trainval_clean.txt --list_valset list_valset_clean.txt --list_testset list_testset_clean.txt --ngpu 1 --verbose 3 > train_model_XGBc_substract_E_Norm.out
#python3 main.py --mode train_model --model XGBc --features Name,RMSD,Label,Energy --name XGBc_subs_noE_Norm_ --normalize yes --hyperparameter 100 --seed 345 --maxevals 50 --CSV_data_dir CSV_good_bad_poses/ --CSV_suffix _good_bad_substract.csv --list_trainset list_trainval_clean.txt --list_valset list_valset_clean.txt --list_testset list_testset_clean.txt --ngpu 1 --verbose 3 > train_model_XGBc_substract_noE_Norm.out


# Evaluate accuracy on other sets
#python3 main.py --mode test_model --model LRc --model_name LRc_subs_E_NoNorm.pkl --CSV_data_dir PDB_10/ --normalize no --list_trainset list_trainval_clean.txt --list_testset list_testset_clean.txt --verbose 3 > LRc_PDB10_trainval.out
#python3 main.py --mode test_model --model LRc --model_name LRc_subs_E_NoNorm.pkl --CSV_data_dir PDB_10/ --normalize no --list_trainset list_all_clean.txt --list_testset list_testset_clean.txt --verbose 3 > LRc_PDB10_all.out
#python3 main.py --mode test_model --model XGBc --model_name XGBc_subs_E_NoNorm.json --CSV_data_dir PDB_10/ --normalize no --list_trainset list_trainval_clean.txt --list_testset list_testset_clean.txt --verbose 3 > XGBc_PDB10_trainval.out
#python3 main.py --mode test_model --model XGBc --model_name XGBc_subs_E_NoNorm.json --CSV_data_dir PDB_10/ --normalize no --list_trainset list_trainset_clean.txt --list_testset list_testset_clean.txt --verbose 3 > XGBc_PDB10_trainset.out
#python3 main.py --mode test_model --model XGBc --model_name XGBc_subs_E_NoNorm.json --CSV_data_dir PDB_10/ --normalize no --list_trainset list_all_clean.txt --list_testset list_testset_clean.txt --verbose 3 > XGBc_PDB10_all.out

#python3 main.py --mode test_model --model LRc --model_name LRc_subs_E_NoNorm.pkl --CSV_data_dir FittedSet_top10/ --CSV_suffix _score-results_bitstring_top10poses.csv --normalize no --list_trainset Fitted_list_full.txt --verbose 3 > LRc_Fitted10.out
#python3 main.py --mode test_model --model XGBc --model_name XGBc_subs_E_NoNorm.json --CSV_data_dir FittedSet_top10/ --CSV_suffix _score-results_bitstring_top10poses.csv --normalize no --list_trainset Fitted_list_full.txt --verbose 3 > XGBc_Fitted10.out
#python3 main.py --mode test_model --model LRc --model_name LRc_subs_E_NoNorm.pkl --CSV_data_dir AstexSet_top10/ --CSV_suffix _scored-results_bitstring.csv --normalize no --list_trainset Astex_list_full.txt --verbose 3 > LRc_Astex10.out
#python3 main.py --mode test_model --model XGBc --model_name XGBc_subs_E_NoNorm.json --CSV_data_dir AstexSet_top10/ --CSV_suffix _scored-results_bitstring.csv --normalize no --list_trainset Astex_list_full.txt --verbose 3 > XGBc_Astex10.out



#python3 main.py --mode test_model --model LR --model_name LR_model_2024-07-18_19-25-11.pkl --set Astex --normalize yes --verbose 3 > test_model.out
#python3 main.py --mode test_model --model XGBc --model_name XGBc_mismatch_PDB.json --set PDB_mismatch --test_on train --list_trainset train.txt --list_testset test.txt --normalize no --verbose 3 #> test_model_mismatch_XGBc.out
#python3 main.py --mode test_model --model LRc --model_name LR_model_2024-09-11_11h33-18.pkl --set PDB_mismatch --test_on test --list_trainset train.txt --list_testset test.txt --normalize no --verbose 3 #> test_model_mismatch_LRc.out

# Plot stuff
#python3 main.py --mode plotstuff
python3 main.py --mode plotstuff --features Name,RMSD,Label --normalize yes --CSV_data_dir CSV_good_bad_poses/ --CSV_suffix _good_bad_substract.csv --list_trainset list_trainset_clean.txt --list_valset list_valset_clean.txt --list_testset list_testset_clean.txt --verbose 3	

#python3 main.py --mode plotstuff --set CSV_HB05x --CSV_data_dir CSV_HB05x/ --list_trainset list_all_clean.txt
#python3 main.py --mode plotstuff --set PDB_10 --CSV_data_dir PDB_10/ --list_trainset list_10.txt
#python3 main.py --mode plotstuff --features Name,Label --normalize yes --CSV_data_dir CSV/ --list_trainset list_trainval_clean.txt --list_testset list_testset_clean.txt --verbose 3
#python3 main.py --mode plotstuff --model LRc --model_name LR_model_2024-09-12_14h08-09.pkl --features Name,Label --normalize no --CSV_data_dir PDB_10/ --list_trainset train.txt --list_testset test.txt --verbose 3 > accuracy_LRc.out

#python3 main.py --mode split_good_bad_poses --verbose 3 --raw_data_dir raw_data_poses/ --max_number_of_poses 25 --max_rmsd_good 1.75 --min_rmsd_bad 3.0 --energy_threshold 25 --list_trainset list_trainval_clean.txt --list_testset list_testset_clean.txt > split.out

#rm raw_data_poses/*_output_evolving_population.sdf raw_data_poses/*_output_initial_population.sdf raw_data_poses/*_output_initial_pose.sdf raw_data_poses/*_output.sdf raw_data_poses/*_temp.sdf

#python3 main.py --mode prepare_complexes --list_trainset list_trainval_clean.txt --list_testset list_testset_clean.txt --verbose 3 --max_number_of_poses 6 --train_mode docking --raw_data_dir raw_data_poses/ --data_dir data_poses/ --keys_dir  keys_poses/  > prepare-complexes.out 

#python3 main.py --mode prepare_graphs --train_keys train_keys.pkl --test_keys test_keys.pkl --max_rmsd_good 1.75 --min_rmsd_bad 3.0 --max_number_of_poses 6 --keys_dir keys_poses/ --data_dir data_poses/ --graphs_dir data_poses_graphs/ --scrambling_graphs False  --good_and_bad_pairs True > prepare-graphs.out

#python3 main.py --mode train --n_graph_layers 4 --n_FC_layers 3 --verbose 5 --output training_20240125_PY --graphs_dir data_poses_graphs/ --keys_dir keys_poses/ --train_mode docking --graph_as_input True --epoch 1000 --lr 0.0001 --embedding_size 200 --model_dense_neurons  128 --batch_size 6 --dropout_rate 0.3 --atom_feature_element False --atom_feature_metal True --atom_feature_atom_size False --atom_feature_electronegativity False --atom_feature_partial_charge True --atom_feature_logP True --atom_feature_MR False --atom_feature_TPSA True --atom_feature_HBA_HBD True --atom_feature_aromaticity True --atom_feature_number_of_Hs True --atom_feature_formal_charge True --bond_feature_bond_order True --bond_feature_conjugation True --good_and_bad_pairs True --max_num_of_systems 0 --debug 0 --model_attention_heads 4 --weight_decay 0.0001 --restart training_20240125_PY_last.pth > Training_20240528.r1.out


# random: log (1/2) = -0.693147
# 29.6% 0.683
