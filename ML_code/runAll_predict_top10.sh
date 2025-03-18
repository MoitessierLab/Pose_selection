scale=$1
mkdir all_${scale}

cp List_set_PDBBind/list_testset_clean.txt PDB_10/
cp List_set_PDBBind/list_trainset_clean.txt PDB_10/
cp List_set_PDBBind/list_trainval_clean.txt PDB_10/
cp List_set_PDBBind/list_valset_clean.txt PDB_10/
cp List_set_PDBBind/list_all_clean.txt PDB_10/

# PDBBind test set
#python3 main.py --mode test_model --model LRc --model_name all_${scale}/LRc_all --CSV_data_dir PDB_10/ --CSV_suffix _top10.csv --normalize yes --list_trainset list_testset_clean.txt --verbose 3 > LRc_predictions_PDBtop10_test.out
#python3 main.py --mode test_model --model XGBc --model_name all_${scale}/XGBc_all --CSV_data_dir PDB_10/ --CSV_suffix _top10.csv --normalize yes --list_trainset list_testset_clean.txt --verbose 3 > XGBc_predictions_PDBtop10_test.out
mv Accuracy_pred_LRc.png all_${scale}/Accuracy_pred_LRc_all_PDBtop10_test.png
mv Accuracy_pred_XGBc.png all_${scale}/Accuracy_pred_XGBc_all_PDBtop10_test.png
mv LRc_predictions_PDBtop10_test.out all_${scale}/
mv XGBc_predictions_PDBtop10_test.out all_${scale}/

# PDBBind train set
#python3 main.py --mode test_model --model LRc --model_name all_${scale}/LRc_all --CSV_data_dir PDB_10/ --CSV_suffix _top10.csv --normalize yes --list_trainset list_trainval_clean.txt --verbose 3 > LRc_predictions_PDBtop10_trainval.out
#python3 main.py --mode test_model --model XGBc --model_name all_${scale}/XGBc_all --CSV_data_dir PDB_10/ --CSV_suffix _top10.csv --normalize yes --list_trainset list_trainset_clean.txt --verbose 3 > XGBc_predictions_PDBtop10_train.out
mv Accuracy_pred_LRc.png all_${scale}/Accuracy_pred_LRc_all_PDBtop10_trainval.png
mv Accuracy_pred_XGBc.png all_${scale}/Accuracy_pred_XGBc_all_PDBtop10_train.png
mv LRc_predictions_PDBtop10_trainval.out all_${scale}/
mv XGBc_predictions_PDBtop10_train.out all_${scale}/

# PDBBind val set
python3 main.py --mode test_model --model LRc --model_name all_${scale}/LRc_all --CSV_data_dir PDB_10/ --CSV_suffix _top10.csv --normalize yes --list_trainset list_valset_clean.txt --verbose 3 > LRc_predictions_PDBtop10_val.out
#python3 main.py --mode test_model --model XGBc --model_name all_${scale}/XGBc_all --CSV_data_dir PDB_10/ --CSV_suffix _top10.csv --normalize yes --list_trainset list_valset_clean.txt --verbose 3 > XGBc_predictions_PDBtop10_val.out
mv Accuracy_pred_LRc.png all_${scale}/Accuracy_pred_LRc_all_PDBtop10_val.png
mv Accuracy_pred_XGBc.png all_${scale}/Accuracy_pred_XGBc_all_PDBtop10_val.png
mv LRc_predictions_PDBtop10_test.out all_${scale}/
mv XGBc_predictions_PDBtop10_test.out all_${scale}/

# PDBBind all set
#python3 main.py --mode test_model --model LRc --model_name all_${scale}/LRc_all --CSV_data_dir PDB_10/ --CSV_suffix _top10.csv --normalize yes --list_trainset list_all_clean.txt --verbose 3 > LRc_predictions_PDBtop10_full.out
#python3 main.py --mode test_model --model XGBc --model_name all_${scale}/XGBc_all --CSV_data_dir PDB_10/ --CSV_suffix _top10.csv --normalize yes --list_trainset list_all_clean.txt --verbose 3 > XGBc_predictions_PDBtop10_full.out
mv Accuracy_pred_LRc.png all_${scale}/Accuracy_pred_LRc_all_PDBtop10_full.png
mv Accuracy_pred_XGBc.png all_${scale}/Accuracy_pred_XGBc_all_PDBtop10_full.png
mv LRc_predictions_PDBtop10_trainval.out all_${scale}/
mv XGBc_predictions_PDBtop10_train.out all_${scale}/

# Fitted set
#python3 main.py --mode test_model --model LRc --model_name all_${scale}/LRc_all --CSV_data_dir FittedSet_top10/ --CSV_suffix _score-results_bitstring_top10poses.csv --normalize yes --list_trainset Fitted_list_full.txt --verbose 3 > LRc_predictions_FittedSetTop10.out
#python3 main.py --mode test_model --model XGBc --model_name all_${scale}/XGBc_all --CSV_data_dir FittedSet_top10/ --CSV_suffix _score-results_bitstring_top10poses.csv --normalize yes --list_trainset Fitted_list_full.txt --verbose 3 > XGBc_predictions_FittedSetTop10.out
mv Accuracy_pred_LRc.png all_${scale}/Accuracy_pred_LRc_all_FittedSet.png
mv Accuracy_pred_XGBc.png all_${scale}/Accuracy_pred_XGBc_all_FittedSet.png
mv LRc_predictions_FittedSetTop10.out all_${scale}/
mv XGBc_predictions_FittedSetTop10.out all_${scale}/


# Astex set
#python3 main.py --mode test_model --model LRc --model_name all_${scale}/LRc_all --CSV_data_dir AstexSet_top10/ --CSV_suffix _scored-results_bitstring.csv --normalize yes --list_trainset Astex_list_full.txt --verbose 3 > LRc_predictions_AstexSetTop10.out
#python3 main.py --mode test_model --model XGBc --model_name all_${scale}/XGBc_all --CSV_data_dir AstexSet_top10/ --CSV_suffix _scored-results_bitstring.csv --normalize yes --list_trainset Astex_list_full.txt --verbose 3 > XGBc_predictions_AstexSetTop10.out
mv Accuracy_pred_LRc.png all_${scale}/Accuracy_pred_LRc_all_AstexSet.png
mv Accuracy_pred_XGBc.png all_${scale}/Accuracy_pred_XGBc_all_AstexSet.png
mv LRc_predictions_AstexSetTop10.out all_${scale}/
mv XGBc_predictions_AstexSetTop10.out all_${scale}/