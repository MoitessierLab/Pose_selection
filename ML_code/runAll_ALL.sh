#scale=$1
scale="0x"
mkdir all_${scale}
mkdir models
cp LP-PDBBind2024/list_valset_clean.txt CSV_good_bad_poses_${scale}/
cp LP-PDBBind2024/list_trainval_clean.txt CSV_good_bad_poses_${scale}/
cp LP-PDBBind2024/list_trainset_clean.txt CSV_good_bad_poses_${scale}/
cp LP-PDBBind2024/list_testset_clean.txt CSV_good_bad_poses_${scale}/

# Subtract pairs
python3 main.py --mode subtract_set --verbose 3 --CSV_data_dir CSV_good_bad_poses_${scale}/ --CSV_suffix _poses_H2O_output_${scale}-results_bitstring.csv --list_trainset list_trainval_clean.txt --list_testset list_testset_clean.txt > substract_good_bad_poses.out
mv substract_good_bad_poses.out all_${scale}/

resi_name="all"

## Train XGB
python3 main.py --mode train_model --model XGBc --features Name,RMSD,Label --name ${resi_name} --normalize yes --hyperparameter 100 --seed 345 --maxevals 50 --CSV_data_dir CSV_good_bad_poses_${scale}/ --CSV_suffix _good_bad_substract.csv --list_trainset list_trainset_clean.txt --list_valset list_valset_clean.txt --list_testset list_testset_clean.txt --plotmetrics yes --verbose 3 > train_model_XGBc_${resi_name}.out

# Train linear regression
python3 main.py --mode train_model --model LRc --features Name,RMSD,Label --name ${resi_name} --normalize yes --seed 345 --CSV_data_dir CSV_good_bad_poses_${scale}/ --CSV_suffix _good_bad_substract.csv --list_trainset list_trainset_clean.txt --list_valset list_valset_clean.txt --list_testset list_testset_clean.txt --plotmetrics yes --verbose 3 > train_model_LRc_${resi_name}.out

mv models/LRc_${resi_name}.pkl all_${scale}/
mv models/XGBc_${resi_name}.json all_${scale}/
mv train_model_LRc_${resi_name}.out all_${scale}/
mv train_model_XGBc_${resi_name}.out all_${scale}/

mv ${resi_name}_features.csv all_${scale}/
mv ${resi_name}_Permutation_importance_HeatMap.png all_${scale}/
mv ${resi_name}_Permutation_importance_HeatMap_sorted.png all_${scale}/
mv ${resi_name}_Permutation_importance.png all_${scale}/
mv ${resi_name}_SHAP_values_mean.png all_${scale}/
mv ${resi_name}_SHAP_values.png all_${scale}/
mv ${resi_name}_XGBc_AUC.png all_${scale}/
mv ${resi_name}_XGBc_error.png all_${scale}/
mv ${resi_name}_XGBc_logloss.png all_${scale}/
mv ${resi_name}_XGB_feature_importance.png all_${scale}/

# Plot all features
python3 main.py --mode plot_ALL --CSV_data_dir all_${scale}/

mv FeatureImportances.png all_${scale}/
mv Cover.png all_${scale}/
mv TotalCover.png all_${scale}/
mv Gain.png all_${scale}/
mv TotalGain.png all_${scale}/
mv Weight.png all_${scale}/
mv SHAP.png all_${scale}/
mv all_SHAP_values_top10.png all_${scale}/
mv all_SHAP_values_top15.png all_${scale}/
mv all_SHAP_values_top20.png all_${scale}/
mv all_SHAP_values_top20_horizontal.png all_${scale}/
mv Permutations.png all_${scale}/
mv Coefficients.png all_${scale}/
