scale="0x"
mkdir RESI_${scale}
cp List_set_PDBBind/list_valset_clean.txt CSV_good_bad_poses_${scale}/
cp List_set_PDBBind/list_trainval_clean.txt CSV_good_bad_poses_${scale}/
cp List_set_PDBBind/list_trainset_clean.txt CSV_good_bad_poses_${scale}/
cp List_set_PDBBind/list_testset_clean.txt CSV_good_bad_poses_${scale}/

# Subtract pairs
python3 main.py --mode subtract_set --verbose 3 --CSV_data_dir CSV_good_bad_poses_${scale}/ --CSV_suffix _poses_H2O_output_${scale}-results_bitstring.csv --list_trainset list_trainval_clean.txt --list_testset list_testset_clean.txt > substract_good_bad_poses.out
mv substract_good_bad_poses.out RESI_${scale}/

while IFS= read -r residue; 
do

	resi_name=$(echo "$residue" | cut -c1-3)

	# Train XGB
	python3 main.py --mode train_model --model XGBc --features Name,RMSD,Label --features_train ${residue} --name ${resi_name}_ --normalize yes --hyperparameter 100 --seed 345 --maxevals 50 --CSV_data_dir CSV_good_bad_poses_${scale}/ --CSV_suffix _good_bad_substract.csv --list_trainset list_trainset_clean.txt --list_valset list_valset_clean.txt --list_testset list_testset_clean.txt --plotmetrics yes --verbose 3 > train_model_XGBc_${resi_name}.out
	
	# Train linear regression
	python3 main.py --mode train_model --model LRc --features Name,RMSD,Label --features_train ${residue} --name ${resi_name} --normalize yes --seed 345 --CSV_data_dir CSV_good_bad_poses_${scale}/ --CSV_suffix _good_bad_substract.csv --list_trainset list_trainset_clean.txt --list_valset list_valset_clean.txt --list_testset list_testset_clean.txt --plotmetrics yes --verbose 3 > train_model_LRc_${resi_name}.out

	mv models/LRc_${resi_name}.pkl RESI_${scale}/
	mv models/XGBc_${resi_name}.json RESI_${scale}/
	mv train_model_LRc_${resi_name}.out RESI_${scale}/
	mv train_model_XGBc_${resi_name}.out RESI_${scale}/
	
	mv ${resi_name}_features.csv RESI_${scale}/
	mv ${resi_name}_Permutation_importance_HeatMap.png RESI_${scale}/
	mv ${resi_name}_Permutation_importance_HeatMap_sorted.png RESI_${scale}/
	mv ${resi_name}_Permutation_importance.png RESI_${scale}/
	mv ${resi_name}_SHAP_values_mean.png RESI_${scale}/
	mv ${resi_name}_SHAP_values.png RESI_${scale}/
	mv ${resi_name}_XGBc_AUC.png RESI_${scale}/
	mv ${resi_name}_XGBc_error.png RESI_${scale}/
	mv ${resi_name}_XGBc_logloss.png RESI_${scale}/
	mv ${resi_name}_XGB_feature_importance.png RESI_${scale}/

done < list_AA.txt

# Plot RESI features
python3 main.py --mode plot_RESI --CSV_data_dir RESI_${scale}/

mv FeatureImportances.png RESI_${scale}/
mv Cover.png RESI_${scale}/
mv TotalCover.png RESI_${scale}/
mv Gain.png RESI_${scale}/
mv TotalGain.png RESI_${scale}/
mv Weight.png RESI_${scale}/
mv SHAP.png RESI_${scale}/
mv all_SHAP_values_top10.png RESI_${scale}/
mv all_SHAP_values_top15.png RESI_${scale}/
mv all_SHAP_values_top20.png RESI_${scale}/
mv all_SHAP_values_top20_horizontal.png RESI_${scale}/
mv Permutations.png RESI_${scale}/
mv Coefficients.png RESI_${scale}/