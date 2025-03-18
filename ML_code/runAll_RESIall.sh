scale=$1
mkdir RESIall_${scale}
cp List_set_PDBBind/list_valset_clean.txt CSV_good_bad_poses_${scale}/
cp List_set_PDBBind/list_trainval_clean.txt CSV_good_bad_poses_${scale}/
cp List_set_PDBBind/list_trainset_clean.txt CSV_good_bad_poses_${scale}/
cp List_set_PDBBind/list_testset_clean.txt CSV_good_bad_poses_${scale}/

# Subtract pairs
python3 main.py --mode subtract_set --verbose 3 --CSV_data_dir CSV_good_bad_poses_${scale}/ --CSV_suffix _poses_H2O_output_${scale}-results_bitstring.csv --list_trainset list_trainval_clean.txt --list_testset list_testset_clean.txt > substract_good_bad_poses.out
mv substract_good_bad_poses.out RESIall_${scale}/

residue=$(tr '\n' ',' < list_AA.txt | sed 's/,$//')
resi_name="ALL"

# Train XGB
python3 main.py --mode train_model --model XGBc --features Name,RMSD,Label --features_train Energy,${residue} --name ${resi_name} --normalize yes --hyperparameter 100 --seed 345 --maxevals 50 --CSV_data_dir CSV_good_bad_poses_${scale}/ --CSV_suffix _good_bad_substract.csv --list_trainset list_trainset_clean.txt --list_valset list_valset_clean.txt --list_testset list_testset_clean.txt --plotmetrics yes --verbose 3 > train_model_XGBc_${resi_name}.out

# Train linear regression
python3 main.py --mode train_model --model LRc --features Name,RMSD,Label --features_train Energy,${residue} --name ${resi_name} --normalize yes --seed 345 --CSV_data_dir CSV_good_bad_poses_${scale}/ --CSV_suffix _good_bad_substract.csv --list_trainset list_trainset_clean.txt --list_valset list_valset_clean.txt --list_testset list_testset_clean.txt --plotmetrics yes --verbose 3 > train_model_LRc_${resi_name}.out

mv models/LRc_${resi_name}.pkl RESIall_${scale}/
mv models/XGBc_${resi_name}.json RESIall_${scale}/
mv train_model_LRc_${resi_name}.out RESIall_${scale}/
mv train_model_XGBc_${resi_name}.out RESIall_${scale}/

mv ${resi_name}_features.csv RESIall_${scale}/
mv ${resi_name}_Permutation_importance_HeatMap.png RESIall_${scale}/
mv ${resi_name}_Permutation_importance_HeatMap_sorted.png RESIall_${scale}/
mv ${resi_name}_Permutation_importance.png RESIall_${scale}/
mv ${resi_name}_SHAP_values_mean.png RESIall_${scale}/
mv ${resi_name}_SHAP_values.png RESIall_${scale}/
mv ${resi_name}_XGBc_AUC.png RESIall_${scale}/
mv ${resi_name}_XGBc_error.png RESIall_${scale}/
mv ${resi_name}_XGBc_logloss.png RESIall_${scale}/
mv ${resi_name}_XGB_feature_importance.png RESIall_${scale}/

for AA in ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL; do

	head -1 RESIall_${scale}/ALL_features.csv > RESIall_${scale}/${AA}_features.csv
	grep "^${AA}" RESIall_${scale}/ALL_features.csv >> RESIall_${scale}/${AA}_features.csv

done

# Plot RESIall features
python3 main.py --mode plot_RESI --CSV_data_dir RESIall_${scale}/

mv FeatureImportances.png RESIall_${scale}/
mv Cover.png RESIall_${scale}/
mv TotalCover.png RESIall_${scale}/
mv Gain.png RESIall_${scale}/
mv TotalGain.png RESIall_${scale}/
mv Weight.png RESIall_${scale}/
mv SHAP.png RESIall_${scale}/
mv all_SHAP_values_top10.png RESIall_${scale}/
mv all_SHAP_values_top15.png RESIall_${scale}/
mv all_SHAP_values_top20.png RESIall_${scale}/
mv all_SHAP_values_top20_horizontal.png RESIall_${scale}/
mv Permutations.png RESIall_${scale}/
mv Coefficients.png RESIall_${scale}/


