Author: Anne Labarre
McGill University, Montreal, QC, Canada



1.	Input CSV files from 25 correct and 25 incorrect poses (no correction factor)

		tar xzf CSV_good_bad_poses_1x

2.	To generate subtracted energy-based paired protein-ligand complexes and train models

		runAll_ALL.sh 			-- Train LRc and XGBc models using all features
		runAll_RESI_all.sh		-- Train LRc and XGBc models using total energy and all residues (ignoring the ligand internal energy)
		runAll_RESI.sh			-- Train LRc and XGBc models using total energy and single residues

3. To predict the docking accuracies

		runAll_predict_top10.sh