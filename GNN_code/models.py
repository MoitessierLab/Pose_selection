# Author: Benjamin Weiser
# Edited and adapted by Anne Labarre
# Date created: 2024-06-17
# McGill University, Montreal, QC, Canada

import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import os
import time
import matplotlib;
matplotlib.use('Agg')  # for backend to write file no render in window
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
#import json
#import hyperopt
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from sklearn.linear_model import LinearRegression
import math
import pickle
from prepare_input_data import split_set
from plot_models import XGB_feature_importance, SHAP_values, permutation_imp, plot_eval_metrics
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from tqdm import tqdm
import io
#import cupy as cp
#import cudf
from sklearn.metrics import mean_absolute_error

def train_LR(args):
    
    now = time.localtime()
    t = "%04d-%02d-%02d_%02dh%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

# ----------- Split, normalize, randomize the dataset

    #train_set_X, train_set_y, test_set_X, test_set_y, features = split_set(args)
    train_set_X, train_set_y, val_set_X, val_set_y, test_set_X, test_set_y, features = split_set(args)
    print('| Combining Train and Validation sets                                                                               |')
    print('|-------------------------------------------------------------------------------------------------------------------|')
    train_set_X = pd.concat([train_set_X, val_set_X], axis=0)
    train_set_y = pd.concat([train_set_y, val_set_y], axis=0)

# ----------- Model fitting

    print('| Starting model training for Linear Regression                                                                     |')
    now_start = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now_start.tm_year, now_start.tm_mon, now_start.tm_mday, now_start.tm_hour, now_start.tm_min, now_start.tm_sec)
    print('| %s                                                                                               |' % s)
    print('|-------------------------------------------------------------------------------------------------------------------|')
    model = LinearRegression()
    model.fit(train_set_X, train_set_y)

    now_end = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now_end.tm_year, now_end.tm_mon, now_end.tm_mday, now_end.tm_hour, now_end.tm_min, now_end.tm_sec)
    print('| Model finished training at %-86s |' % s)
    print('|-------------------------------------------------------------------------------------------------------------------|')

# ----------- Model features

    # Slope
    print('| Coefficients (feature importances):                                                                               |')

    feature_coeff_pairs = list(zip(train_set_X.columns, model.coef_)) # Match the features and coefficients
    sorted_feature_coeff_pairs = sorted(feature_coeff_pairs, key=lambda x: x[1], reverse=True) # Sort the pairs in decreasing order of the coefficient

    for feature, coeff in sorted_feature_coeff_pairs: # Print the sorted feature names and coefficients
        print('| %-20s %-92s |' % (feature, coeff))
    print('|-------------------------------------------------------------------------------------------------------------------|')

    # y-interecept
    print('| y-Intercept: %-100f |' % model.intercept_)

    # R squared
    r_sq = model.score(test_set_X, test_set_y)
    print('| Coefficient of determination, R-squared: %-72f |' % r_sq)
    print('|-------------------------------------------------------------------------------------------------------------------|')

# ----------- Save the model

    t = "%04d-%02d-%02d_%02dh%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    modelname = 'LR_model_' + t + '.pkl'
    pickle.dump(model, open('models/' + modelname, 'wb'))
    print('| Model saved as "LR_model_%s.pkl"                                                                 |' % t)

# ----------- Test the model on the test set

    prediction_y = model.predict(test_set_X)
    R2_score_test = model.score(test_set_X, test_set_y)
    R2_score_pred = model.score(test_set_X, prediction_y)
#    print('| Testing the model:                                                                                                |')
#    print('| R-squared for the prediction on the test set: %-83s |' % R2_score_test )
#    print('| R-squared for the prediction on the predictions: %-83s |' % R2_score_pred )
#    print('|-------------------------------------------------------------------------------------------------------------------|')

    # MAE on the train set
    expected_train = train_set_y
    predicted_train = model.predict(train_set_X)
    MAE_train = mean_absolute_error(expected_train, predicted_train)
    print("| MAE train + val set (LRr): " + str(MAE_train))

    # MAE on the test set
    expected_test = test_set_y
    predicted_test = model.predict(test_set_X)
    MAE_test = mean_absolute_error(expected_test, predicted_test)
    print("| MAE test set (LRr):        " + str(MAE_test))

    Astex = pd.read_csv('CSV/Astex_top10_combined.csv')
    # print(Astex)
    Astex_y = Astex['Label']
    features_to_drop = ['Name', 'RMSD', 'Label']
    Astex_X = Astex.drop(columns=features_to_drop, axis=1)
    Astex_X_norm = preprocessing.normalize(Astex_X)
    expected_Astex = Astex_y
    predicted_Astex = model.predict(Astex_X_norm)
    MAE_Astex = mean_absolute_error(expected_Astex, predicted_Astex)
    print("| MAE Astex set (LRr):       " + str(MAE_Astex))

def train_LRc(args):

    now = time.localtime()
    t = "%04d-%02d-%02d_%02dh%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

# ----------- Split, normalize, randomize the dataset

    train_set_X, train_set_y, val_set_X, val_set_y, test_set_X, test_set_y, features = split_set(args)
    print('| Combining Train and Validation sets                                                                               |')
    print('|-------------------------------------------------------------------------------------------------------------------|')
    train_set_X = pd.concat([train_set_X, val_set_X], axis=0)
    train_set_y = pd.concat([train_set_y, val_set_y], axis=0)

# ----------- Model fitting

    print('| Starting model training for Linear Regression                                                                     |')
    now_start = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now_start.tm_year, now_start.tm_mon, now_start.tm_mday, now_start.tm_hour, now_start.tm_min, now_start.tm_sec)
    print('| %s                                                                                               |' % s)
    print('|-------------------------------------------------------------------------------------------------------------------|')
    model = LogisticRegression(solver='lbfgs', max_iter=100)
    model.fit(train_set_X, train_set_y)

    now_end = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now_end.tm_year, now_end.tm_mon, now_end.tm_mday, now_end.tm_hour, now_end.tm_min, now_end.tm_sec)
    print('| Model finished training at %-86s |' % s)
    print('|-------------------------------------------------------------------------------------------------------------------|')

# ----------- Model features

    if args.plotmetrics == 'yes':

        # Get the coefficients
        coef_values = np.ravel(model.coef_)
        coef_values = [x[0] if isinstance(x, (list, np.ndarray)) else x for x in coef_values]
        coef = pd.DataFrame(zip(train_set_X.columns, coef_values), columns=['Features', 'Coefficients'])

        #coef = pd.DataFrame(zip(train_set_X.columns, np.transpose(model.coef_)), columns=['Features', 'Coefficients'])
        # Remove square brackets in coef['Coefficients']
        #coef['Coefficients'] = coef['Coefficients'].apply(lambda x: x[0] if isinstance(x, list) else x)

        if os.path.isfile(args.name + '_features.csv') is True:
            xgb = pd.read_csv(args.name + '_features.csv', index_col=False)
            xgb = pd.concat([xgb, coef['Coefficients']], axis=1)
            xgb.to_csv(args.name + '_features.csv', index=False)
        else:
            coef.to_csv(args.name + '_features.csv', index=False)

        # coef = coef.sort_values('coef')
        # y_int = model.intercept_
        # for index, row in coef.iterrows():
        #     print('| %-20s %-92.4f |' % (row['Features'], row['Coefficients'][0]))
        # print('| y-intercept: %-92.4f |' % y_int)
        # print('|-------------------------------------------------------------------------------------------------------------------|')


    # # Slope
    # print('| Coefficients (feature importances):                                                                               |')
    #
    # feature_coeff_pairs = list(zip(train_set_X.columns, model.coef_)) # Match the features and coefficients
    # sorted_feature_coeff_pairs = sorted(feature_coeff_pairs, key=lambda x: x[1], reverse=True) # Sort the pairs in decreasing order of the coefficient
    #
    #
    # # y-interecept
    # print('| y-Intercept: %-100f |' % model.intercept_)
    #
    # # R squared
    # r_sq = model.score(test_set_X, test_set_y)
    # print('| Coefficient of determination, R-squared: %-72f |' % r_sq)
    # print('|-------------------------------------------------------------------------------------------------------------------|')
    #
    # #model.named_steps['m'].coef_

# ----------- Test the model on the test set

    # make predictions for train data
    y_pred_train = model.predict(train_set_X)
    predictions_train = [round(value) for value in y_pred_train]
    # evaluate predictions
    accuracy_train = accuracy_score(train_set_y, predictions_train)
    print('| Model accuracy on the train + validation set: %.2f%%                                                              |' % (accuracy_train * 100.0))

    # make predictions for test data
    y_pred_test = model.predict(test_set_X)
    predictions_test = [round(value) for value in y_pred_test]
    # evaluate predictions
    accuracy_test = accuracy_score(test_set_y, predictions_test)
    print('| Model accuracy on the test set: %.2f%%                                                                            |' % (accuracy_test * 100.0))

    #prediction_y = model.predict(test_set_X)
    #R2_score_test = model.score(test_set_X, test_set_y)
    #R2_score_pred = model.score(test_set_X, prediction_y)
#    print('| Testing the model:                                                                                                |')
#    print('| R-squared for the prediction on the test set: %-83s |' % R2_score_test )
#    print('| R-squared for the prediction on the predictions: %-83s |' % R2_score_pred )
#    print('|-------------------------------------------------------------------------------------------------------------------|')

# ----------- Save the model

    t = "%04d-%02d-%02d_%02dh%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    modelname = 'LR_model_' + t + '.pkl'
    #pickle.dump(model, open('models/' + modelname, 'wb'))
    pickle.dump(model, open('models/LRc_' + args.name + '.pkl', 'wb'))
    print('| Model saved as "LR_%s.pkl"                                                                                    |' % args.name)

def train_XGBr(args):

    now = time.localtime()
    t = "%04d-%02d-%02d_%02dh%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

# ----------- Split, normalize, randomize the dataset

    train_set_X, train_set_y, val_set_X, val_set_y, test_set_X, test_set_y, features = split_set(args)

# ----------- Hyperparameter tuning

    if args.hyperparameter == '0':
        print('| No hyperparameter search performed                                                                                |')
        print('|-------------------------------------------------------------------------------------------------------------------|')
        
        best = {
            'max_depth': args.max_depth,
            'gamma': args.gamma,
            'reg_alpha': args.reg_alpha,
            'reg_lambda': args.reg_lambda,
            'min_child_weight': args.min_child_weight,
            'n_estimators': args.n_estimators,
            'random_state': args.seed}

        print('| No hyperparameter search performed, reverting to default or user defined hyperparameters                           |')
        print('|   max_depth:            %-30.0f                                                            |' % best['max_depth'])
        print('|   gamma:                %-30.16f                                                            |' % best['gamma'])
        print('|   reg_alpha:            %-30.1f                                                            |' % best['reg_alpha'])
        print('|   reg_lambda:           %-30.17f                                                            |' % best['reg_lambda'])
        print('|   min_child_weight:     %-30.1f                                                            |' % best['min_child_weight'])
        print('|   n_estimators:         %-30.1f                                                            |' % best['n_estimators'])
        print('|-------------------------------------------------------------------------------------------------------------------|')

    else:
        now_start = time.localtime()
        s = "%04d-%02d-%02d %02d:%02d:%02d" % (now_start.tm_year, now_start.tm_mon, now_start.tm_mday, now_start.tm_hour, now_start.tm_min, now_start.tm_sec)
        print('| Starting hyperparameter search for eXtreme Gradient Boosting - Regressor at %-37s |' %s)

        if args.hyperparameter == '100': # train on the full set
            print('| Tuning using %i%% of the Train set                                                                                |' % args.hyperparameter )
            hyp_set_X = train_set_X
            hyp_set_y = train_set_y


        else: # train on a part of the set (faster for testing the script)
            print('| Tuning using %i%% of the Train set                                                                                 |' % args.hyperparameter)
            numlines = int((args.hyperparameter/100) * train_set_X.shape[0])
            hyp_set_X = train_set_X.iloc[0:numlines, :]
            hyp_set_y = train_set_y.iloc[0:numlines, ]

    #if pm['hyp_tune'] == 1:
    #    if pm['use_some_data_hyp'] ==1 :
            #X = dataset[0:pm['use_data_size_hyp'],:]
            #yy = y[0:pm['use_data_size_hyp'],]
    #    else:
            #yy=y
    #        hyp_y = train_set_y

        # --- Search space --------------------------
        space = {
            'max_depth': hp.choice('max_depth', np.arange(3, 30, 1, dtype=int)),
            'gamma': hp.uniform('gamma', 1,9),
            'reg_alpha' : hp.quniform('reg_alpha', 20,180,1),
            'reg_lambda' : hp.uniform('reg_lambda', 0,1),
            'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
            'n_estimators': args.n_estimators,
            'random_state': args.seed
            # 'n_jobs': 2
        }

        # --- Objective function --------------------

        # def hyperopt_train_test(params):
        #     #X_ = X[:]
        #     #train_set_X_ = train_set_X[:]
        #     reg = XGBRegressor(**params)
        #     #reg = XGBRegressor(**params, tree_method="hist", device="cuda")
        #     reg.random_state = args.seed
        #     #return -cross_val_score(model, X, yy, scoring='neg_mean_absolute_error', cv=5).mean()
        #     return -cross_val_score(reg, hyp_set_X, hyp_set_y, scoring='neg_mean_absolute_error', cv=5).mean()

        eval_set = [(hyp_set_X, hyp_set_y), (val_set_X, val_set_y)]
        eval_metric = "mae"
        def objective(space):
            reg=XGBRegressor(n_estimators=space['n_estimators'],
                                 max_depth = int(space['max_depth']),
                                 gamma = space['gamma'],
                                 reg_alpha = int(space['reg_alpha']),
                                 min_child_weight=int(space['min_child_weight']),
                                 eval_metric=eval_metric,
                                 random_state=args.seed,
                                 early_stopping_rounds=10
                             )

            #evaluation = [(train_set_X, train_set_y), (test_set_X, test_set_y)]
            #reg.fit(train_set_X, train_set_y, eval_set=evaluation, eval_metric="auc", early_stopping_rounds=10, verbose=False)
            reg.fit(hyp_set_X, hyp_set_y, eval_set=eval_set, verbose=False)

            y_pred = reg.predict(val_set_X)
            #accuracy = accuracy_score(test_set_y, pred > 0.5)
            mae = mean_absolute_error(val_set_y, y_pred)
            #accuracy = accuracy_score(val_set_y, y_pred > 0.5)
            print("SCORE:", mae)
            return {'loss': mae, 'status': STATUS_OK}

        # Optimization algorithm
        trials = Trials()
        # best = fmin(f, space, algo=tpe.suggest, max_evals=args.maxevals, trials=trials)
        best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = args.maxevals, trials = trials)

        # global bestloss
        # bestloss = 1
        # 
        # def f(params):
        #     global bestloss
        #     loss = hyperopt_train_test(params)
        #     #print('eval MAE: ', loss)
        #     #print('| Eval MAE: %-103s |' % str(loss))
        #     if loss < bestloss:
        #         bestloss = loss
        #         #print('new best MAE:', bestloss, params)
        #         #print('| New best MAE: %-101s |' % (str(bestloss))) #, %-80s |' % (str(bestloss), str(params)))
        #     return {'loss': loss, 'status': STATUS_OK}
        # 
        # # --- Optimization algorithm --------------------
        # 
        # trials = Trials()
        # # best = fmin(fn = objective,
        # #             space = space,
        # #             algo = tpe.suggest,
        # #             max_evals = 100,
        # #             trials = trials)
        # best = fmin(f, space, algo=tpe.suggest, max_evals=args.maxevals, trials=trials)
        # # https://docs.rapids.ai/deployment/stable/examples/xgboost-gpu-hpo-job-parallel-ngc/notebook/

        #print('best:')
        # print(best)
        now_end = time.localtime()
        s = "%04d-%02d-%02d %02d:%02d:%02d" % (now_end.tm_year, now_end.tm_mon, now_end.tm_mday, now_end.tm_hour, now_end.tm_min, now_end.tm_sec)
        print('| Hyperparameter search finished at %-79s |' % s)
        print('|-------------------------------------------------------------------------------------------------------------------|')
        print('| Best hyperparameters                                                                                              |')
        print('|   max_depth:            %-30.0f                                                            |' % best['max_depth'])
        print('|   gamma:                %-30.16f                                                            |' % best['gamma'])
        print('|   reg_alpha:            %-30.1f                                                            |' % best['reg_alpha'])
        print('|   reg_lambda:           %-30.17f                                                            |' % best['reg_lambda'])
        print('|   min_child_weight:     %-30.1f                                                            |' % best['min_child_weight'])
        print('|-------------------------------------------------------------------------------------------------------------------|')

        #with open(pm['output_filename'], 'a') as fileOutput:
        #    fileOutput.write(name + ' best params: ' + str(best) + '\n')

# ----------- Graphs for Hyperparameter tuning optimization

        parameters = ['max_depth', 'gamma', 'reg_alpha' , 'reg_lambda', 'min_child_weight']  # , 'criterion']
        f, axes = plt.subplots(ncols=5, figsize=(15, 5))
        cmap = plt.cm.jet
        i = 0
        for i, val in enumerate(parameters):
            # print(i, val)
            xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
            ys = [-t['result']['loss'] for t in trials.trials]
            axes[int(i)].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5, c=cmap(float(i) / len(parameters)))
            axes[int(i)].set_title(val)
        # axes[i,i/3].set_ylim([0.9,1.0])

        graphname = t + '_XGBr_hp_search.png'
        plt.savefig('hyperparameters/' + graphname)

# ----------- Model fitting

#   model.fit(X, y)

    now_start = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now_start.tm_year, now_start.tm_mon, now_start.tm_mday, now_start.tm_hour, now_start.tm_min, now_start.tm_sec)
    print('| Starting model training for eXtreme Gradient Boosting - regressor at %-44s |' %s )
    #print('|-------------------------------------------------------------------------------------------------------------------|')

    # load the data set on gpu
    # https://github.com/dmlc/xgboost/issues/9791
    # train_set_X = cp.array(train_set_X)
    # train_set_y = cp.array(train_set_y)
    # test_set_X = cp.array(test_set_X)
    # test_set_y = cp.array(test_set_y)

    #model = XGBRegressor(tree_method="hist", device="cuda")
    model = XGBRegressor(max_depth=best['max_depth'] + 3, gamma=best['gamma'], reg_alpha=best['reg_alpha'], reg_lambda=best['reg_lambda'], min_child_weight=best['min_child_weight'], n_estimators=180, tree_method="hist", device="gpu")
    #model = XGBRegressor(max_depth=best['max_depth'] + 3, gamma=best['gamma'], reg_alpha=best['reg_alpha'], reg_lambda=best['reg_lambda'], min_child_weight=best['min_child_weight'], n_estimators=180)
    model.fit(train_set_X, train_set_y)

    now_end = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now_end.tm_year, now_end.tm_mon, now_end.tm_mday, now_end.tm_hour, now_end.tm_min, now_end.tm_sec)
    print('| Model finished training at %-86s |' % s)
    #print('|-------------------------------------------------------------------------------------------------------------------|')

    #scores = cross_val_score(model, train_set_X, train_set_y, scoring='neg_mean_absolute_error', cv=5, n_jobs=1)
    #scores = np.absolute(scores)

    #scored = {}
    #scored['score'] = model.score(X=testSet_norm, y=scores_test)
    #scored['score'] = model.score(X=test_set_X, y=test_set_y)
    #scored['cv'] = str(scores)

    #scores_pred_test = model.predict(testSet_norm)
    #prediction_y = model.predict(test_set_X)

    print('|-------------------------------------------------------------------------------------------------------------------|')
    print('| Training summary                                                                                                  |')
    print('|   Best hyperparameters                                                                                            |') # %-90s |' % (str(best)))
    print('|      max_depth:                 %-22.0f                                                            |' % best['max_depth'])
    print('|      gamma:                     %-22.16f                                                            |' % best['gamma'])
    print('|      reg_alpha:                 %-22.1f                                                            |' % best['reg_alpha'])
    print('|      reg_lambda:                %-22.17f                                                            |' % best['reg_lambda'])
    print('|      min_child_weight:          %-22.1f                                                            |' % best['min_child_weight'])
    #print('|   Mean MAE:                     %.3f (%.3f)                                                                     |' % (scores.mean(), scores.std()))
    #print('|   5-fold cross validation:      %-81s |' % (str(scores)))
    #print('|   Test score:                   %-81s |' % (str(scored['score'])))
    print('|-------------------------------------------------------------------------------------------------------------------|')
    #print('| Accuracy score: %-90s |' % (str(accuracy_score(test_set_y, prediction_y))))
    #print('| Test Score: ' scored, accuracy_score(scores_test, scores_pred_test))

# ----------- Saving the model

    modelname = t + '_XGBr_model.json'
    model.save_model('models/' + modelname)
    #pickle.dump(model, open(modelname, 'wb'))
    print('| Model saved as %s_XGBr_model.json"                                                                |' % t)



        #with open(pm['output_filename'], 'a') as fileOutput:
            #fileOutput.write(name + ' best params: ' + str(best) + '\n')
            #fileOutput.write('5-fold cross validation: ' + str(scored['cv']) + '\n')
            #fileOutput.write(
            #    'Test Score: ' + str(scored['score']) + 'Accuracy Score: ' + str(
            #        accuracy_score(scores_test, scores_pred_test)) + '\n')

        #testresults = CYP_inhibition_functions.testresults(scores_test,scores_pred_test)
        #print(testresults)
        #scored = {**scored, **testresults}
        #scored = pd.DataFrame.from_dict(scored, orient='index', columns=[name])
        #print(scored)

        #scored = pd.DataFrame.from_dict(scored, orient='index')
        #return scored

def train_XGBc(args):

    now = time.localtime()
    t = "%04d-%02d-%02d_%02dh%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

# ----------- Split, normalize, randomize the dataset

    train_set_X, train_set_y, val_set_X, val_set_y, test_set_X, test_set_y, features = split_set(args)

# ----------- Hyperparameter tuning

    if args.hyperparameter == 0:
        # Uses default hyperparameter value if no args is defined
        # else uses user-defined hp
        best = {
            'max_depth': args.max_depth,
            'gamma': args.gamma,
            'reg_alpha': args.reg_alpha,
            'reg_lambda': args.reg_lambda,
            'min_child_weight': args.min_child_weight,
            'n_estimators': args.n_estimators,
            'random_state': args.seed}

        print('| No hyperparameter search performed, reverting to default or user defined hyperparameters                           |')
        print('|   max_depth:            %-30.0f                                                            |' % best['max_depth'])
        print('|   gamma:                %-30.16f                                                            |' % best['gamma'])
        print('|   reg_alpha:            %-30.1f                                                            |' % best['reg_alpha'])
        print('|   reg_lambda:           %-30.17f                                                            |' % best['reg_lambda'])
        print('|   min_child_weight:     %-30.1f                                                            |' % best['min_child_weight'])
        print('|   n_estimators:         %-30.1f                                                            |' % best['n_estimators'])
        print('|-------------------------------------------------------------------------------------------------------------------|')

    else:
        now_start = time.localtime()
        s = "%04d-%02d-%02d %02d:%02d:%02d" % (now_start.tm_year, now_start.tm_mon, now_start.tm_mday, now_start.tm_hour, now_start.tm_min, now_start.tm_sec)
        print('| Starting hyperparameter search for eXtreme Gradient Boosting - Classifier at %-36s |' %s)

        if args.hyperparameter == 100: # train on 100% of the set
            print('| Tuning using %i%% of the Train set                                                                                |' % args.hyperparameter )
            hyp_set_X = train_set_X
            hyp_set_y = train_set_y

        else: # train on a percentage of the set (faster for testing the script)
            print('| Tuning using %i%% of the Train set                                                                                 |' % args.hyperparameter)
            numlines = int((args.hyperparameter/100) * train_set_X.shape[0])
            hyp_set_X = train_set_X[0:numlines, :]
            hyp_set_y = train_set_y[0:numlines, ]

        # Search space
        space = {
            'max_depth': hp.choice('max_depth', np.arange(3, 30, 1, dtype=int)),
            'gamma': hp.uniform('gamma', 1,9),
            'reg_alpha' : hp.quniform('reg_alpha', 20,180,1),
            'reg_lambda' : hp.uniform('reg_lambda', 0,1),
            'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
            'n_estimators': args.n_estimators,
            'random_state': args.seed
            #'n_jobs': 2
        }

        # Objective function (cross-validation)
        global bestloss
        bestloss = 1
        def f(params):
            global bestloss
            clf = XGBClassifier(**params)
            clf.random_state = args.seed
            loss = -cross_val_score(clf, val_set_X, val_set_y, scoring='neg_mean_absolute_error', cv=5).mean()
            if loss < bestloss:
                bestloss = loss
            return {'loss': loss, 'status': STATUS_OK}

        # Objective function (Bayesian optimization)
        eval_set = [(hyp_set_X, hyp_set_y), (val_set_X, val_set_y)]
        eval_metric = "auc"
        def objective(space):
            clf=XGBClassifier(
                n_estimators=space['n_estimators'],
                max_depth = int(space['max_depth']),
                gamma = space['gamma'],
                reg_alpha = int(space['reg_alpha']),
                min_child_weight = int(space['min_child_weight']),
                eval_metric = eval_metric,
                random_state = args.seed,
                early_stopping_rounds=10)
            clf.fit(hyp_set_X, hyp_set_y, eval_set=eval_set, verbose=False)

            y_pred = clf.predict(val_set_X)
            accuracy = accuracy_score(val_set_y, y_pred > 0.5)
            #CV_score = cross_val_score(clf, hyp_set_X, hyp_set_y, scoring=eval_metric, cv=5)
            #print("SCORE:", accuracy)
            return {'loss': -accuracy, 'status': STATUS_OK}

        # Optimization algorithm
        trials = Trials()
        # best = fmin(f, space, algo=tpe.suggest, max_evals=args.maxevals, trials=trials)
        best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = 100, trials = trials)

        '''
        # --- Objective function --------------------

        def hyperopt_train_test(params):
            #X_ = X[:]
            #train_set_X_ = train_set_X[:]
            clf = XGBClassifier(**params)
            #reg = XGBRegressor(**params, tree_method="hist", device="cuda")
            clf.random_state = args.seed
            #return -cross_val_score(model, X, yy, scoring='neg_mean_absolute_error', cv=5).mean()
            return -cross_val_score(clf, hyp_set_X, hyp_set_y, scoring='neg_mean_absolute_error', cv=5).mean()

        def objective(space):

            eval_set = [(train_set_X, train_set_y), (test_set_X, test_set_y)]
            eval_metric = "auc"

            clf=XGBClassifier(
                n_estimators=space['n_estimators'],
                max_depth = int(space['max_depth']),
                gamma = space['gamma'],
                reg_alpha = int(space['reg_alpha']),
                min_child_weight = int(space['min_child_weight']),
                eval_metric = eval_metric,
                seed = args.seed,
            )

            clf.fit(hyp_set_X, hyp_set_y, eval_set=eval_set, early_stopping_rounds=10, verbose=False)

            pred = clf.predict(hyp_set_X)
            accuracy = accuracy_score(hyp_set_y, pred > 0.5)
            CV_score = cross_val_score(clf, hyp_set_X, hyp_set_y, scoring=eval_metric, cv=5)
            print("SCORE:", accuracy)
            return {'loss': -accuracy, 'status': STATUS_OK}

        global bestloss
        bestloss = 1

        def f(params):
            global bestloss

            clf = XGBClassifier(**params)
            clf.random_state = args.seed
            loss = -cross_val_score(clf, hyp_set_X, hyp_set_y, scoring='neg_mean_absolute_error', cv=5).mean()

            #loss = hyperopt_train_test(params)
            #print('eval MAE: ', loss)
            #print('| Eval MAE: %-103s |' % str(loss))
            if loss < bestloss:
                bestloss = loss
                #print('new best MAE:', bestloss, params)
                #print('| New best MAE: %-101s |' % (str(bestloss))) #, %-80s |' % (str(bestloss), str(params)))
            return {'loss': loss, 'status': STATUS_OK}

        # --- Optimization algorithm --------------------

        # best_hyperparams gives the optimal parameters that best fit model and better loss function value
        # trials is an object that contains or stores all the relevant information such as hyperparameter,
        # loss-functions for each set of parameters that the model has been trained
        # fmin is an optimization function that minimizes the loss function

        trials = Trials()
        # best = fmin(fn = objective,
        #             space = space,
        #             algo = tpe.suggest,
        #             max_evals = 100,
        #             trials = trials)
        best = fmin(f, space, algo=tpe.suggest, max_evals=args.maxevals, trials=trials)
        # https://docs.rapids.ai/deployment/stable/examples/xgboost-gpu-hpo-job-parallel-ngc/notebook/

        #print('best:')
        # print(best)
        '''

        now_end = time.localtime()
        s = "%04d-%02d-%02d %02d:%02d:%02d" % (now_end.tm_year, now_end.tm_mon, now_end.tm_mday, now_end.tm_hour, now_end.tm_min, now_end.tm_sec)
        print('| Hyperparameter search finished at %-79s |' % s)
        print('|-------------------------------------------------------------------------------------------------------------------|')
        print('| Best hyperparameters                                                                                              |')
        print('|   max_depth:            %-30.0f                                                            |' % best['max_depth'])
        print('|   gamma:                %-30.16f                                                            |' % best['gamma'])
        print('|   reg_alpha:            %-30.1f                                                            |' % best['reg_alpha'])
        print('|   reg_lambda:           %-30.17f                                                            |' % best['reg_lambda'])
        print('|   min_child_weight:     %-30.1f                                                            |' % best['min_child_weight'])
        print('|-------------------------------------------------------------------------------------------------------------------|')

# ----------- Graphs for Hyperparameter tuning optimization

        parameters = ['max_depth', 'gamma', 'reg_alpha', 'reg_lambda', 'min_child_weight']  # , 'criterion']
        f, axes = plt.subplots(ncols=5, figsize=(15, 5))
        cmap = plt.cm.jet
        i = 0
        for i, val in enumerate(parameters):
            # print(i, val)
            xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
            ys = [-t['result']['loss'] for t in trials.trials]
            axes[int(i)].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5, c=cmap(float(i) / len(parameters)))
            axes[int(i)].set_title(val)
        # axes[i,i/3].set_ylim([0.9,1.0])

        graphname = t + '_XGBc_hp_search.png'
        plt.savefig('hyperparameters/' + graphname)

# ----------- Model fitting

    now_start = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now_start.tm_year, now_start.tm_mon, now_start.tm_mday, now_start.tm_hour, now_start.tm_min, now_start.tm_sec)
    print('| Starting model training for eXtreme Gradient Boosting - classifier at %-43s |' %s )

    # load the data set on gpu
    # https://github.com/dmlc/xgboost/issues/9791
    # train_set_X = cp.array(train_set_X)
    # train_set_y = cp.array(train_set_y)
    # test_set_X = cp.array(test_set_X)
    # test_set_y = cp.array(test_set_y)

    eval_set = [(train_set_X, train_set_y), (val_set_X, val_set_y), (test_set_X, test_set_y)]
    eval_metric = ["error", "logloss", "auc"]

    # Booster parameters
    predefined_hp = {
        'max_depth': 26,
        'gamma': 1.0374304322098675,
        'reg_alpha': 20.0,
        'reg_lambda': 0.99026662932484100,
        'min_child_weight': 7.0,
        'n_estimators': 180,
        'seed': 0}

    model = XGBClassifier(
        max_depth=best['max_depth'],
        gamma=best['gamma'],
        reg_alpha=best['reg_alpha'],
        reg_lambda=best['reg_lambda'],
        min_child_weight=best['min_child_weight'],
        n_estimators=180,
        # early_stopping_rounds=1,
        eval_metric=eval_metric,
        random_state=args.seed,
        tree_method="hist",
        device="gpu")

    model.fit(train_set_X, train_set_y, eval_set=eval_set, verbose=False)

    now_end = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now_end.tm_year, now_end.tm_mon, now_end.tm_mday, now_end.tm_hour, now_end.tm_min, now_end.tm_sec)
    print('| Model finished training at %-86s |' % s)

# ----------- Model accuracy

    if args.plotmetrics == 'yes':
        plot_eval_metrics(args, model)

        df = pd.DataFrame()
        df['Features'] = np.array(model.get_booster().feature_names).tolist()
        df['FeatureImportances'], df['Cover'], df['TotalCover'], df['Gain'], df['TotalGain'], df['Weight'] = XGB_feature_importance(args, model)
        df['SHAP'] = SHAP_values(args, model, train_set_X)
        df['Permutations'] = permutation_imp(args, model, test_set_X, test_set_y)

        #df = pd.concat([df, XGB_feature_importance(args, model)], axis=1)
        #df = pd.concat([df, SHAP_values(args, model, train_set_X)], axis=1)
        #df = pd.concat([df, permutation_imp(args, model, test_set_X, test_set_y)], axis=1)
        #df.to_csv(args.name + 'features.csv', index=False)

        if os.path.isfile(args.name + '_features.csv') is True:
            LR = pd.read_csv(args.name + '_features.csv', index_col=False)
            df = pd.concat([df, LR['Coefficients']], axis=1)
            df.to_csv(args.name + '_features.csv', index=False)
        else:
            df.to_csv(args.name + '_features.csv', index=False)

    # make predictions for train data
    y_pred_train = model.predict(train_set_X)
    predictions_train = [round(value) for value in y_pred_train]
    # evaluate predictions
    accuracy_train = accuracy_score(train_set_y, predictions_train)
    print('| Model accuracy on the train set: %.2f%%                                                                           |' % (accuracy_train * 100.0))

    # make predictions for validation data
    y_pred_val = model.predict(val_set_X)
    predictions_val = [round(value) for value in y_pred_val]
    # evaluate predictions
    accuracy_val = accuracy_score(val_set_y, predictions_val)
    print('| Model accuracy on the validation set: %.2f%%                                                                       |' % (accuracy_val * 100.0))

    # make predictions for test data
    y_pred_test = model.predict(test_set_X)
    predictions_test = [round(value) for value in y_pred_test]
    # evaluate predictions
    accuracy_test = accuracy_score(test_set_y, predictions_test)
    print('| Model accuracy on the test set: %.2f%%                                                                            |' % (accuracy_test * 100.0))

    # scores = cross_val_score(model, train_set_X, train_set_y, scoring='neg_mean_absolute_error', cv=5, n_jobs=1)

    if args.hyperparameter != '0':
        print('|-------------------------------------------------------------------------------------------------------------------|')
        print('| Training summary                                                                                                  |')
        print('|   Best hyperparameters                                                                                            |') # %-90s |' % (str(best)))
        print('|      max_depth:                 %-22.0f                                                            |' % best['max_depth'])
        print('|      gamma:                     %-22.16f                                                            |' % best['gamma'])
        print('|      reg_alpha:                 %-22.1f                                                            |' % best['reg_alpha'])
        print('|      reg_lambda:                %-22.17f                                                            |' % best['reg_lambda'])
        print('|      min_child_weight:          %-22.1f                                                            |' % best['min_child_weight'])
        #print('|   Mean MAE:                     %.3f (%.3f)                                                                     |' % (scores.mean(), scores.std()))
        #print('|   5-fold cross validation:      %-81s |' % (str(scores)))
        #print('|   Test score:                   %-81s |' % (str(scored['score'])))
        print('|-------------------------------------------------------------------------------------------------------------------|')
    #print('| Accuracy score: %-90s |' % (str(accuracy_score(test_set_y, prediction_y))))
    #print('| Test Score: ' scored, accuracy_score(scores_test, scores_pred_test))

# ----------- Saving the model
    modelname = t + '_XGBc_model.json'
    model.save_model('models/XGBc_' + args.name + '.json')
    #print('| Model saved as %s_XGBc_model.json"                                                                |' % t)
    print('| Model saved as XGBc_%s.json"                                                                |' % args.name)

