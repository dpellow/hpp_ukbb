import os

#######################################################################
# TODO !!
os.environ["OMP_NUM_THREADS"] = "3" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "3" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "3" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "3" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "3" # export NUMEXPR_NUM_THREADS=1

from LabUtils.Utils import mkdirifnotexists

from LabData.DataAnalyses.UKBB_10k.process_predictions import *


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from glob import glob

from LabQueue.qp import fakeqp as qp # qp # fakeqp as qp #
from LabUtils.addloglevels import sethandlers

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV,  ShuffleSplit, cross_validate
from sklearn.metrics import make_scorer, r2_score, precision_recall_curve, explained_variance_score
from sklearn.linear_model import Lasso
from sklearn.utils import resample

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from scipy.stats import spearmanr, pearsonr

class BinaryRoundingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.binary_columns = ['gender', 'dominant_hand'] # TODO: add here


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        ####################################
        binary_columns = X.columns[(X.nunique(dropna=True) == 2) &
                                    (X.apply(lambda x: np.isin(x.dropna().unique(), [0, 1]).all()))]
        #print(binary_columns)
        columns_to_round = list(set(binary_columns) & set(X_transformed.columns))
        #####################################
        #columns_to_round = list(set(self.binary_columns) & set(X_transformed.columns))
        X_transformed[columns_to_round] = np.round(X_transformed[columns_to_round])
        return X_transformed

class NormalizationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler=StandardScaler().set_output(transform='pandas')
        self.binary_columns = ['gender', 'dominant_hand'] # TODO: add here
        self.transform_cols = []

    def fit(self, X, y=None):
        self.binary_columns = X.columns[(X.nunique(dropna=True) == 2) &
                                    (X.apply(lambda x: np.isin(x.dropna().unique(), [0, 1]).all()))]
        self.transform_cols = list(set(X_transformed.columns) - set(binary_columns))
        print(self.transform_cols)

        self.scaler.fit(X[:, self.transform_cols])
        return self

    def transform(self, X):
        X_transformed = X.copy()

        self.binary_columns = X.columns[(X.nunique(dropna=True) == 2) &
                                    (X.apply(lambda x: np.isin(x.dropna().unique(), [0, 1]).all()))]
        self.transform_cols = list(set(X_transformed.columns) - set(binary_columns))
        print(self.transform_cols)

        X_transformed[:,self.transform_cols] = self.scaler.transform(X_transformed[:,self.transform_cols])
        return X_transformed


def remove_rare(df,val=np.nan,frac=.1):
    if np.isnan(val):
        to_drop = df.isna().sum()
    else:
        to_drop = (df==val).sum()
    to_drop = to_drop[to_drop > len(df)*(frac)].index#(1-frac)].index
    df = df.drop(to_drop, axis=1)
    return df

def run_lasso(preds,vars_df,drop_missing=False,dropval=np.nan,normalize=False):
    n_splits = 5 #TODO: should be an arg

    X_resampled, y_resampled = resample(vars_df, preds)
    X_resampled = X_resampled.reset_index(drop=True)
    y_resampled = y_resampled.reset_index(drop=True)

    folds_outcomes = pd.DataFrame(index=range(n_splits), columns = ['score','alpha']+vars_df.columns.tolist())
    y_preds = pd.DataFrame(index=y_resampled.index)#preds.index)
    # in each fold
    kfolds = KFold(shuffle=True,n_splits = n_splits)

    for i, (train_idx, test_idx) in enumerate(kfolds.split(X_resampled, y_resampled)):#vars_df, preds)):
        vars_train, vars_test = X_resampled.iloc[train_idx, :], X_resampled.iloc[test_idx, :]#vars_df.iloc[train_idx, :], vars_df.iloc[test_idx, :]
        if drop_missing:
            vars_train = remove_rare(vars_train, val=dropval, frac=.667)
            #print(vars_train.columns.to_list())
            vars_test = vars_test.reindex(vars_train.columns, axis=1)
        preds_train, preds_test =  y_resampled.iloc[train_idx], y_resampled.iloc[test_idx]# preds.iloc[train_idx], preds.iloc[test_idx]

        # impute missing vars
        imp = Pipeline([
            ('imputer', IterativeImputer(n_nearest_features=15)),#verbose=2)),
            ('rounding', BinaryRoundingTransformer()),
        ])
        imp[0].set_output(transform='pandas')
        if normalize:
            imp.steps.append(['normalize',NormalizationTransformer()])
        pipe = Pipeline([("imp", imp),
                         ("lasso", Lasso())])
        params = {"lasso__alpha" : [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1,2,5,10,100]}
        search = GridSearchCV(pipe, params, cv=3, n_jobs=3)
        search.fit(vars_train,preds_train)

        folds_outcomes.loc[i,['score','alpha']] = [search.best_score_, search.best_params_['lasso__alpha']]
        folds_outcomes.loc[i,vars_train.columns.tolist()] = search.best_estimator_[-1].coef_
        #folds_outcomes.loc[i, vars_df.columns.tolist()] = search.best_estimator_[-1].coef_

        # predict on held out, save predictions
        y_pred = search.best_estimator_.predict(vars_test)
        # print(test_idx)
        # print(vars_test.index)
        y_preds.loc[vars_test.index,'lasso_pred'] = y_pred#'lasso_pred'] = y_pred

    # evaluate across all folds
    evals = pd.DataFrame(index=[0])
    evals.loc[0,'r2'] = r2_score(y_resampled,y_preds)#(preds,y_preds)
    evals.loc[0,'ev'] = explained_variance_score(y_resampled,y_preds)#preds,y_preds)
    evals.loc[0,'pearson_r'], evals.loc[0,'pearson_p'] = pearsonr(y_preds.values.ravel(),y_resampled.values.ravel())#preds.values.ravel())
    evals.loc[0, 'spearman_r'], evals.loc[0, 'spearman_p'] = spearmanr(y_preds.values.ravel(), y_resampled.values.ravel())#preds.values.ravel())
    evals.loc[0,'num_pred'] = y_preds.shape[0]

    # return evaluation
    return evals,folds_outcomes

def loader_vs_outcome(loader, outcome_id, outcome_name, i, normalize=False,above_baseline=False):
    # load outcome predictions
    try:
        preds = load_outcome(outcome_id) # from process predictions
    except:
        return
    # load loader variables
    loader_df = get_loader_vars(loader,drop_frac=0.0)
    if above_baseline and loader != 'age_gender_bmi':
        age_gender_bmi = load_age_gender_bmi()
        merged = age_gender_bmi.merge(loader_df, left_index=True, right_index=True)
        loader_df = merged.dropna(subset=['age', 'gender', 'bmi'])

    shared_subjects = preds.index.intersection(loader_df.index)
    preds = preds.loc[shared_subjects]
    loader_df = loader_df.loc[shared_subjects]

    # drop columns that overlap input features
    drop_features_dict = {'sleep': ['MeanPRSleep'],
                          'ecg': ['hr_bpm'],
                          'abi': ['l_brachial_pressure', 'r_brachial_pressure'],
                          # 'l_ankle_pressure', 'r_ankle_pressure'
                          'cgm': ['eA1C'],
                          'dexa': ['body_comp_trunk_region_percent_fat', 'body_comp_total_region_percent_fat'],
                          'diet': [],
                          }
    if loader in drop_features_dict:
        drop_cols = drop_features_dict[loader]
        loader_df = loader_df.drop(columns = drop_cols)

    dropval= -9 if loader=='microbiome' else np.nan

    evals,cv_perf = run_lasso(preds,loader_df,normalize=normalize,drop_missing=True,dropval=dropval)

    if above_baseline:
        agb_evals, agb_cv_perf = run_lasso(preds,loader_df[['age','gender','bmi']],normalize=normalize,drop_missing=True,dropval=dropval)

        fname = f'lasso_CV_age_gender_bmi_from_{loader}_outcomes_{i}.csv'
        if normalize:
            fname = "normalized_" + fname


        agb_cv_perf.to_csv(os.path.join(topdir, str(outcome_id), 'tenk', fname), index=True)

        fname = f'lasso_eval_age_gender_bmi_from_{loader}_{i}.csv'
        if normalize:
            fname = "normalized_" + fname

        agb_evals.loc[0, 'loader'] = loader
        agb_evals.loc[0, 'condition'] = outcome_name
        agb_evals.loc[0, 'cancer'] = (str(outcome_id)[0] == 'C')
        agb_evals.loc[0, 'type'] = 'loader'
        agb_evals.to_csv(os.path.join(topdir, str(outcome_id), 'tenk', fname), index=True)


    fname = f'lasso_CV_{loader}_outcomes_{i}.csv'
    if normalize:
        fname = "normalized_"+fname
    if above_baseline:
        fname = 'age_gender_bmi_'+fname

    cv_perf.to_csv(os.path.join(topdir,str(outcome_id),'tenk',fname),index=True)


    fname = f'lasso_eval_{loader}_{i}.csv'
    if normalize:
        fname = "normalized_"+fname
    if above_baseline:
        fname = 'age_gender_bmi_'+fname

    evals.loc[0,'loader'] = f'{loader}_{i}'
    evals.loc[0,'condition'] = outcome_name
    evals.loc[0,'cancer'] = (str(outcome_id)[0] == 'C')
    evals.loc[0,'type'] = 'loader'
    evals.to_csv(os.path.join(topdir,str(outcome_id),'tenk',fname),index=True)
    return evals

def system_vs_outcome(system_file, outcome_id, outcome_name, i, normalize=False):
    # load outcome predictions
    try:
        preds = load_outcome(outcome_id) # from process predictions
    except:
        return
    # load loader variables
    loader_df = get_system_vars(system_file)
    shared_subjects = preds.index.intersection(loader_df.index)
    preds = preds.loc[shared_subjects]

    loader_df = loader_df.loc[shared_subjects]

    evals,cv_perf = run_lasso(preds,loader_df,normalize)

    system = os.path.splitext(os.path.basename(system_file))[0]

    fname = f'lasso_CV_{system}_system_outcomes_{i}.csv'
    if normalize:
        fname = 'normalized_'+fname
    cv_perf.to_csv(os.path.join(topdir,str(outcome_id),'tenk',
                                fname),index=True)

    fname = f'lasso_eval_{system}_{i}.csv'
    if normalized:
        fname = 'normalized_'+ fname
    evals.loc[0,'loader'] = system
    evals.loc[0,'condition'] = outcome_name
    evals.loc[0,'cancer'] = (str(outcome_id)[0] == 'C')
    evals.loc[0,'type'] = 'system'
    evals.to_csv(os.path.join(topdir, str(outcome_id), 'tenk', fname), index=True)
    return evals




if __name__ == '__main__':

    sethandlers()

    old_cwd = os.getcwd()
    os.chdir(OUTPATH)

    loaders = ['microbiome']#['dietnew', 'sleep',  'ecg', 'cgm', 'dexa', \
               # 'retina', 'microbiome', 'age_gender_bmi','carotid','abi','metabolomics','liver']
     #          'bloodtests', 'bodymeasures', 'age_gender_bmi','carotid','abi','metabolomics','liver']

    systems_files = glob(DATAPATH+'/body_systems/Xs/*.csv')
    systems_files = [f for f in systems_files if 'male' not in f]

    outcomes = pd.read_csv(OUTCOMES_LIST).astype({"UKBB Field ID": 'int'})
    outcomes_dict = dict(zip(outcomes['UKBB Field ID'].values, outcomes['UKBB Description'].values))
    for o in outcomes_dict:
        outcomes_dict[o] = '('.join(outcomes_dict[o][:-1].split('(')[1:])


    test_dict = {131036 : "alzheimer's disease"}

    cancer_outcomes = pd.read_csv(CANCER_OUTCOMES)
    cancer_outcomes_dict = dict(zip(cancer_outcomes['ICD10'].values, cancer_outcomes['Type of cancer'].values))

    normalize=False
    above_baseline=True

    runlist = []
    results_list = []

    with qp(jobname='lasso', _delete_csh_withnoerr=False, q=['himem7.q'], _trds_def=9, max_r=10000,
            _mem_def='5G') as q:
        q.startpermanentrun()
        for i in range(200,201):
            runlist += [q.method(loader_vs_outcome, (loader, outcome, outcomes_dict[outcome], i, normalize,above_baseline))
                        for loader in loaders for outcome in outcomes_dict.keys()] # [131342]]#


        # runlist += [q.method(loader_vs_outcome, (loader, outcome, cancer_outcomes_dict[outcome]))
        #             for loader in loaders for outcome in cancer_outcomes_dict.keys()]
        #
        # runlist += [q.method(system_vs_outcome, (system_file, outcome, outcomes_dict[outcome]))
        #             for system_file in systems_files for outcome in outcomes_dict.keys()]
        #
        # runlist += [q.method(system_vs_outcome, (system_file, outcome, cancer_outcomes_dict[outcome]))
        #             for system_file in systems_files for outcome in cancer_outcomes_dict.keys()]

        results_list = q.waitforresults(runlist)

    # turn the results into dataframes for all loaders/systems vs all outcomes/cancers
    all_results = pd.concat(results_list,axis=0)

    loaders_res = all_results.loc[(all_results['type'] == 'loader') & (all_results['cancer'] == False)]
    fname = 'lasso_results_loaders_v_conditions_new.csv' if not normalize else 'normalized_lasso_results_loaders_v_conditions_new.csv'
    if above_baseline: fname = "above_baseline_" + fname
    loaders_res.drop(columns = ['type','cancer']).to_csv(os.path.join(topdir,fname))

    # loaders_res = all_results.loc[(all_results['type'] == 'loader') & (all_results['cancer'] == True)]
    # fname = 'lasso_results_loaders_v_cancers.csv' if not normalize else 'normalized_lasso_results_loaders_v_cancers.csv'
    # if above_baseline: fname = "above_baseline_" + fname
    # loaders_res.drop(columns=['type', 'cancer']).to_csv(os.path.join(topdir, fname))

    # systems_res = all_results.loc[(all_results['type'] == 'system') & (all_results['cancer'] == False)]
    # fname = 'lasso_results_systems_v_conditions.csv' if not normalize else 'normalized_lasso_results_systems_v_conditions.csv'
    # if above_baseline: fname = "above_baseline_" + fname
    # systems_res.drop(columns = ['type','cancer']).to_csv(os.path.join(topdir,fname))
    # systems_res = all_results.loc[(all_results['type'] == 'system') & (all_results['cancer'] == True)]
    # fname = 'lasso_results_systems_v_cancers.csv' if not normalize else 'normalized_lasso_results_systems_v_cancers.csv'
    # if above_baseline: fname = "above_baseline_" + fname
    # systems_res.drop(columns=['type', 'cancer']).to_csv(os.path.join(topdir, fname))

    os.chdir(old_cwd)







