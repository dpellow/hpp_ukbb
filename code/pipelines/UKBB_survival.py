import os

import pandas as pd
import numpy as np
import re

import logging

import matplotlib

import pickle

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from LabData.DataAnalyses.TenK_Trajectories.utils import add_age_gender_to_df

from LabData.DataAnalyses.TenK_Trajectories.UKBB_10K import ukbb_10k_unit_conversion

from LabData.DataLoaders.UkbbLoader import UkbbLoader

from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabData.DataLoaders.LifeStyleLoader import LifeStyleLoader
from LabData.DataLoaders.FamilyMedicalConditionsLoader import FamilyMedicalConditionsLoader
from LabData.DataLoaders.MentalLoader import MentalLoader
from LabData.DataLoaders.HormonalStatusLoader import HormonalStatusLoader
from LabData.DataLoaders.FamilyHistoryLoader import FamilyHistoryLoader
from LabData.DataLoaders.DEXALoader import DEXALoader
from LabData.DataLoaders.SymptomsLoader import SymptomsLoader

from LabQueue.qp import qp #fakeqp as qp

from LabUtils.Utils import mkdirifnotexists

from lifelines import CoxPHFitter
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

import pingouin as pg


LOADER_MAPPING = {'BodyMeasures': BodyMeasuresLoader,
                 'BloodTests': BloodTestsLoader,
                 'FamilyMedicalConditions': FamilyMedicalConditionsLoader,
                 'Mental': MentalLoader,
                 'HormonalStatus': HormonalStatusLoader,
                 'LifeStyle': LifeStyleLoader,
                 'FamilyHistory': FamilyHistoryLoader,
                  'Symptoms': SymptomsLoader}

FEATURE_MAPPING = '/net/mraid20/export/jafar/UKBioBank/davidpel/UKBB_data/joint_features.csv' #'/home/davidpel/PycharmProjects/UKBB_10k/joint_features.csv'
OUTCOMES_LIST = '/net/mraid20/export/jafar/UKBioBank/davidpel/UKBB_data/outcomes.csv'
CANCER_OUTCOMES = '/net/mraid20/export/jafar/UKBioBank/davidpel/UKBB_data/cancer_outcomes.csv'
OUTCOME_GROUPS = '/net/mraid20/export/jafar/UKBioBank/davidpel/UKBB_data/outcome_groups.json'
OUTCOMES_DATES = '/net/mraid20/export/jafar/UKBioBank/davidpel/UKBB_data/all_outcome_dates.csv'
COMPETING_PATH = '/net/mraid20/export/jafar/UKBioBank/davidpel/UKBB_data/competing.json'
OUTPATH = '/net/mraid20/export/jafar/UKBioBank/davidpel/UKBB_10K_outputs'
CACHEPATH = '/net/mraid20/export/jafar/UKBioBank/davidpel/UKBB_10K_outputs/cache'
DATAPATH = '/net/mraid20/export/jafar/UKBioBank/davidpel/UKBB_data/'

logger = logging.getLogger("logger")

############################ NOTES #####################################################################################
# Multiple-choice where more than one option can be selected is handled differently in UKB and 10K
# 10K: All selected choices are one-hot encoded. The column names are 'feature_name__selection 'where selection is the google
#      translation of the hebrew option, or in some cases, the hebrew option itself
# UKB: A single column is used with a list whose length is the maximum number of options selected by any single individual
#      Each entry in the list is the code of one of the options selected or nan after all options have been listed
#      In our loader the list elements are loaded as independent columns named 'feature_name (i)' where i is the index in the list
# Here I keep the column as feature names in ukbb and convert to 1hot only in process_joint. This let me maintain most of the conversion
# and mapping code. The 1hot names are based on the UKB dictionaries and so the 10K column names are converted to match those
#
#
#
########################################################################################################################

def get_10k_multi_basename(feature_name):
    if "__" in feature_name:
        feature_name = '__'.join(feature_name.split("__")[:-1])
    return feature_name

def load_prep_ukbb_mapped_features(features_table):
    ukbb_loader = UkbbLoader()
    ukbb_data = ukbb_loader.load_ukbb(which = features_table["UKBB Field ID"].to_list())
    ukbb_data.rename(columns={k: k.split(' - visit')[0] for k in ukbb_data.columns}, inplace=True)
    ukbb_data = process_repeating_columns(ukbb_data,features_table)
    ukbb_data = ukbb_data.fillna(np.nan)

    ukbb_data = ukbb_data.dropna(how='all')
    logger.info("Loaded UKBB data")
    return ukbb_data

def load_prep_10k_mapped_features(features_table):
    tenk_data = pd.DataFrame()
    ## only run on people who have come in for the first visit
    bm_md = BodyMeasuresLoader().get_data(study_ids=['10K'], research_stage=['baseline']).df_metadata
    reg_ids = bm_md.index.get_level_values(0).tolist() # ids of everyone who has come in for a visit

    for loader in set(pd.unique(features_table['Loader']).tolist()).intersection(set(LOADER_MAPPING.keys())):
        loader_features = features_table['10K Name'][features_table['Loader'] == loader].to_list()
        if loader == 'BodyMeasures':
            data = add_age_gender_to_df(
                LOADER_MAPPING[loader]().get_data(study_ids=['10K'], research_stage=['baseline'],
                                                  groupby_reg='first',reg_ids=reg_ids, min_col_present_frac=0.333)).df#,
                                                  #cols=loader_features)).df
        else:
            data = LOADER_MAPPING[loader]().get_data(study_ids=['10K'], research_stage=['baseline'],
                                                     groupby_reg='first',reg_ids=reg_ids, min_col_present_frac=0.333
                                                     ).df #cols=loader_features,
        data = data.reset_index(level=[1]).drop(['Date'], axis=1)
        # now drop the columns that aren't in feature_table - have to do it this way
        # because of the multiple choice columns
        keep_cols = [col for col in data.columns.tolist() if (col in loader_features or get_10k_multi_basename(col) in loader_features)]
        data = data[keep_cols]
        tenk_data = pd.concat([tenk_data, data], axis=1)

    tenk_data = tenk_data.fillna(np.nan)
    tenk_data = tenk_unit_correction(features_table,tenk_data.copy())
    tenk_data = tenk_data.dropna(how='all')
    logger.info("Loaded tenk data")
    return tenk_data


def load_joint_features(regularized=True, use_cache=True): # TODO: remove this - it's in the class now
    if use_cache:
        tenk_path = os.path.join(CACHEPATH,'tenk_processed_features.csv')
        ukbb_path = os.path.join(CACHEPATH, 'ukbb_processed_features.csv')
        if os.path.exists(tenk_path) and os.path.exists(ukbb_path):
            logger.info("Loading features from cache: {}, {}".format(tenk_path,ukbb_path))
            return pd.read_csv(tenk_path,index_col=0), pd.read_csv(ukbb_path,index_col=0)

    features_table = pd.read_csv(FEATURE_MAPPING).astype({"UKBB Field ID":'int'})
    features_table = features_table.dropna(how='any')
    logger.info("Loading features")
    ukbb_data = load_prep_ukbb_mapped_features(features_table)
    tenk_data = load_prep_10k_mapped_features(features_table)

    ukbb_data = rename_ukbb_fields_to_match_tenks_names(ukbb_data, tenk_data, features_table)

    logger.info("Processing features")
    tenk_data, ukbb_data = process_joint_features(features_table,tenk_data,ukbb_data,regularized)

    if use_cache:
        if not os.path.isdir(CACHEPATH):
            os.makedirs(CACHEPATH)
        tenk_data.to_csv(tenk_path)
        ukbb_data.to_csv(ukbb_path)

    return tenk_data, ukbb_data

def process_repeating_columns(ukbb_data,features_table):
    showcase_dict = UkbbLoader().ukbb_data_dic_showcase
    coding_df = UkbbLoader().ukbb_coding
    repeating_columns = []
    ukbb_data_reform = pd.DataFrame()
    for col in ukbb_data.columns:
        if re.search('\(\d\)', col):
            if col.split('(')[0] not in repeating_columns:
                repeating_columns.append(re.sub('\(\d+\)', '', col))
        else:
            ukbb_data_reform[col] = ukbb_data[col]
    for col in repeating_columns:
        # processing depends on datatype
        feature_row = features_table.loc[features_table["UKBB Description"].str.contains(col.replace('(', '\(').replace(')', '\)'))]
        ukbb_field_id = feature_row["UKBB Field ID"].item()
        showcase_metadata = showcase_dict.loc[showcase_dict["FieldID"] == ukbb_field_id]
        datatype = showcase_metadata["ValueType"].item()
        if datatype == "Categorical multiple":
            ukbb_data_reform[col] = ukbb_data.filter(regex=col.replace('(', '\(').replace(')', '\)')).values.tolist()
        else:
            ukbb_data_reform[col] = ukbb_data.filter(regex=col.replace('(', '\(').replace(')', '\)')).mean(axis=1)
    return ukbb_data_reform

def get_mean_of_repeating_columns(ukbb_data):
    repeating_columns = []
    ukbb_data_reform = pd.DataFrame()
    for col in ukbb_data.columns:
        if re.search('\(\d\)', col):
            if col.split('(')[0] not in repeating_columns:
                repeating_columns.append(col.split('(')[0])
        else:
            ukbb_data_reform[col] = ukbb_data[col]
    for col in repeating_columns:
        ukbb_data_reform[col] = ukbb_data.filter(regex=col.replace('(', '\(').replace(')', '\)')).mean(axis=1)
    return ukbb_data_reform

def get_max_of_repeating_columns(ukbb_data):
    repeating_columns = []
    ukbb_data_reform = pd.DataFrame()
    for col in ukbb_data.columns:
        if re.search('-\d', col):
            if col.split('-')[0] not in repeating_columns:
                repeating_columns.append(col.split('-')[0])
        else:
            ukbb_data_reform[col] = ukbb_data[col]
    for col in repeating_columns:
        ukbb_data_reform[col] = ukbb_data.filter(regex=col).max(axis=1)
    return ukbb_data_reform



def tenk_unit_correction(features_table, data):
    loaders_conv_dict = ukbb_10k_unit_conversion.tenk_to_ukbb_unit_conv
    for col in data.columns:
        loader = features_table.loc[(features_table["10K Name"]==col) | (features_table["10K Name"]==get_10k_multi_basename(col))]["Loader"].item()+"Loader"
        if loader in loaders_conv_dict:
            func = loaders_conv_dict[loader].get(col,None)
            if func is not None:
                data[col] = data[col].map(func)
    # fix names of multiple choice columns
    trans_dict = ukbb_10k_unit_conversion.tenk_to_ukbb_choice_conv
    for loader in features_table["Loader"].values:
        if loader+"Loader" in trans_dict:
            data.rename(trans_dict[loader+"Loader"],axis='columns',inplace=True)
    data = data.groupby(level=0,axis=1).max()#.apply(lambda x : x.apply(max(axis=1)))

    return data

# def rename_ukbb_fields_to_match_tenks_names(ukbb_data, tenk_data, features_table):
#     names = {}
#     for tenk_col_name in tenk_data.columns:
#         ukbb_col_name = features_table[features_table['10K Name'] == tenk_col_name]['UKBB Description'].values[0]
#         names[ukbb_col_name] = tenk_col_name
#     renamed_ukbb = ukbb_data.rename(columns=names)
#     renamed_ukbb = renamed_ukbb.reindex(columns=list(tenk_data.columns))
#     return renamed_ukbb

def rename_ukbb_fields_to_match_tenks_names(ukbb_data, tenk_data, features_table):
    names = dict(zip(features_table['UKBB Description'],features_table['10K Name']))
    renamed_ukbb = ukbb_data.copy().rename(columns=names)
 #   renamed_ukbb = renamed_ukbb.reindex(columns=list(tenk_data.columns))
    return renamed_ukbb


def remove_outliers_for_dfs(data, min_quantile=0.01, max_quantile=0.99):
    data_nol = data[data.between(data.quantile(min_quantile), data.quantile(max_quantile))]
    return data_nol

def process_joint_features(features_table,tenk_data,ukbb_data,regularized=True):
    showcase_dict = UkbbLoader().ukbb_data_dic_showcase
    coding_df = UkbbLoader().ukbb_coding

    loader_counters = {}

    for feature in features_table["10K Name"]:

        feature_row = features_table.loc[features_table["10K Name"] == feature]
        ukbb_field_id = feature_row["UKBB Field ID"].item()
        showcase_metadata = showcase_dict.loc[showcase_dict["FieldID"] == ukbb_field_id]
        datatype = showcase_metadata["ValueType"].item()

        # multiple category features have different names in the tenk_data
        # the special cases are features that are multiple in the tenk but not ukbb
        if feature not in tenk_data.columns and \
                datatype != "Categorical multiple" and \
                feature != 'cereal_type' and feature != 'bread_type_mainly_eat' and 'tobacco_type_' not in feature: # gah
            if feature in ukbb_data.columns:
                ukbb_data.drop(feature,axis=1,inplace=True)
            continue
        # also drop features in less than 1/5 UKB participants
        if feature in ukbb_data.columns:
            if ukbb_data[feature].isna().sum() > len(ukbb_data[feature])*0.8:
                ukbb_data.drop(feature, axis=1, inplace=True)
                tenk_data.drop(tenk_data.filter(regex=feature).columns.tolist(), axis=1, inplace=True)
                continue

        loader_counters[feature_row['Loader'].values[0]] = loader_counters.get(feature_row['Loader'].values[0],0) + 1

        if datatype == "Continuous":
            # remove outliers for continuous numerical features
            tenk_data[feature] = remove_outliers_for_dfs(tenk_data[feature])
            ukbb_data[feature] = remove_outliers_for_dfs(ukbb_data[feature])

        elif datatype == "Integer":
            tenk_data[feature] = tenk_data[feature].astype('float')
            ukbb_data[feature] = ukbb_data[feature].astype('float')
            #TODO: probably need different treatment of ages, time periods etc.

        elif "categorical" in datatype.lower():
            # categorical variables should be one-hot encoded
            # drop "don't know", "prefer not to answer"
            coding_scheme = showcase_metadata["Coding"].item()
            coding_dict = coding_df.loc[coding_scheme][['Value', 'Meaning']].set_index('Value').to_dict()['Meaning']
            dropped = regularized
            for code,meaning in coding_dict.items():
                #TODO: Assess if this v is the right way to handle these options
                if "do not know" in meaning.lower() or "prefer not to answer" in meaning.lower() or "not employed" in meaning.lower(): # not employed treated as don't know in loader
                    tenk_data.drop('%s (%s)' % (feature, meaning), axis=1, inplace=True, errors='ignore') #
                    dropped = True
                    continue
                    #TODO: something else?
                if 'multiple' in datatype.lower():
                    #TODO: remove this v
                    if '%s (%s)' % (feature, meaning) not in tenk_data.columns:
                         continue
                    ukbb_data['%s (%s)' % (feature, meaning)] = 1. * (ukbb_data[feature].apply(lambda x: float(code) in x))  # (ukbb_data.filter(regex='%s\(?' % feature) == meaning)
                    ukbb_data.loc[ukbb_data[feature].map(lambda x: all(np.isnan(x))), '%s (%s)' % (feature, meaning)]= np.nan
                    if feature_row['Loader'].item() == "Symptoms" or feature_row['Loader'].item() == "LifeStyle":
                        #TODO ALL loaders should process multiple choices this way
                        pass
                    else:
                        # TODO: These should be handled like symptoms
                        tenk_data['%s (%s)' % (feature, meaning)] = 1. * (tenk_data[feature].astype(float) == float(code))  # (ukbb_data.filter(regex='%s\(?' % feature) == meaning)
                        tenk_data['%s (%s)' % (feature, meaning)][tenk_data[feature].isna()] = np.nan
                    # if tenk_data['%s (%s)' % (feature, meaning)].isna().sum() > len(tenk_data['%s (%s)' % (feature, meaning)])*0.5 \
                    if len(tenk_data.filter(like=(feature)).dropna(how='all')) < len(tenk_data['%s (%s)' % (feature, meaning)])*0.333 \
                      or '%s (%s)' % (feature, meaning) not in tenk_data.columns:
                        # these weren't dropped yet - drop them now
                        ukbb_data.drop('%s (%s)' % (feature, meaning), axis=1, inplace=True)
                        tenk_data.drop('%s (%s)' % (feature, meaning), axis=1, inplace=True)
                        dropped = True
                else:
                    ukbb_data['%s (%s)' % (feature, meaning)] = 1.*(ukbb_data[feature].astype(float)==float(code))#(ukbb_data.filter(regex='%s\(?' % feature) == meaning)
                    ukbb_data.loc[ukbb_data[feature].isna(), '%s (%s)' % (feature, meaning)] = np.nan
                    if feature == 'bread_type_mainly_eat' or feature=='cereal_type' or 'tobacco_type_' in feature: continue # gah - this is not MC in UKB but is in 10k
                    tenk_data['%s (%s)' % (feature, meaning)] = 1.*(tenk_data[feature].astype(float)==float(code))#(ukbb_data.filter(regex='%s\(?' % feature) == meaning)
                    tenk_data.loc[tenk_data[feature].isna(), '%s (%s)' % (feature, meaning)]=np.nan
            if not dropped:
                # need to drop one category WHEN NOT REGULARIZING
                #TODO: Separate handling for regularized/non-regularized
                for code, meaning in coding_dict.items():
                    if code == 0 or "no" in meaning.lower():
                        col = '%s (%s)' % (feature, meaning)
                        try:
                            ukbb_data.drop(col, axis=1, inplace=True)
                            tenk_data.drop(col, axis=1, inplace=True)
                            dropped = True
                            break
                        except KeyError:
                            continue
            if not dropped:
                for meaning in list(coding_dict.values()):
                    #meaning = list(coding_dict.values())[0]
                    col = '%s (%s)' % (feature, meaning)
                    if col in ukbb_data.columns and col in tenk_data.columns:
                        ukbb_data.drop(col, axis=1, inplace=True)
                        tenk_data.drop(col, axis=1, inplace=True)
                        break

            ukbb_data.drop(feature, axis=1, inplace=True)
            tenk_data.drop(feature, axis=1, inplace=True,errors='ignore')

        elif datatype == "Date":
            pass #TODO

        else:
            print(datatype)
            #TODO

    # catch some feature/meaning types that were not matched between the data sets
    ukbb_features = ukbb_data.columns.tolist()
    tenk_features = tenk_data.columns.tolist()
    for feature in ukbb_features:
        if feature not in tenk_data.columns:
            logger.info("Dropping {} from tenk".format(feature))
            ukbb_data.drop(feature,axis=1,inplace=True)
    for feature in tenk_features:
        if feature not in ukbb_data.columns:
            logger.info("Dropping {} from ukbb".format(feature))
            tenk_data.drop(feature,axis=1,inplace=True)

    ukbb_data = ukbb_data.fillna(np.nan)
    ukbb_data = ukbb_data.dropna(how='all')
    tenk_data = tenk_data.fillna(np.nan)
    tenk_data = tenk_data.dropna(how='all')
    logger.info("Processed data")
    return tenk_data, ukbb_data

def get_ukbb_date_fields(field_ids,instances=[0]):
    df = UkbbLoader().load_ukbb(which=field_ids,instances=instances, rename_cols=False)
    df = df.apply(pd.to_datetime)
    if len(instances) > 1:
        df = get_max_of_repeating_columns(df)
    df.rename(columns={k: int(k.split('-')[0]) for k in df.columns}, inplace=True)
    return df

def get_ukbb_cohort(outcome_field_id,use_cache=True):

    if use_cache:
        cache_file = os.path.join(CACHEPATH,"cohort_{}.csv".format(outcome_field_id))
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file,index_col=0)

    # determine which participants had the outcome *after* registration
    # and their time to event
    # TODO: exclude participants that....
    # TODO: left censoring for people with outcome before registration date ?
    # determine censoring time of everyone else?
    logger.info("Getting cohort for field {}".format(outcome_field_id))
    assessment_date_field_id = 53
    death_date_field_id = 40000
    lost_to_follow_up_field_id = 191
    outcome_dates = get_ukbb_date_fields([assessment_date_field_id, outcome_field_id,
                                          death_date_field_id, lost_to_follow_up_field_id])


    outcome_dates['durations'] = outcome_dates[outcome_field_id] - outcome_dates[assessment_date_field_id]
    outcome_dates['event'] = outcome_dates['durations']>pd.Timedelta(0)

    # get all censoring durations
    # # deaths of non-cases
    outcome_dates.loc[outcome_dates['event']==False, 'durations'] = outcome_dates[death_date_field_id]-outcome_dates[assessment_date_field_id]
    # # lost to follow up
    outcome_dates.loc[outcome_dates['durations'].isna(), 'durations'] = outcome_dates[lost_to_follow_up_field_id]-outcome_dates[assessment_date_field_id]
    # # last visit date or last outcome date of non-cases

    # Dan Coster's suggestion:
    # take the maximum of any followup time for each individual as the censoring time
    # remove subjects without any followup

    #TODO: Should it just be second assessment? or also the imaging vists? It does look like they
    #      have questionnaires during the imaging visits.
    second_assessment_dates = get_ukbb_date_fields([assessment_date_field_id],instances=[1,2,3]) # second assessments


    all_outcome_dates = pd.read_csv(OUTCOMES_DATES)#,index_col=0)
    all_outcome_dates = all_outcome_dates[all_outcome_dates["Description"].str.startswith("Date")]
    dates_field_ids = all_outcome_dates['Field ID'].to_list()
    all_outcome_dates = get_ukbb_date_fields(dates_field_ids)
    all_outcome_dates[assessment_date_field_id] = second_assessment_dates
    max_outcome_dates = all_outcome_dates.max(axis=1)
    outcome_dates.loc[outcome_dates['durations'].isna(), 'durations'] = max_outcome_dates - \
                                                                        outcome_dates[assessment_date_field_id]

    # drop subjects if: - outcome was before first assessment
    #                   - no followup (i.e. no censoring date)
    outcome_dates = outcome_dates[outcome_dates['durations'] > pd.Timedelta(0)]
    outcome_dates = outcome_dates[~outcome_dates['durations'].isna()]
    outcome_dates['durations'] = outcome_dates['durations'].apply(lambda x: x.days)
    outcome_dates = outcome_dates[['event','durations']]

    if use_cache:
        if not os.path.isdir(CACHEPATH):
            os.makedirs(CACHEPATH)
        outcome_dates.to_csv(cache_file)
    logger.info("Loaded cohort")
    return outcome_dates



def plot_joint_features(tenk_data,ukbb_data):
    for col in tenk_data.columns:
        plot_joint_feature_distr(tenk_data[col],ukbb_data[col])

def plot_joint_feature_distr(tenk_feature,ukbb_feature):
    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot()
    minx = min(min(ukbb_feature.dropna()), min(tenk_feature.dropna()))
    maxx = max(max(ukbb_feature.dropna()), max(tenk_feature.dropna()))
    ukbb_hist, ukbb_bins = np.histogram(ukbb_feature.dropna(), range=(minx, maxx))
    ukbb_hist = ukbb_hist / np.sum(ukbb_hist)
    tenk_hist, tenk_bins = np.histogram(tenk_feature.dropna(), range=(minx, maxx))
    tenk_hist = tenk_hist / np.sum(tenk_hist)
    ax.bar(ukbb_bins[:-1], ukbb_hist, alpha=0.5, label="UKB ({})".format(len(ukbb_feature.dropna())),
           width=np.diff(ukbb_bins), align='edge')
    ax.bar(tenk_bins[:-1], tenk_hist, alpha=0.5, label="10K ({})".format(len(tenk_feature.dropna())),
           width=np.diff(tenk_bins), align='edge')
    val = tenk_feature.name
    ax.set_title(val)
    ax.legend()
    # u, p = mannwhitneyu(ukbb_feature.dropna(), tenk_feature.dropna())
    # ax.text(0.65, 0.75, "mwu pval: {:.3f}".format(p), transform=ax.transAxes)
    plt.savefig(os.path.join(OUTPATH,f'{val.replace("/","_")}.png'))
    plt.clf()


def join_ukbb_data(cohort,features):
    df = pd.concat([cohort,features],axis=1,join='inner')
    return df

def run_impute(dirpath):
    data_array = np.genfromtxt(os.path.join(dirpath,'to_impute.csv'),delimiter=",")
    imputer = SimpleImputer(strategy='median')
    data_array = imputer.fit_transform(data_array)
    np.savetxt("imputed.csv",data_array,delimiter=",")

def run_imputation(data,field,use_cache=True,preds=False):
    if use_cache:
        cache_file = os.path.join(CACHEPATH,"imputed_{}.csv".format(field))
        if preds: cache_file = os.path.join(CACHEPATH,"tenk_imputed_{}.csv".format(field))
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file,index_col=0)

    old_cwd = os.getcwd()
    os.chdir(OUTPATH)
    logger.info("Starting imputation")
    # save the data as a numpy array to disk so that it can be loaded by qp job
    dropFilter = data.filter(['durations', 'event'])
    feature_data = data.drop(dropFilter, axis=1)
    feature_data.to_csv(os.path.join(OUTPATH, 'to_impute.csv'), header=None, index=None)
    with qp(jobname='imputation', _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=1, max_u=100,
            _mem_def='25G') as q:
        q.startpermanentrun()
        qmethod = q.method(run_impute, (OUTPATH,))
        q.waitforresults(qmethod)
    logger.info("Done imputation")
    os.chdir(old_cwd)
    feature_data = pd.read_csv(os.path.join(OUTPATH, 'imputed.csv'), names=feature_data.columns)
    os.remove(os.path.join(OUTPATH, 'imputed.csv'))
    os.remove(os.path.join(OUTPATH, 'to_impute.csv'))
    feature_data = feature_data.set_index(data.index)
    data.update(feature_data)

    if use_cache:
        if not os.path.isdir(CACHEPATH):
            os.makedirs(CACHEPATH)
        data.to_csv(cache_file)

    return data


def run_cox(data,field,folds=3,use_cache=True):

    if use_cache:
        cache_file = os.path.join(OUTPATH, "models", "cox_model_{}_80.pkl".format(field))
        with open(cache_file, 'rb') as f:
            model = pickle.load(f)
        return model


    train_data, test_data = train_test_split(data,test_size=0.2)

    cox = CoxPHFitter()
    logger.info("Starting fit")
    #data.drop('falling_asleep_during_daytime (All of the time)', axis=1,inplace=True)
    cox.fit(train_data,'durations','event')
    logger.info("Done fit")

    train_concordance = cox.score(train_data, scoring_method="concordance_index")
    test_concordance = cox.score(test_data, scoring_method="concordance_index")
    train_ll = cox.score(train_data)
    test_ll = cox.score(test_data)

    logger.info("Train ll: {}\nTrain concordance: {}\nTest ll: {}\nTest concordance: {}".format(train_ll,train_concordance,test_ll,test_concordance))

  #  cox.check_assumptions(train_data,show_plots=True)
   # cox.print_summary()

    savepath = os.path.join(OUTPATH,"models","cox_model_{}_80.pkl".format(field))
    with open(savepath,'wb') as f:
        pickle.dump(cox,f)

def predict_on_tenk(field,use_cache=True):
    model_path = os.path.join(OUTPATH,"models","cox_model_{}_80.pkl".format(field))
    with open(model_path,'rb') as f:
        model = pickle.load(f)
    tenk_path = os.path.join(CACHEPATH, 'tenk_processed_features.csv')
    tenk_data = pd.read_csv(tenk_path,index_col=0)
    tenk_data = run_imputation(tenk_data,field,preds=True)
    predictions = model.predict_survival_function(tenk_data,[3650]).transpose().rename_axis('RegistrationCode').rename(columns={3650:'preds'})
    if use_cache:
        mkdirifnotexists(CACHEPATH)
        predictions.to_csv(os.path.join(CACHEPATH,"cox_preds_{}".format(field)))
    return predictions

def get_corrs(preds):
    dexa_data = add_age_gender_to_df(DEXALoader().get_data(study_ids=['10K'], research_stage=['baseline'],
                                                     groupby_reg='first', min_col_present_frac=0.5)).df
    dexa_data = dexa_data.reset_index(level=[1]).drop(['Date'], axis=1)
    data = preds.merge(dexa_data, left_index=True, right_index=True, how='left')
    regression_ret= pd.DataFrame(index=dexa_data.columns,columns=['regression_pval','regression_coeff'])
    partial_ret = pd.DataFrame(index=dexa_data.columns,columns=['partial_pearson','pearson_pval','partial_spearman','spearman_pval'])
    for col in dexa_data.columns:
        if col=='age' or col=='gender': continue
        x = data[['age','gender',col]]
        x=sm.add_constant(x)
        model = sm.OLS(data['preds'],x,missing='drop').fit()
        pvals = model.pvalues*len(dexa_data.columns)
        coeffs = model.params
        if pvals[col] < 0.05:
            regression_ret.loc[col]['regression_pval'] = pvals[col]
            regression_ret.loc[col]['regression_coeff'] = coeffs[col]
        partial_pearson = pg.partial_corr(data,col,'preds',covar=['age','gender'])
        if partial_pearson['p-val'].values[0]*len(dexa_data.columns) < 0.05:
            partial_ret.loc[col]['partial_pearson'] = partial_pearson['r'].values[0]
            partial_ret.loc[col]['pearson_pval'] = partial_pearson['p-val'].values[0]
        partial_spearman = pg.partial_corr(data,col,'preds',covar=['age','gender'],method="spearman")
        if partial_spearman['p-val'].values[0]*len(dexa_data.columns) < 0.05:
            partial_ret.loc[col]['partial_spearman'] = partial_spearman['r'].values[0]
            partial_ret.loc[col]['spearman_pval'] = partial_spearman['p-val'].values[0]
    return regression_ret.dropna(how='all'), partial_ret.dropna(how='all')


def main():
    #tenk_data,ukbb_data = load_joint_features(use_cache=False)
    # plot_joint_features(tenk_data,ukbb_data)
    ## outcome_field_ids = get_outcomes()
    outcome_field_ids = [131306] #
    # #TODO: Competing risks
    for field in outcome_field_ids:
    #     ukb_cohort = get_ukbb_cohort(field)
    #     ukbb_data = join_ukbb_data(ukb_cohort,ukbb_data)
    #     imputed_data = run_imputation(ukbb_data,field)
    #     run_cox(imputed_data,field,folds=1)
    #     # get predictions on tenk
         preds = predict_on_tenk(field)
         model_pred_corrs_regression, model_pred_corrs_partial = get_corrs(preds)
    #     model_pred_corrs_regression.info()
    #     model_pred_corrs_partial.info()
    #     mkdirifnotexists(CACHEPATH)
    #     model_pred_corrs_regression.to_csv(os.path.join(CACHEPATH, "cox_pred_regression_{}".format(field)))
    #     model_pred_corrs_partial.to_csv(os.path.join(CACHEPATH, "cox_pred_partial_corr_{}".format(field)))



if __name__ == "__main__":
    logfile = os.path.join(OUTPATH, 'test_cox_run.log')
    logging.basicConfig(filemode='w',filename=logfile,level=logging.INFO, format='%(asctime)s: %(message)s',datefmt='%d/%m/%Y %H:%M')
    logger = logging.getLogger("logger")
    main()



