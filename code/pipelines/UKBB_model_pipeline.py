import os
from sys import exit

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import json

from LabUtils.Utils import mkdirifnotexists

from LabData.DataAnalyses.UKBB_10k.UKBB_survival import *
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, GridSearchCV,  ShuffleSplit, cross_validate
from sklearn.metrics import make_scorer

from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.ensemble import RandomForestRegressor


from lifelines.utils.sklearn_adapter import sklearn_adapter
from lifelines.utils import concordance_index

from sksurv.ensemble import RandomSurvivalForest
from sksurv.datasets import get_x_y
from sksurv.metrics import concordance_index_censored as sksurv_concordance

#import xgboost as xgb
import optuna

import matplotlib

import pickle, glob

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import joblib

#JOINT_FEATURES_MAPPING = '/net/mraid08/export/jafar/UKBioBank/davidpel/UKBB_data/joint_features.csv'


from sklearn.base import TransformerMixin, BaseEstimator,RegressorMixin
class Debugger(BaseEstimator, TransformerMixin):
    def transform(self, data):
        print(data.shape)
        print(data.head())
        # data_corr = data.corr()
        # data_corr.to_csv(os.path.join(CACHEPATH,'corrs.csv'))
        return data

    def fit(self, data, y=None, **fit_params):
        return self

class XGB_in(BaseEstimator, TransformerMixin):

    def transform(self, data, y=None):
        y_lb = data['durations'].to_numpy(dtype=float)
        y_ub = data['durations'].to_numpy(dtype=float)
        y_ub[~data['event'].to_numpy(dtype=np.bool)] = np.inf
        dmat = xgb.DMatrix(np.float32(data.drop(['event','durations'],axis=1)))
        dmat.set_float_info('label_lower_bound',y_lb)
        dmat.set_float_info('label_upper_bound', y_ub)
        return dmat

    def fit(self, data, y=None, **fit_params):
        return self

class XGBWrap(BaseEstimator,RegressorMixin):
    def __init__(self,learning_rate=0.001, aft_loss_distribution = 'normal', aft_loss_distribution_scale = 0.01,
                 max_depth = 5,min_child_weight = 1000,colsample_bytree = 0.5, reg_alpha = 0.0001, reg_lambda = 0.0001,num_boost_rounds = 10000):#,
        self.learning_rate = learning_rate
        self.aft_loss_distribution = aft_loss_distribution
        self.aft_loss_distribution_scale = aft_loss_distribution_scale
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.preds = []
        self.true = []
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.num_boost_rounds = num_boost_rounds
        self.xgb_model = None


    def fit(self,X,y=None):
        dmat = xgb.DMatrix(X.drop(['durations','event'],axis=1))
        dmat.set_float_info('label', y)
    #    dmat.set_float_info('label_lower_bound',y['label_lower_bound'])
    #    dmat.set_float_info('label_upper_bound', y['label_upper_bound'])
        self.xgb_model=xgb.train({#{'learning_rate':self.learning_rate,
     #                               'aft_loss_distribution':self.aft_loss_distribution,
     #                               'aft_loss_distribution_scale':self.aft_loss_distribution_scale,
     #                                'max_depth' : self.max_depth,
     #                                'colsample_bytree':self.colsample_bytree,
     #                                'min_child_weight':self.min_child_weight,
     #                                'reg_lambda':self.reg_lambda,
     #                                'reg_alpha' : self.reg_alpha,
     #                                'verbosity':1,
                                     'objective':'survival:cox', #'survival:aft', #
                                      'base_score':1,
     #                                'eval_metric':'cox-nloglik', #'aft-nloglik', #
     #                                'tree_method':'hist',
     #                                'subsample' : 0.7,
     #                                'nthread' : 16
                                  },
                                 dmat,
                                 num_boost_round=100)#self.num_boost_rounds)#, ###########
                                 #early_stopping_rounds=50)
        #self.xgb_model.train(dmat)
        return self

    def predict(self,X):
        dmat = xgb.DMatrix(X.drop(['durations','event'],axis=1))
        return self.xgb_model.predict(dmat)


def lifelines_scorer(clf,X,y):
    y_pred = clf.predict(X)
    return concordance_index(y,y_pred,X['event'])

def lifelines_sksurv_scorer(clf,X,y):
    y_pred = clf.predict(X)
    return sksurv_concordance(X['event'],y,-y_pred)[0]

def xgb_sksurv_scorer(clf,X,y):
    y_pred = clf.predict(X)
    return sksurv_concordance(y<0,abs(y),-y_pred)[0]
    #return sksurv_concordance(y['label_upper_bound']==np.inf,y['label_lower_bound'],-y_pred)[0]

def rsf_scorer(clf,X,y):
    y_pred = clf.predict(X)
    return concordance_index(y['durations'],-y_pred,y['event'])

def xgb_scorer(clf,X,y):
    y_pred = clf.predict(X)
    return concordance_index(abs(y),y_pred,y<0)
    #return concordance_index(y['label_lower_bound'],y_pred,y['label_upper_bound']==np.inf)



class UKBB_model():
    def __init__(self, model, preprocessing=None, imputation=None, feature_fields=None, outcome_field=131306, use_cache = True,train=False,tenk=False, model_kwargs={}):
        self.model = model
        self.model_kwargs = model_kwargs
        self.preprocessing = preprocessing
        self.imputation = imputation
        self.feature_fields = feature_fields
        self.outcome_field = outcome_field
        self.use_cache = use_cache
        self.train=train
        self.tenk=tenk
        self._paths_setup()
        self.logger = self._create_logger()
        self.features_table = pd.read_csv(FEATURE_MAPPING).astype({"UKBB Field ID": 'int'})
        self.features_table.dropna(how='any',inplace=True)
        if self.feature_fields is None:
            self.feature_fields = self.features_table["UKBB Field ID"].to_list()
        else:
            self.features_table = self.features_table.loc[self.features_table["UKBB Field ID"].isin(self.feature_fields)]

        # dataframes that will be assigned in the pipeline
        self.cohort = pd.DataFrame()
        self.train_features = pd.DataFrame()
        self.train_data = pd.DataFrame()

        self.simple_impute=None
        self.iterative_zero_impute=None
        self.iterative_mode_impute=None
        self.standardizer=None
        self.quantilizer=None
        self.pipe=None



    def _paths_setup(self):
        if self.use_cache:
            mkdirifnotexists(CACHEPATH)
            tenk_path = "tenk_features"
            ukbb_path = "ukbb_features"
            if self.feature_fields is not None:
                tenk_path += "_"+"_".join([str(f) for f in self.feature_fields])
                ukbb_path += "_" + "_".join([str(f) for f in self.feature_fields])
            if self.model == 'coxph':
                tenk_path+= "_drop_1hot_col"
                ukbb_path += "_drop_1hot_col"
            self.tenk_features_path = os.path.join(CACHEPATH, tenk_path + ".csv")
            self.ukbb_features_path = os.path.join(CACHEPATH, ukbb_path + ".csv")

            if self.imputation is not None:
                assert self.imputation in ["iter_zero", "iter_mode","simple"]
                self.ukbb_features_path = os.path.join(CACHEPATH, self.imputation + "_imputed_" + ukbb_path + ".csv")

            self.cohort_path = os.path.join(CACHEPATH,"cohort_{}.csv".format(self.outcome_field))
        self.competing_path = COMPETING_PATH
        self.outcome_group_path = OUTCOME_GROUPS
        self.outpath = os.path.join(OUTPATH,self.model,str(self.outcome_field))
        if self.feature_fields is not None:
            self.outpath = os.path.join(self.outpath,"_".join([str(s) for s in self.feature_fields]))
        if self.train:
            self.outpath = os.path.join(self.outpath, 'full_data')
        if self.tenk:
            self.outpath = os.path.join(self.outpath, 'tenk')
        if os.path.isfile(os.path.join(self.outpath, "{}_{}_optunasearch.csv".format(self.outcome_field, self.model))):
            exit("Output file exists for {}".format(self.outcome_field))
        mkdirifnotexists(self.outpath)
       # self.results_path = os.path.join(self.outpath,'grid_search.csv')

    def _create_logger(self):
        logfile_name = "_".join(map(str,[x for x in [self.model,self.outcome_field] if x is not None]))+".log"
        mkdirifnotexists(os.path.join(self.outpath,"logs"))
        logfile = os.path.join(self.outpath, "logs", logfile_name)
        logging.basicConfig(filemode='w', filename=logfile, level=logging.INFO, format='%(asctime)s: %(message)s',
                            datefmt='%d/%m/%Y %H:%M')
        logger = logging.getLogger("logger")
        return logger

    def get_numerical(self,data):
        numerical_columns = []
        showcase_dict = UkbbLoader().ukbb_data_dic_showcase
        for feature in self.features_table["10K Name"]:
            if len(data.filter(like=feature).columns) > 0:
                feature_row = self.features_table.loc[self.features_table["10K Name"] == feature]
                ukbb_field_id = feature_row["UKBB Field ID"].item()
                showcase_metadata = showcase_dict.loc[showcase_dict["FieldID"] == ukbb_field_id]
                datatype = showcase_metadata["ValueType"].item()
                if datatype == "Integer" or datatype == "Continuous" and feature in data.columns:
                    numerical_columns.append(feature)
        return numerical_columns

    def get_categorical(self,data):
        categorical_columns = []
        showcase_dict = UkbbLoader().ukbb_data_dic_showcase
        for feature in self.features_table["10K Name"]:
            if len(data.filter(like=feature).columns) > 0:
                feature_row = self.features_table.loc[self.features_table["10K Name"] == feature]
                ukbb_field_id = feature_row["UKBB Field ID"].item()
                showcase_metadata = showcase_dict.loc[showcase_dict["FieldID"] == ukbb_field_id]
                datatype = showcase_metadata["ValueType"].item()
                if "categorical" in datatype.lower():
                    categorical_columns+= data.filter(regex=feature+"\s\((.*)\)").columns.tolist() #.append(feature)
        return categorical_columns

    def get_cancer_cohort(self):
        assessment_date_field_id = 53
        death_date_field_id = 40000
        lost_to_follow_up_field_id = 191
        # for people with multiple occurences, take first
        # 40013 - ICD9, 40005 - date, 40006 - ICD10, 40009 - # occurrences
        df = UkbbLoader().load_ukbb(which=[40005, 40006], instances=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                    rename_cols=False)  # very few people have more than 10 occurrences of cancer
        df = df.replace('nan', np.NaN)  # NaNs are encoded as strings ??
        df = df.dropna(how='all')

        mask = df.apply(lambda x: x.astype(str).str.startswith(self.outcome_field)).any(axis=1)
        df = df[mask]
        type_df = df.filter(like='40006')
        type_df = type_df[sorted(type_df.columns.tolist())]
        date_df = df.filter(like='40005')
        date_df = date_df[sorted(date_df.columns.tolist())]

        def foo(iid, icd, df):
            p = df.loc[iid].str.find(icd)
            return int(p.loc[p == 0,].index.str.split("visit ")[0][-1])

        iids = type_df.index.to_list()
        visit_list = list(map(lambda iid: foo(iid, icd=self.outcome_field, df=type_df), iids))
        visit_dict = dict(zip(iids, visit_list))
        cancer_dates = pd.Series(
            dict(zip(iids, list(map(lambda iid: date_df.loc[iid].iloc[visit_dict.get(iid)], iids)))))
        outcome_dates = get_ukbb_date_fields(
            [assessment_date_field_id, death_date_field_id, lost_to_follow_up_field_id])
        outcome_dates[self.outcome_field] = pd.to_datetime(cancer_dates)

        outcome_dates['durations'] = outcome_dates[self.outcome_field] - outcome_dates[assessment_date_field_id]

        outcome_dates['event'] = outcome_dates['durations'] > pd.Timedelta(0)
        return outcome_dates

    def load_cohort(self):
        if self.use_cache and os.path.isfile(self.cohort_path):
            logger.info("Loaded cohort from cache")
            self.cohort = pd.read_csv(self.cohort_path,index_col=0)
            return self.cohort

        # determine which participants had the outcome *after* registration
        # and the time to event

        logger.info("Getting cohort for field {}".format(self.outcome_field))
        assessment_date_field_id = 53
        death_date_field_id = 40000
        lost_to_follow_up_field_id = 191

        if type(self.outcome_field) is str: # cancer
            outcome_dates = self.get_cancer_cohort()

        else:

            # grouped outcomes
            with open(self.outcome_group_path) as f:
                group_dict = json.load(f)
            group_list = group_dict.get(str(self.outcome_field),[])
            outcome_fields = group_list+[self.outcome_field]

            outcome_dates = get_ukbb_date_fields(outcome_fields + [assessment_date_field_id,
                                                  death_date_field_id, lost_to_follow_up_field_id])
            outcome_dates['durations'] = outcome_dates[outcome_fields].min(axis=1) - outcome_dates[assessment_date_field_id]

            outcome_dates['event'] = outcome_dates['durations'] > pd.Timedelta(0)

            # get all censoring durations
            # competing outcomes
            with open(self.competing_path) as f:
                competing_dict = json.load(f)
            competing_list = []
            for field in outcome_fields:
                competing_list += competing_dict.get(str(self.outcome_field),[])
            if len(competing_list)>0:
                competing_dates = get_ukbb_date_fields(competing_list)
                min_competing = competing_dates.min(axis=1)
                outcome_dates['durations'] = min_competing-outcome_dates[assessment_date_field_id]

        # # deaths of non-cases
        outcome_dates.loc[(outcome_dates['durations'].isna()) & (outcome_dates['event'] == False), 'durations'] = \
            outcome_dates[death_date_field_id] - outcome_dates[assessment_date_field_id]
        # # lost to follow up
        outcome_dates.loc[outcome_dates['durations'].isna(), 'durations'] = outcome_dates[lost_to_follow_up_field_id] - \
                                                                            outcome_dates[assessment_date_field_id]
        # Dan Coster's suggestion:
        # take the maximum of any followup time for each individual as the censoring time
        # remove subjects without any followup

        second_assessment_dates = get_ukbb_date_fields([assessment_date_field_id],
                                                       instances=[1, 2, 3])  # second assessments

        all_outcomes_dates = pd.read_csv(OUTCOMES_DATES)
        all_outcomes_dates = all_outcomes_dates[all_outcomes_dates["Description"].str.startswith("Date")]
        dates_field_ids = all_outcomes_dates['Field ID'].to_list()
        all_outcomes_dates = get_ukbb_date_fields(dates_field_ids)
        all_outcomes_dates[assessment_date_field_id] = second_assessment_dates
        max_outcome_dates = all_outcomes_dates.max(axis=1)
        outcome_dates.loc[outcome_dates['durations'].isna(), 'durations'] = max_outcome_dates - \
                                                                            outcome_dates[assessment_date_field_id]

        # drop subjects if: - outcome was before first assessment
        #                   - no followup (i.e. no censoring date)
        outcome_dates = outcome_dates[outcome_dates['durations'] > pd.Timedelta(0)]
        outcome_dates = outcome_dates[~outcome_dates['durations'].isna()]
        outcome_dates['durations'] = outcome_dates['durations'].apply(lambda x: x.days)
        self.cohort = outcome_dates[['event', 'durations']]

        if self.use_cache:
            self.cohort.to_csv(self.cohort_path)
        logger.info("Loaded cohort")
        return self.cohort


    def _plot_joint_features(self):
        ukbb_data = load_prep_ukbb_mapped_features(self.features_table)
        tenk_data = load_prep_10k_mapped_features(self.features_table)

        ukbb_data = rename_ukbb_fields_to_match_tenks_names(ukbb_data, tenk_data, self.features_table)

        tenk_data, ukbb_data = process_joint_features(self.features_table, tenk_data, ukbb_data)
        plot_joint_features(tenk_data, ukbb_data)

    def load_raw_train_features(self):
        # load data
        # try to get from cache first
        if self.use_cache and os.path.isfile(self.ukbb_features_path):
            ukbb_data = pd.read_csv(self.ukbb_features_path, index_col=0)
            logger.info("Loaded joint features from cache")
        else:
            # get the ukbb data - joint features, preprocessing etc..
            logger.info("Getting joint UKB, 10K features")
            ukbb_data = load_prep_ukbb_mapped_features(self.features_table)
            tenk_data = load_prep_10k_mapped_features(self.features_table)

            ukbb_data = rename_ukbb_fields_to_match_tenks_names(ukbb_data, tenk_data, self.features_table)

            tenk_data, ukbb_data = process_joint_features(self.features_table, tenk_data, ukbb_data,
                                                          self.model != 'coxph')

            if self.use_cache:
                logger.info("Saving features")
                ukbb_data.to_csv(self.ukbb_features_path)
                if not os.path.isfile(self.tenk_features_path):
                    tenk_data.to_csv(self.tenk_features_path)

        self.train_features = ukbb_data
        return self.train_features


    def load_train_data(self):
        self.train_data = pd.concat([self.load_cohort(), self.load_raw_train_features()], axis=1, join='inner')
        return self.train_data

    def define_training_pipeline_elems(self):

        numerical_cols = self.get_numerical(self.train_features)
        categorical_cols = self.get_categorical(self.train_features)

        self.simple_impute = ColumnTransformer(
            [('num_imp',SimpleImputer(strategy='median'),numerical_cols),
            ('cat_imp', SimpleImputer(strategy='most_frequent'),categorical_cols)],
            remainder='passthrough',verbose=True,verbose_feature_names_out=False
         #   n_jobs=16
            )#,

        self.simple_impute.set_output(transform='pandas')

        self.iterative_zero_impute = ColumnTransformer(
            [('num_imp', IterativeImputer(n_nearest_features=15,verbose=2),numerical_cols),
            ('cat_imp', SimpleImputer(strategy='constant',fill_value=0,add_indicator=True),categorical_cols)],
            remainder='passthrough',verbose_feature_names_out=False,verbose=True,
            n_jobs=2)
        self.iterative_zero_impute.set_output(transform='pandas')

        self.iterative_mode_impute = ColumnTransformer(
            [('num_imp', IterativeImputer(n_nearest_features=15,verbose=2),numerical_cols),
            ('cat_imp', SimpleImputer(strategy='most_frequent'),categorical_cols)],
            remainder='passthrough',verbose_feature_names_out=False,verbose=True,
            n_jobs=2)
        self.iterative_mode_impute.set_output(transform='pandas')

        self.standardizer = ColumnTransformer(
            [('num_norm', StandardScaler(),numerical_cols)],
            remainder='passthrough',verbose=True,verbose_feature_names_out=False,
            n_jobs=2
        )
        self.standardizer.set_output(transform='pandas')

        self.quantilizer = ColumnTransformer(
            [('num_norm', QuantileTransformer(),numerical_cols)],
            remainder='passthrough',verbose=True,verbose_feature_names_out=False,
            n_jobs=2
        )
        self.quantilizer.set_output(transform='pandas')

        CoxClass = sklearn_adapter(CoxPHFitter, event_col="event")
        self.pipe = Pipeline([('imp', "pasthrough"),
                        #      ('debugger2',Debugger()),
                              ('prep',"passthrough"),
                              ('model',CoxClass() )]) #CoxPHFitter(duration_col='durations',event_col='event'))])
      #  self.pipe.set_output(transform='pandas')
        return self.pipe

    def get_optuna_grid_for_model(self,model):
        if model == 'coxph' or model == 'regularized_cox':

            param_grid = {
                #
                #     'imp' : optuna.distributions.CategoricalDistribution([self.simple_impute]),
                # #     'prep' : optuna.distributions.CategoricalDistribution(["passthrough",self.standardizer, self.quantilizer]),
                # #     'model__penalizer' : optuna.distributions.FloatDistribution(0.000001,10,log=True),
                # #     'model__l1_ratio' : optuna.distributions.FloatDistribution(0.0,1.0),
                # #
                #
                #     # 'imp' : optuna.distributions.CategoricalDistribution([self.iterative_zero_impute, self.iterative_mode_impute]),
                #     # 'imp__num_imp__estimator' : optuna.distributions.CategoricalDistribution([RandomForestRegressor(n_estimators=50,max_depth=10,max_samples=0.5)]),#BayesianRidge(verbose=True),
                #     # 'imp__num_imp__tol' : optuna.distributions.FloatDistribution(1e-3,1e-1),
                #     'prep': optuna.distributions.CategoricalDistribution(["passthrough", self.standardizer, self.quantilizer]),
                #     'model__penalizer': optuna.distributions.FloatDistribution(0.000001,10),
                #     'model__l1_ratio' : optuna.distributions.FloatDistribution(0.0,1.0)
            }
            Y = self.train_data['durations']#.pop('durations')
            X = self.train_data.drop('durations',axis=1)
            scorer = lifelines_scorer

        if model == 'rsf':
            param_grid = {
                #
                    'imp' : optuna.distributions.CategoricalDistribution([self.simple_impute]),
                    'model': optuna.distributions.CategoricalDistribution([RandomSurvivalForest(verbose=4, max_samples=0.1)]),
                    'model__n_estimators' : optuna.distributions.IntDistribution(1,5),
                    'prep': optuna.distributions.CategoricalDistribution(["passthrough", self.standardizer, self.quantilizer]),

            }
            X, Y = get_x_y(self.train_data, attr_labels=('event', 'durations'), pos_label=True)
            scorer = rsf_scorer

        if model == 'xgboost':
            param_grid = {
                        #    'imp' : optuna.distributions.CategoricalDistribution(["passthrough",self.simple_impute]),#,]
                            'imp': optuna.distributions.CategoricalDistribution([self.iterative_zero_impute, self.iterative_mode_impute]),  # ,],
                            'imp__num_imp__estimator': optuna.distributions.CategoricalDistribution([BayesianRidge(verbose=True),
                                                     RandomForestRegressor(n_estimators=50, max_depth=10,
                                                                           max_samples=0.5)]),
                            'imp__num_imp__tol': optuna.distributions.FloatDistribution(0.0001,0.1,log=True),
                            'prep' : optuna.distributions.CategoricalDistribution([self.standardizer,"passthrough", self.quantilizer]),
                            'model': optuna.distributions.CategoricalDistribution([XGBWrap()]),#xgb_model],
                            'model__num_boost_rounds' : optuna.distributions.IntDistribution(100,100000, log=True),
                            'model__learning_rate': optuna.distributions.FloatDistribution(0.000001,0.1,log=True),
                            'model__aft_loss_distribution': optuna.distributions.CategoricalDistribution(['normal', 'logistic', 'extreme']),
                            'model__aft_loss_distribution_scale': optuna.distributions.FloatDistribution(0.5, 1.5),
                            'model__max_depth': optuna.distributions.IntDistribution(3,5),
                            'model__min_child_weight' : optuna.distributions.IntDistribution(10,10000,log=True),
                            'model__colsample_bytree' : optuna.distributions.FloatDistribution(0.05,0.75,step = 0.05),
                            'model__reg_alpha': optuna.distributions.FloatDistribution(0.000001, 1,log=True),
                            'model__reg_lambda': optuna.distributions.FloatDistribution(0.000001, 1, log=True),
                          } #### TEST
            y_lb = self.train_data['durations'].to_numpy(dtype=float)
            y_ub = self.train_data['durations'].to_numpy(dtype=float)
            y_ub[~self.train_data['event'].to_numpy(dtype='bool')] = np.inf
            Y = pd.DataFrame({'label_lower_bound':y_lb,'label_upper_bound':y_ub})
###############################
            Y = self.train_data['durations'].to_numpy(dtype=float) #################
            Y[~self.train_data['event'].to_numpy(dtype='bool')] *= -1. #############
            X = self.train_data#.drop(['durations','event'],axis=1)
            scorer=xgb_scorer
        return param_grid, X, Y, scorer

    def get_grid_for_model(self,model):
        if model == 'coxph' or model == 'regularized_cox':

            param_grid = [
                {
                    'imp' : [self.simple_impute],
                    'prep' : ["passthrough",self.standardizer, self.quantilizer],
                    'model__penalizer' : [0.0001,0.001,0.01,0.1,1,10],
                    'model__l1_ratio' : [0.0,0.01,0.1,0.25,0.5,0.75,0.9,0.99,1.0]
                },
                {
                    'imp' : [self.iterative_zero_impute, self.iterative_mode_impute],
                    'imp__num_imp__estimator' : [BayesianRidge(verbose=True),RandomForestRegressor(n_estimators=100,max_depth=10,max_samples=0.5)],
                    'imp__num_imp__tol' : [1e-3,1e-2,1e-1],
                    'prep': ["passthrough", self.standardizer, self.quantilizer],
                    'model__penalizer': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                    'model__l1_ratio': [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
                }
            ]
            Y = self.train_data['durations']#.pop('durations')
            X = self.train_data.drop('durations',axis=1)
            scorer = lifelines_scorer

        elif model == "rsf":

            param_grid = [{'imp' : [self.simple_impute],
                           'model' : [RandomSurvivalForest(verbose=4,max_samples=0.1)],
                           'model__n_estimators' : [100,1000]}] # TEST

            X, Y = get_x_y(self.train_data, attr_labels=('event', 'durations'), pos_label=True)
            scorer = rsf_scorer


        elif model == "xgboost":
            # TODO: figure out survival:cox objective
           # xgb_model = XGBWrap()#base_params)
            param_grid = [{
                            'imp' : ["passthrough"],#self.simple_impute],#,],
                            'prep' : [self.standardizer],#"passthrough",],# self.quantilizer],
                            'model': [XGBWrap()],#xgb_model],
                            'model__learning_rate': [0.0001],
                            'model__aft_loss_distribution': ['normal'],# 'logistic', 'extreme'],
                            'model__aft_loss_distribution_scale': [ 1, 1.5, 2],
                            'model__max_depth': [3],
                            'model__min_child_weight' : [1000],#1000,10000]
                            'model__colsample_bytree' : [0.05,0.5],
                            'model__reg_alpha': [ 0.0001, 0.01, 0.1, 1],
                          }] #### TEST

            y_lb = self.train_data['durations'].to_numpy(dtype=float)
            y_ub = self.train_data['durations'].to_numpy(dtype=float)
            y_ub[~self.train_data['event'].to_numpy(dtype='bool')] = np.inf
            Y = pd.DataFrame({'label_lower_bound':y_lb,'label_upper_bound':y_ub})
            X = self.train_data#.drop(['durations','event'],axis=1)
            scorer=xgb_scorer

        elif model == "sksurv_gbm":
            pass # TODO

        elif model == "sksurv_cgbm":
            pass # TODO

        elif model == "deepsurv":
            pass #TODO


        return param_grid,X,Y,scorer


    def cox_objective(self,trial):
        cox_model = sklearn_adapter(CoxPHFitter, event_col="event")

        prep = trial.suggest_categorical('prep', [self.standardizer, "passthrough", self.quantilizer])
        model=cox_model()
        ######################################################################
        model__penalizer = trial.suggest_int('model__penalizer', -3, 1)# -4, 1) #### NOTE - this is for ones that were singular
        model__l1_ratio = trial.suggest_float('model__l1_ratio', 0.0, 1.0)

        imp_method = trial.suggest_categorical("imp_method", ["simple", "iter"])

        if imp_method == "simple":
            simp_imp = trial.suggest_categorical("simp_imp", [self.simple_impute])
            self.pipe = Pipeline([('imp', simp_imp),
                                  ('prep', prep),
                                  ('model', model)])

        else:
            imp = trial.suggest_categorical("imp", [self.iterative_zero_impute, self.iterative_mode_impute])
            imp__num_imp__tol = trial.suggest_int('imp__num_imp__tol', -4, -2)#, log=True)
            self.pipe = Pipeline([('imp', imp),
                                  ('prep', prep),
                                  ('model', model)])
            self.pipe.set_params(**{"imp__num_imp__tol": 10 ** imp__num_imp__tol})
    #    self.pipe.set_output(transform='pandas')
        self.pipe.set_params(**{"model__penalizer": 10 ** model__penalizer,
                                "model__l1_ratio": model__l1_ratio
                                })

        cv = ShuffleSplit(n_splits=3, test_size=0.3)
        _, X, Y, scorer = self.get_optuna_grid_for_model(self.model)
        scores = cross_validate(self.pipe,X,Y,scoring={"lifelines" : scorer, "sksurv":lifelines_sksurv_scorer},cv=cv,n_jobs=1)
        logger.info("Lifelines: {}, sksurv: {}".format(scores['test_lifelines'].mean(), scores['test_sksurv'].mean()))
        return scores['test_lifelines'].mean()

    def rsf_objective(self,trial):

        prep = trial.suggest_categorical('prep', [self.standardizer, "passthrough", self.quantilizer])
        model=RandomSurvivalForest(verbose=2,max_samples=0.5)#,n_jobs=12)
        model__n_estimators = trial.suggest_int('model__n_estimators', 50,200, step=50)
        model__max_depth = trial.suggest_int('model__max_depth', 5,21,step=2)
        model__max_features = trial.suggest_categorical('model__max_features',[None, "log2", 0.5])

        imp_method = trial.suggest_categorical("imp_method", ["simple", "iter"])
        if imp_method == "simple":
            simp_imp = trial.suggest_categorical("simp_imp", [self.simple_impute])
            self.pipe = Pipeline([('imp', simp_imp),
                                  ('prep', prep),
                                  ('model', model)])
        else:
            imp = trial.suggest_categorical("imp", [self.iterative_zero_impute, self.iterative_mode_impute])
            imp__num_imp__tol = trial.suggest_int('imp__num_imp__tol', -4, -2)#, log=True)
            self.pipe = Pipeline([('imp', imp),
                                  ('prep', prep),
                                  ('model', model)])
            self.pipe.set_params(**{"imp__num_imp__tol": 10 ** imp__num_imp__tol})
     #   self.pipe.set_output(transform='pandas')
        self.pipe.set_params(**{"model__n_estimators": model__n_estimators,
                                "model__max_depth" : model__max_depth,
                                "model__max_features" : model__max_features
                                })

        cv = ShuffleSplit(n_splits=3, test_size=0.3)
        _, X, Y, scorer = self.get_optuna_grid_for_model(self.model)
        scores = cross_validate(self.pipe,X,Y,scoring={"pipe_score" : None, "lifelines_cind" : scorer},cv=cv)
        logger.info("Lifelines: {}, Self: {}".format(scores['test_lifelines_cind'].mean(),scores['test_pipe_score'].mean()))
        return scores['test_lifelines_cind'].mean()


    def xgb_objective(self,trial):
        xgb_model = XGBWrap()
        prep = trial.suggest_categorical('prep', [self.standardizer, "passthrough", self.quantilizer])
        model =  xgb_model # trial.suggest_categorical("model",[XGBWrap()]),  #
##        model__num_boost_rounds = trial.suggest_int("model__num_boost_rounds",2,4)
##        model__learning_rate =trial.suggest_int('model__learning_rate', -5,-1)
    #    model__aft_loss_distribution = trial.suggest_categorical('model__aft_loss_distribution',
    #                                                             ['normal', 'logistic', 'extreme'])
    #    model__aft_loss_distribution_scale = trial.suggest_float('model__aft_loss_distribution_scale', 0.5, 1.5,step=0.1)
##        model__max_depth = trial.suggest_int('model__max_depth',3, 5)
##        model__min_child_weight = trial.suggest_int('model__min_child_weight', 1, 4)
##        model__colsample_bytree = trial.suggest_float('model__colsample_bytree',0.2, 0.7, step=0.1)
##        model__reg_alpha = trial.suggest_int('model__reg_alpha',-6,0)
##        model__reg_lambda = trial.suggest_int('model__reg_lambda',-6,0)

        imp_method = trial.suggest_categorical("imp_method", ["simple", "iter"])

        if imp_method == "simple":
            simp_imp = trial.suggest_categorical("simp_imp", [self.simple_impute])
            self.pipe = Pipeline([('imp', simp_imp),
                                  ('prep', prep),
                                  ('model', model)])
        else:
            imp = trial.suggest_categorical("imp", [self.iterative_zero_impute, self.iterative_mode_impute])
            imp__num_imp__tol = trial.suggest_int('imp__num_imp__tol', -4, -2)  # , log=True)
            self.pipe = Pipeline([('imp', imp),
                                  ('prep', prep),
                                  ('model', model)])
            self.pipe.set_params(**{"imp__num_imp__tol": 10 ** imp__num_imp__tol})

     #   imp = trial.suggest_categorical("imp", ["passthrough", self.simple_impute])
     #
     #    self.pipe = Pipeline([('imp', imp),
     #                          ('prep', prep),
     #                          ('model', model)])

        # self.pipe.set_params(**{"model__num_boost_rounds": 10 ** model__num_boost_rounds,
        #                         "model__learning_rate": 10 ** model__learning_rate,
        #                    #     "model__aft_loss_distribution": model__aft_loss_distribution,
        #                    #     "model__aft_loss_distribution_scale": model__aft_loss_distribution_scale,
        #                         "model__max_depth": model__max_depth,
        #                         "model__min_child_weight": 10 ** model__min_child_weight,
        #                         "model__colsample_bytree": model__colsample_bytree,
        #                         "model__reg_alpha": 10 ** model__reg_alpha,
        #                         "model__reg_lambda": 10 ** model__reg_lambda
        #                         })

     #   self.pipe.set_output(transform='pandas')
        cv = ShuffleSplit(n_splits=3, test_size=0.3)
        _, X, Y, scorer = self.get_optuna_grid_for_model(self.model)
        scores = cross_validate(self.pipe,X,Y,scoring={"lifelines":scorer,"sksurv":xgb_sksurv_scorer,"self":None},cv=cv)
        logger.info("Lifelines: {}, sksurv: {}, self: {}".format(scores['test_lifelines'].mean(), scores['test_sksurv'].mean(), scores['test_self'].mean()))
        return scores['test_lifelines'].mean()


    def run_optuna_study(self):
        study = optuna.create_study(direction="maximize")
        if self.model == "xgboost":
            study.optimize(self.xgb_objective, n_trials=150)
        elif self.model=='cox' or self.model == 'regularized_cox':
            #############################################################
            study.optimize(self.cox_objective, n_trials=120, n_jobs=1)#150,n_jobs=1) ###NOTE: for singular, removed one option
        elif self.model=='rsf':
            study.optimize(self.rsf_objective, n_trials=150)
        results = pd.DataFrame(study.trials_dataframe())
        results_path = os.path.join(self.outpath, "{}_{}_optunasearch.csv".format(self.outcome_field, self.model))
        if os.path.isfile(results_path):
            prev_res = pd.read_csv(results_path)
            results = pd.concat([prev_res,results])
        results.to_csv(results_path, index=False)

    # def run_optuna_grid(self):
    #     param_grid, X, Y, scorer = self.get_optuna_grid_for_model(self.model)
    #
    #     logger.info("Running the parameter grid")
    #
    #     # cv = RepeatedKFold(n_splits=3,n_repeats=3)
    #     cv = ShuffleSplit(n_splits=3, test_size=0.3)
    #
    #     search = optuna.integration.OptunaSearchCV(self.pipe, param_distributions=param_grid, cv=cv, verbose=4, scoring= scorer,
    #                         refit=False, error_score="raise", n_trials=200)#, n_jobs = 12
    #                        # )  # ,n_jobs=4,pre_dispatch=8)#,scoring=[make_scorer(ll),make_scorer(concordance)])#['concordance_index','log_likelihood',])#
    #     #        grid = GridSearchCV(self.pipe,param_grid=param_grid,cv=cv,verbose=4,scoring={"pipe_score" : None, "lifelines_cind" : scorer},refit=False,error_score='raise')#,n_jobs=4,pre_dispatch=8)#,scoring=[make_scorer(ll),make_scorer(concordance)])#['concordance_index','log_likelihood',])#
    #
    #     search.fit(X, Y)
    #     results = pd.DataFrame(search.trials_dataframe())
    #     results_path = os.path.join(self.outpath, "{}_{}_optunasearch.csv".format(self.outcome_field, self.model))
    #     if os.path.isfile(results_path):
    #         prev_res = pd.read_csv(results_path)
    #         results = pd.concat([prev_res,results])
    #     results.to_csv(results_path, index=False)



    def run_gridsearch_pipeline(self):
        self.load_train_data()
        self.define_training_pipeline_elems()
        self.run_optuna_study()#grid()#
    #    self.run_grid()

   # def get

    def train_model(self,pipe_dict,params_dict):

        self.load_train_data()
        self.define_training_pipeline_elems()

        _,X,Y,_ = self.get_grid_for_model(pipe_dict['model'])

        imp = "passthrough"
        prep = "passthrough"
        model = sklearn_adapter(CoxPHFitter, event_col="event")()

        if pipe_dict['imp'] == "iter_zero":
            imp = self.iterative_zero_impute
        elif pipe_dict['imp'] == 'iter_mode':
            imp = self.iterative_mode_impute
        elif pipe_dict['imp'] == "simple":
            imp = self.simple_impute

        if pipe_dict['prep'] == 'standard':
            prep = self.standardizer
        elif pipe_dict['prep'] == 'quantile':
            prep = self.quantilizer

        if pipe_dict['model'] == 'rsf':
            model = RandomSurvivalForest(verbose=2, n_jobs=6, max_samples=0.5)
        elif pipe_dict['model'] == 'xgboost':
            model = XGBWrap()

        self.pipe = Pipeline([('imp', imp),
                              ('prep', prep),
                              ('model', model)])
     #   self.pipe.set_output(transform='pandas')
        self.pipe.set_params(**params_dict)

        self.pipe.fit(X,Y)

        if pipe_dict['model'] == 'regularized_cox':
            print(self.pipe['model'].lifelines_model)
            # #TODO: plot the top ones only
            # ax = self.pipe['model'].lifelines_model.plot()
            # fig_path = os.path.join(self.outpath,"{}_cox_model.png".format(self.outcome_field))
            # plt.tight_layout()
            # plt.savefig(fig_path)
            self.pipe['model'].lifelines_model.print_summary()
#            plt.clf()
            summary_df = self.pipe['model'].lifelines_model.summary
            summary_df['corrected'] = summary_df['p']*len(summary_df) # bonferroni correction
            summary_df = summary_df[summary_df['corrected']<0.05]
            summary_df = summary_df.sort_values('coef')
            print("{} significant features".format(len(summary_df)))
            siglist = summary_df.index.tolist()
            out_df = summary_df[['coef','exp(coef)','p','corrected']]
            out_df.to_csv(os.path.join(self.outpath,"{}_significant.csv".format(self.outcome_field)))
            toplist = self.pipe['model'].lifelines_model.hazard_ratios_.abs().nlargest(10).index.tolist()
            ax = self.pipe['model'].lifelines_model.plot(toplist)
            fig=ax.get_figure()
            fig.set_size_inches(12,6)
            fig_path = os.path.join(self.outpath,"{}_cox_model_top.png".format(self.outcome_field))
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.clf()
            ax = self.pipe['model'].lifelines_model.plot(siglist)
            fig=ax.get_figure()
            fig.set_size_inches(12,6)
            fig_path = os.path.join(self.outpath,"{}_cox_model_sig.png".format(self.outcome_field))
            plt.tight_layout()
            plt.savefig(fig_path)

            modelfile = "_".join(
                [str(x[0]) + "_" + str(x[1]) for x in pipe_dict.items()] + [str(x[0]) + "_" + str(x[1]) for x in
                                                                            params_dict.items()]) + "_cph_model.pkl"
            savepath = os.path.join(self.outpath, modelfile)
            with open(savepath, 'wb') as f:
                pickle.dump(self.pipe['model'].lifelines_model, f)

        pipefile = "_".join(
            [str(x[0])+"_"+str(x[1]) for x in pipe_dict.items()]+ [str(x[0])+"_"+str(x[1]) for x in params_dict.items()]) + "_pipeline.pkl"
        savepath = os.path.join(self.outpath,pipefile)
        with open(savepath,'wb') as f:
            pickle.dump(self.pipe,f)




    def apply_pipeline_to_tenk(self):
        CoxClass = sklearn_adapter(CoxPHFitter, event_col="event")
        parpath = os.path.abspath(os.path.join(self.outpath,os.pardir))
        pipefile = glob.glob(parpath+'/full_data/*_pipeline.pkl')
        assert(len(pipefile)==1), "Pipeline pickle file is not unique"
        with open(pipefile[0],'rb') as f:
            self.pipe = pickle.load(f)
      #      self.pipe.set_output(transform='pandas')
        if self.use_cache and os.path.isfile(self.tenk_features_path):
            tenk_data = pd.read_csv(self.tenk_features_path)
        else:
            # load tenk data
            tenk_data = load_prep_10k_mapped_features(features_table)
            ukbb_data = load_prep_ukbb_mapped_features(features_table)
            ukbb_data = rename_ukbb_fields_to_match_tenks_names(ukbb_data, tenk_data, features_table)
            tenk_data, ukbb_data = process_joint_features(features_table, tenk_data, ukbb_data, regularized)

            if self.use_cache and not os.path.isfile(self.tenk_features_path):
                    tenk_data.to_csv(self.tenk_features_path)

        tenk_data['event'] = np.nan
        preds = self.pipe.predict(tenk_data)
        preds = pd.DataFrame(preds,index=tenk_data['RegistrationCode'],columns=['predictions'])
        preds.to_csv(os.path.join(self.outpath,'tenk_predictions.csv'))
        return preds





