from LabData.DataAnalyses.UKBB_10k.UKBB_survival import *
from LabData.DataLoaders.DEXALoader import DEXALoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabData.DataLoaders.RetinaScanLoader import RetinaScanLoader
from LabData.DataLoaders.GutMBLoader import GutMBLoader
from LabData.DataLoaders.MedicalConditionLoader import MedicalConditionLoader
from LabData.DataLoaders.ItamarSleepLoader import ItamarSleepLoader
from LabData.DataLoaders.ECGTextLoader import ECGTextLoader
from LabData.DataLoaders.ABILoader import ABILoader
from LabData.DataLoaders.PhenoSleepLoader import PhenoSleepLoader
from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader
from LabData.DataLoaders.UltrasoundLoader import  UltrasoundLoader
from LabData.DataLoaders.SerumMetabolomicsLoader import SerumMetabolomicsLoader
from LabData.DataLoaders.DietLoggingLoader import DietLoggingLoader
from LabData.DataAnalyses.TenK_Trajectories.biological_age.predict_age import build_Xs_and_Ys

from LabData.DataAnalyses.TenK_Trajectories.utils import get_diet_logging_around_stage

from LabUtils.Utils import mkdirifnotexists

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from glob import glob


from LabQueue.qp import fakeqp as qp # qp #fakeqp as qp#

from LabUtils.addloglevels import sethandlers

model = "regularized_cox"
topdir = os.path.join(OUTPATH,model)

def load_outcome(outcome):

    pred_dir = os.path.join(topdir,str(outcome),"tenk","full_ukb")

    predfile = os.path.join(pred_dir,"tenk_predictions.csv")
    assert os.path.isfile(predfile), f'No prediction file for {outcome}'
    preds = pd.read_csv(predfile,index_col=0)
    print(f"Loaded predictions for {outcome}")
    return preds

def load_age_gender_bmi(drop_frac = 0.0):
    bm_data = add_age_gender_to_df(
                BodyMeasuresLoader().get_data(
                    study_ids=['10K'], research_stage=['baseline'],
                            groupby_reg='first', min_col_present_frac=drop_frac)).df
    bm_data = bm_data[['age','gender','bmi']]
    bm_data = bm_data.reset_index(level=[1]).drop(['Date'], axis=1)
    print("Loaded age, gender, bmi")
    return bm_data


def load_diet(drop_frac=0.1):
    dlld = DietLoggingLoader().get_data(study_ids=[10]).df
    df = get_diet_logging_around_stage(dlld, stage='baseline', delta_before=2, delta_after=60)
    # hack since couldn't get below to work for category names
    # TODO: LOOK AT NUTRIENTS OR SOMETHING ELSE INSTEAD
    df = DietLoggingLoader().add_food_categories(df)
    df['FoodCategoryID'] = df['MainCategoryEng']
    drop_frac = drop_frac if drop_frac is not None else 0
    df = DietLoggingLoader().daily_mean_food_consumption(df=df,kcal_limit=500, min_col_present_frac=drop_frac, level='FoodCategoryID').df
    return df

def load_dietnew(drop_frac=0.1):
    df = pd.read_csv('/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/Metabolomics_repeat/Xs/nutrients.csv')
    df = df.dropna(thresh=np.round(len(df)*drop_frac),axis=1).set_index('RegistrationCode')
    return df

def load_retina(drop_frac=0.1):
    df = RetinaScanLoader().get_data(study_ids=['10K'], research_stage=['baseline'],min_col_present_frac=drop_frac).df
    df.drop('participant_id', axis=1, inplace=True)
    left = df.xs('l_eye', level=2)
    right = df.xs('r_eye', level=2)
    df = pd.merge(left, right, on='RegistrationCode', suffixes=('_left', '_right'))
    return df


def load_microbiome(drop_frac=0.1):
    mb_data = GutMBLoader().get_data('segal_species', study_ids='10K', research_stage=['baseline'],
                                                  take_log=True, groupby_reg='first',
                                                  min_col_present_frac=drop_frac,min_col_val=1e-4)
    df = mb_data.df.set_index(mb_data.df_metadata['RegistrationCode'])
    ##################################################
    ## TODO: What to do about this V
    ## Leave as -4, or use nans ??
    df = df.replace(-4,np.nan)
    df = df.dropna(how='all')
    df = df.replace(np.nan,-9)

    name_map = mb_data.df_columns_metadata['species'].to_dict()
    for key, value in list(name_map.items()):
        if 'unknown' in value:
            name_map[key] = mb_data.df_columns_metadata.loc[key]['genus']+ '_genus_' + mb_data.df_columns_metadata.loc[key]['Unnamed: 0']
    for key, value in list(name_map.items()):
        if 'unknown' in value:
            name_map[key] = mb_data.df_columns_metadata.loc[key]['family']+ '_family'
        name_map[key] += '_'+ mb_data.df_columns_metadata.loc[key]['Unnamed: 0']
    df = df.rename(columns = name_map)

    return df


def load_microbiome_families(drop_frac=0.1):
    mb_data = GutMBLoader().get_data('segal_family', study_ids='10K', research_stage=['baseline'],
                                                  take_log=True, groupby_reg='first',
                                                  min_col_present_frac=drop_frac,min_col_val=1e-4)
    df = mb_data.df.set_index(mb_data.df_metadata['RegistrationCode'])
    ##################################################
    ## TODO: What to do about this V
    ## Leave as -4, or use nans ??
    df = df.replace(-4,np.nan)
    df = df.dropna(how='all')
    df = df.replace(np.nan,-9)

    name_map = mb_data.df_columns_metadata['family'].to_dict()
    for key, value in list(name_map.items()):
        if 'no_consensus' in value:
            name_map[key] = mb_data.df_columns_metadata.loc[key]['order']+ '_order'
    for key, value in list(name_map.items()):
        name_map[key] += '_'+ mb_data.df_columns_metadata.loc[key]['Unnamed: 0']
    df = df.rename(columns = name_map)

    return df


def load_microbiome_phyla(drop_frac=0.1):
    #code from sachie
    mbl_f_for_p = GutMBLoader().get_data('segal_family', study_ids='10K', research_stage=['baseline'],
                                                  take_log=False, groupby_reg='first',
                                                  min_col_present_frac=drop_frac,min_col_val=1e-4)
    p_name = mbl_f_for_p.df_columns_metadata['phylum']
    p_name = p_name.replace('Firmicutes_A', 'Firmicutes')
    p_name = p_name.replace('Firmicutes_B', 'Firmicutes')
    p_name = p_name.replace('Firmicutes_C', 'Firmicutes')
    abundance = mbl_f_for_p.df.T
    abundance = abundance.replace(0.000100,np.nan)
    abundance = abundance.dropna(how='all')
    abundance = abundance.replace(np.nan,0.000000001)
    data_p = pd.concat([p_name, abundance], axis=1)
    data_p = data_p.groupby('phylum').sum()
    data_p = data_p.T
    mbl_p = data_p.copy()
    mbl_p[mbl_p < 0.0001] = 0.000000001
    mbl_p = np.log10(mbl_p)
    df = mbl_p.join(mbl_f_for_p.df_metadata[['RegistrationCode']])
    df = df.drop(columns='no_consensus')
    df = df.set_index('RegistrationCode')

    return df


def load_dexa(drop_frac = .1):
    df = DEXALoader().get_data(study_ids=['10K'],
                                 research_stage=['baseline'],
                                 min_col_present_frac=drop_frac).df
    df = df.reset_index(level=[1]).drop(['Date'], axis=1)
    df = df.drop_duplicates()
    def g2kg(column):
        if column.name.endswith('_mass'):
            return column/1000
        return column
    df = df.apply(g2kg)
    return df


def load_sleep(drop_frac = .1):
    df = ItamarSleepLoader().get_data(study_ids=['10K'],
                                 research_stage=['baseline'],
                                 min_col_present_frac=drop_frac).df
  #  df = df.reset_index(level=[1]).drop(['Date'], axis=1)

    df = df.reset_index(level=[1,2],drop=True)
    #TODO: should it really just take the mean?
    df = df.groupby(level='RegistrationCode').mean(numeric_only=True)
    return df


def load_medical(drop_frac = .02): #TODO: changed frac (from .1), what should it be ??
    #TODO: this includes subjects that have registered but not been measured. What to do?
    bm_md = BodyMeasuresLoader().get_data(study_ids=['10K'], research_stage=['baseline']).df_metadata
    reg_ids = bm_md.index.get_level_values(0).tolist()
    #########################################################
    medical = MedicalConditionLoader().get_data(reg_ids=reg_ids,#study_ids=['10K'],
                                                research_stage=['baseline'])
    conditions_dict = dict(zip(medical.df_columns_metadata['column_name'], medical.df_columns_metadata['english_name']))
    df = medical.df.drop(columns='Start')
    df = df.reset_index()
    df_1hot = pd.get_dummies(df['medical_condition'])
    df = pd.concat([df,df_1hot],axis=1)
    df = df.set_index('RegistrationCode')
    df = df.drop(columns=['Date','medical_condition'])
    df = df.groupby('RegistrationCode').max()
    # TODO: what is the right % to use here v
    df = df[df.columns[df.sum()>len(df)*drop_frac]]
    df = df.rename(columns = conditions_dict)
    return df



def load_ecg(drop_frac = .1):
    df = ECGTextLoader().get_data(study_ids=['10K'],
                                 research_stage=['baseline'],
                                 min_col_present_frac=drop_frac).df
    df = df.reset_index(level=[1]).drop(['Date'], axis=1)
    df.drop(columns=['conclusion','st_t','non_confirmed_diagnosis','qrs'],inplace=True)
    df=df.transform(pd.to_numeric,axis=1)
    df = df.dropna(axis='columns',thresh=np.round(len(df)*drop_frac))

    return df


def load_carotid(drop_frac = 0.1):
    # carotid_cols = ["com_carotid_a_1_pulsatility_index","com_carotid_a_1_resistive_index","com_carotid_a_1_systolic_to_diastolic_ratio","com_carotid_a_bpm_1_heart_rate","com_carotid_a_m_s_1_end_diastolic_velocity","com_carotid_a_m_s_1_minimum_diastolic_velocity","com_carotid_a_m_s_1_peak_systolic_velocity","com_carotid_a_m_s_1_time_average_mean_velocity","com_carotid_a_m_s_1_time_averaged_peak_velocity","com_carotid_a_mm_1_sv_depth","com_carotid_a_mm_1_velocity_time_integral","com_carotid_a_mm_s2_1_acceleration_index","com_carotid_a_mmhg_1_mean_pressure_gradient","com_carotid_a_mmhg_1_peak_pressure_gradient","com_carotid_a_rad_1_doppler_angle","com_carotid_a_s_1_acceleration_time","distance_mm_1_distance","distance_mm_2_distance","distance_mm_3_distance","imt_tool_1_fit","imt_tool_2_fit","imt_tool_3_fit","imt_tool_4_fit","imt_tool_5_fit","imt_tool_6_fit","imt_tool_mm_1_intima_media_thickness","imt_tool_mm_1_window_width","imt_tool_mm_2_intima_media_thickness","imt_tool_mm_2_window_width","imt_tool_mm_3_intima_media_thickness","imt_tool_mm_3_window_width","imt_tool_mm_4_intima_media_thickness","imt_tool_mm_4_window_width","imt_tool_mm_5_intima_media_thickness","imt_tool_mm_5_window_width","imt_tool_mm_6_intima_media_thickness","imt_tool_mm_6_window_width","intima_media_th_1_fit","intima_media_th_2_fit","intima_media_th_3_fit","intima_media_th_4_fit","intima_media_th_mm_1_intima_media_thickness","intima_media_th_mm_1_window_width","intima_media_th_mm_2_intima_media_thickness","intima_media_th_mm_2_window_width","intima_media_th_mm_3_intima_media_thickness","intima_media_th_mm_3_window_width","intima_media_th_mm_4_intima_media_thickness","intima_media_th_mm_4_window_width"]
    carotid_cols = ["com_carotid_a_1_pulsatility_index", "com_carotid_a_1_resistive_index",
                    "com_carotid_a_1_systolic_to_diastolic_ratio", "com_carotid_a_bpm_1_heart_rate",
                    "com_carotid_a_m_s_1_end_diastolic_velocity", "com_carotid_a_m_s_1_minimum_diastolic_velocity",
                    "com_carotid_a_m_s_1_peak_systolic_velocity", "com_carotid_a_m_s_1_time_average_mean_velocity",
                    "com_carotid_a_m_s_1_time_averaged_peak_velocity", "com_carotid_a_mm_1_sv_depth",
                    "com_carotid_a_mm_1_velocity_time_integral", "com_carotid_a_mm_s2_1_acceleration_index",
                    "com_carotid_a_mmhg_1_mean_pressure_gradient", "com_carotid_a_mmhg_1_peak_pressure_gradient",
                    "com_carotid_a_rad_1_doppler_angle", "com_carotid_a_s_1_acceleration_time",
                    "distance_mm_1_distance", "distance_mm_2_distance", "distance_mm_3_distance", "imt_tool_1_fit",
                    "imt_tool_2_fit", "imt_tool_3_fit", "imt_tool_4_fit", "imt_tool_5_fit", "imt_tool_6_fit",
                    "imt_tool_mm_1_intima_media_thickness", "imt_tool_mm_1_window_width",
                    "imt_tool_mm_2_intima_media_thickness", "imt_tool_mm_2_window_width",
                    "imt_tool_mm_3_intima_media_thickness", "imt_tool_mm_3_window_width",
                    "imt_tool_mm_4_intima_media_thickness", "imt_tool_mm_4_window_width",
                    "imt_tool_mm_5_intima_media_thickness", "imt_tool_mm_5_window_width",
                    "imt_tool_mm_6_intima_media_thickness", "imt_tool_mm_6_window_width", "intima_media_th_1_fit",
                    "intima_media_th_2_fit", "intima_media_th_3_fit", "intima_media_th_4_fit",
                    "intima_media_th_mm_1_intima_media_thickness", #"intima_media_th_mm_1_window_width",
                    "intima_media_th_mm_2_intima_media_thickness", #"intima_media_th_mm_2_window_width",
                    "intima_media_th_mm_3_intima_media_thickness", #"intima_media_th_mm_3_window_width",
                    "intima_media_th_mm_4_intima_media_thickness"]#, #"intima_media_th_mm_4_window_width"]
    us = UltrasoundLoader().get_data(study_ids=['10K'], research_stage=['baseline'], cols=carotid_cols, min_col_present_frac=drop_frac)
    df = us.df.reset_index(level=[1]).drop(['Date'], axis=1)
    return df

def load_abi(drop_frac = 0.1):
    abi = ABILoader().get_data(study_ids=['10K'], research_stage=['baseline'], min_col_present_frac=drop_frac)
    df = abi.df
    def parse_duration(duration_str):
        if pd.notna(duration_str):
            parts = str(duration_str).split(":")
            seconds = int(parts[0])
            milliseconds = int(parts[1])
            microseconds = int(parts[2])
            return (seconds * 1000) + milliseconds + (microseconds / 1000)
        else:
            return duration_str
    for col in df.columns:
        if col.endswith('_duration'):
            df[col] = df[col].apply(parse_duration)
    df = df.reset_index(level=[1]).drop(['Date'], axis=1)
    return df

def load_liver(drop_frac = 0.1):
    liver_cols = ["att_plus_ssp_plus_db_cm_mhz_1_att_plus","att_plus_ssp_plus_db_cm_mhz_2_att_plus","att_plus_ssp_plus_db_cm_mhz_3_att_plus","att_plus_ssp_plus_db_cm_mhz_4_att_plus","att_plus_ssp_plus_db_cm_mhz_5_att_plus","att_plus_ssp_plus_m_s_1_ssp_plus","att_plus_ssp_plus_m_s_2_ssp_plus","att_plus_ssp_plus_m_s_3_ssp_plus","att_plus_ssp_plus_m_s_4_ssp_plus","att_plus_ssp_plus_m_s_5_ssp_plus","psv_edv_1_resistive_index","psv_edv_1_systolic_to_diastolic_ratio","psv_edv_2_resistive_index","psv_edv_2_systolic_to_diastolic_ratio","psv_edv_3_resistive_index","psv_edv_3_systolic_to_diastolic_ratio","psv_edv_4_resistive_index","psv_edv_4_systolic_to_diastolic_ratio","psv_edv_5_resistive_index","psv_edv_5_systolic_to_diastolic_ratio","psv_edv_m_s_1_end_diastolic_velocity","psv_edv_m_s_1_peak_systolic_velocity","psv_edv_m_s_2_end_diastolic_velocity","psv_edv_m_s_2_peak_systolic_velocity","psv_edv_m_s_3_end_diastolic_velocity","psv_edv_m_s_3_peak_systolic_velocity","psv_edv_m_s_4_end_diastolic_velocity","psv_edv_m_s_4_peak_systolic_velocity","psv_edv_m_s_5_end_diastolic_velocity","psv_edv_m_s_5_peak_systolic_velocity","psv_edv_mm_1_sv_depth","psv_edv_mm_2_sv_depth","psv_edv_mm_3_sv_depth","psv_edv_mm_4_sv_depth","psv_edv_mm_5_sv_depth","psv_edv_mmhg_1_peak_pressure_gradient","psv_edv_mmhg_2_peak_pressure_gradient","psv_edv_mmhg_3_peak_pressure_gradient","psv_edv_mmhg_4_peak_pressure_gradient","psv_edv_mmhg_5_peak_pressure_gradient","q_box_1_stability_index","q_box_2_stability_index","q_box_3_stability_index","q_box_4_stability_index","q_box_5_stability_index","q_box_kpa_1_max_elasticity","q_box_kpa_1_mean_elasticity","q_box_kpa_1_median_elasticity","q_box_kpa_1_min_elasticity","q_box_kpa_1_standard_deviation","q_box_kpa_2_max_elasticity","q_box_kpa_2_mean_elasticity","q_box_kpa_2_median_elasticity","q_box_kpa_2_min_elasticity","q_box_kpa_2_standard_deviation","q_box_kpa_3_max_elasticity","q_box_kpa_3_mean_elasticity","q_box_kpa_3_median_elasticity","q_box_kpa_3_min_elasticity","q_box_kpa_3_standard_deviation","q_box_kpa_4_max_elasticity","q_box_kpa_4_mean_elasticity","q_box_kpa_4_median_elasticity","q_box_kpa_4_min_elasticity","q_box_kpa_4_standard_deviation","q_box_kpa_5_max_elasticity","q_box_kpa_5_mean_elasticity","q_box_kpa_5_median_elasticity","q_box_kpa_5_min_elasticity","q_box_kpa_5_standard_deviation","q_box_m_s_1_max_elasticity","q_box_m_s_1_mean_elasticity","q_box_m_s_1_median_elasticity","q_box_m_s_1_min_elasticity","q_box_m_s_1_standard_deviation","q_box_m_s_2_max_elasticity","q_box_m_s_2_mean_elasticity","q_box_m_s_2_median_elasticity","q_box_m_s_2_min_elasticity","q_box_m_s_2_standard_deviation","q_box_m_s_3_max_elasticity","q_box_m_s_3_mean_elasticity","q_box_m_s_3_median_elasticity","q_box_m_s_3_min_elasticity","q_box_m_s_3_standard_deviation","q_box_m_s_4_max_elasticity","q_box_m_s_4_mean_elasticity","q_box_m_s_4_median_elasticity","q_box_m_s_4_min_elasticity","q_box_m_s_4_standard_deviation","q_box_m_s_5_max_elasticity","q_box_m_s_5_mean_elasticity","q_box_m_s_5_median_elasticity","q_box_m_s_5_min_elasticity","q_box_m_s_5_standard_deviation","q_box_m_s_khz_1_vi_plus_mean","q_box_m_s_khz_1_vi_plus_median","q_box_m_s_khz_1_vi_plus_standard_deviation","q_box_m_s_khz_2_vi_plus_mean","q_box_m_s_khz_2_vi_plus_median","q_box_m_s_khz_2_vi_plus_standard_deviation","q_box_m_s_khz_3_vi_plus_mean","q_box_m_s_khz_3_vi_plus_median","q_box_m_s_khz_3_vi_plus_standard_deviation","q_box_m_s_khz_4_vi_plus_mean","q_box_m_s_khz_4_vi_plus_median","q_box_m_s_khz_4_vi_plus_standard_deviation","q_box_m_s_khz_5_vi_plus_mean","q_box_m_s_khz_5_vi_plus_median","q_box_m_s_khz_5_vi_plus_standard_deviation","q_box_mean_kpa_max_elasticity","q_box_mean_kpa_mean_elasticity","q_box_mean_kpa_median_elasticity","q_box_mean_kpa_min_elasticity","q_box_mean_kpa_standard_deviation","q_box_mean_m_s_khz_vi_plus_mean","q_box_mean_m_s_khz_vi_plus_median","q_box_mean_m_s_khz_vi_plus_standard_deviation","q_box_mean_m_s_max_elasticity","q_box_mean_m_s_mean_elasticity","q_box_mean_m_s_median_elasticity","q_box_mean_m_s_min_elasticity","q_box_mean_m_s_standard_deviation","q_box_mean_mm_qbox_depth","q_box_mean_mm_qbox_diameter","q_box_mean_pa_s_vi_plus_mean","q_box_mean_pa_s_vi_plus_median","q_box_mean_pa_s_vi_plus_standard_deviation","q_box_median_kpa_max_elasticity","q_box_median_kpa_mean_elasticity","q_box_median_kpa_median_elasticity","q_box_median_kpa_min_elasticity","q_box_median_kpa_standard_deviation","q_box_median_m_s_khz_vi_plus_mean","q_box_median_m_s_khz_vi_plus_median","q_box_median_m_s_khz_vi_plus_standard_deviation","q_box_median_m_s_max_elasticity","q_box_median_m_s_mean_elasticity","q_box_median_m_s_median_elasticity","q_box_median_m_s_min_elasticity","q_box_median_m_s_standard_deviation","q_box_median_mm_qbox_depth","q_box_median_mm_qbox_diameter","q_box_median_pa_s_vi_plus_mean","q_box_median_pa_s_vi_plus_median","q_box_median_pa_s_vi_plus_standard_deviation","q_box_mm_1_qbox_depth","q_box_mm_1_qbox_diameter","q_box_mm_2_qbox_depth","q_box_mm_2_qbox_diameter","q_box_mm_3_qbox_depth","q_box_mm_3_qbox_diameter","q_box_mm_4_qbox_depth","q_box_mm_4_qbox_diameter","q_box_mm_5_qbox_depth","q_box_mm_5_qbox_diameter","q_box_pa_s_1_vi_plus_mean","q_box_pa_s_1_vi_plus_median","q_box_pa_s_1_vi_plus_standard_deviation","q_box_pa_s_2_vi_plus_mean","q_box_pa_s_2_vi_plus_median","q_box_pa_s_2_vi_plus_standard_deviation","q_box_pa_s_3_vi_plus_mean","q_box_pa_s_3_vi_plus_median","q_box_pa_s_3_vi_plus_standard_deviation","q_box_pa_s_4_vi_plus_mean","q_box_pa_s_4_vi_plus_median","q_box_pa_s_4_vi_plus_standard_deviation","q_box_pa_s_5_vi_plus_mean","q_box_pa_s_5_vi_plus_median","q_box_pa_s_5_vi_plus_standard_deviation"]
    df = UltrasoundLoader().get_data(study_ids=['10K'], research_stage=['baseline'], cols=liver_cols, min_col_present_frac=drop_frac).df
    df = df.reset_index(level=[1]).drop(['Date'], axis=1)
    return df

def load_metabolomics(drop_frac = 0.1):
    sm = SerumMetabolomicsLoader().get_data(study_ids=['10K'], precomputed_loader_fname='metab_10k_data_RT_clustering',
                                                   min_col_present_frac=drop_frac)
    sm.df['RegistrationCode'] = sm.df_metadata['RegistrationCode']
    df = sm.df.set_index('RegistrationCode')

    return df

def load_cgm(drop_frac = .1):
    #TODO: this should be updated to its own loader
    cgm = pd.read_csv('/net/mraid20/export/genie/LabData/Analyses/ayyak/CGM/iglu/iglu_no_tails_v2.csv')
    cgm = cgm.drop(columns=['Unnamed: 0', 'connection_id', 'research_stage'])
    cgm = cgm.set_index('registration_code')
    cgm.index.rename('RegistrationCode', inplace=True)
    cgm.index = cgm.index.map(lambda x: '10K_'+str(x))
    cgm = cgm.groupby(level='RegistrationCode').first()#mean(numeric_only=True)
    cgm = cgm.dropna(axis='columns', thresh=np.round(len(cgm) * drop_frac))
    return cgm


def load_bloodtests(drop_frac=.1):
    df = BloodTestsLoader().get_data(study_ids=['10K'], min_col_present_frac=drop_frac,
                                     groupby_reg='latest', research_stage=['baseline']).df
    df = df.reset_index(level=[1]).drop(['Date'], axis=1)
    return df


def load_bodymeasures(drop_frac=.1):
    df = BodyMeasuresLoader().get_data(
                    study_ids=['10K'], research_stage=['baseline'],
                    min_col_present_frac=drop_frac, groupby_reg='first').df
    df = df.reset_index(level=[1]).drop(['Date'], axis=1)
    df = df.drop(columns = ['bmi','age','gender','on_hormone_therapy','is_getting_period'],errors='ignore')
    hand_dict = {'Right':1, 'Left':0}
    df = df.replace({'dominant_hand':hand_dict})
    df = df.transform(pd.to_numeric, axis=1)
    return df

def load_cgm_main(drop_frac = .1):
    cgm = pd.read_csv('/net/mraid20/export/genie/LabData/Analyses/ayyak/CGM/iglu/iglu_no_tails_v2.csv')
    cgm = cgm.drop(columns=['Unnamed: 0', 'connection_id', 'research_stage'])
    cgm = cgm.set_index('registration_code')
    cgm.index.rename('RegistrationCode', inplace=True)
    cgm.index = cgm.index.map(lambda x: '10K_' + str(x))
    cgm = cgm.groupby(level='RegistrationCode').first()#mean(numeric_only=True)
    cgm = cgm[['in_range_70_180','J_index','MAGE','MODD','CV','HBGI','LBGI']] # 'eA1C',
    cgm = cgm.dropna(axis='columns', thresh=np.round(len(cgm) * drop_frac))
    return cgm

def load_diet_main(drop_frac=0.1):
    df = pd.read_csv('/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/Metabolomics_repeat/Xs/nutrients.csv')
    df = df[['RegistrationCode', 'Protein', 'Total lipid (fat)', 'Carbohydrate, by difference', 'Energy', 'Alcohol, ethyl',
             'Water', 'Fiber, total dietary', 'Sugars, Total']]
    df = df.dropna(thresh=np.round(len(df)*drop_frac),axis=1).set_index('RegistrationCode')
    return df

def load_sleep_main(drop_frac = .1):
    cols_list = ['percent_of_supine_sleep','variability_between_sleep_stages_percents','ahi','mean_saturation',
                 'desaturations_mean_nadir','snore_db_mean']
    df = PhenoSleepLoader().get_data(study_ids=['10K'], cols=cols_list,research_stage=['baseline'], min_col_present_frac=drop_frac).df
    df = df.groupby(level=['RegistrationCode']).mean().drop_duplicates()
    return df

def load_ecg_main(drop_frac = .1):
    # v tentative, according to initial pass of Yotam TODO: finalize
    col_list = ['r_r_ms','pr_ms','qrs_ms','q_ms','q_mv_V2','r_mv_V1','r_mv_V5']
    df = ECGTextLoader().get_data(study_ids=['10K'], cols = col_list,
                                 research_stage=['baseline'],
                                 min_col_present_frac=drop_frac).df
    df = df.reset_index(level=[1]).drop(['Date'], axis=1)
    df=df.transform(pd.to_numeric,axis=1)
    df = df.dropna(axis='columns',thresh=np.round(len(df)*drop_frac))
    return df

def load_retina_main(drop_frac = .1):
    col_list = ['Cup_width','CDR_vertical','Artery_Tortuosity_density','Vein_Tortuosity_density',
                'Vein_Average_width','Artery_Average_width','Artery_Vessel_density','Artery_Fractal_dimension']
    df = RetinaScanLoader().get_data(study_ids=['10K'], cols=col_list, research_stage=['baseline'], min_col_present_frac=drop_frac).df
    df = df.groupby('RegistrationCode').mean()
    return df

def load_metab_centers(drop_frac = .1):
    pass

def load_dexa_main(drop_frac=.1):
    # v tentative, according to initial pass of Yotam TODO: finalize
    cols_list = ['spine_l1_l4_t_score','femur_neck_mean_t_score',
                 'total_scan_vat_mass', 'total_scan_sat_mass', 'body_comp_total_fat_mass',
                 'body_comp_android_fat_mass','body_comp_gynoid_fat_mass','body_comp_total_lean_mass']
    df = DEXALoader().get_data(study_ids=['10K'], cols=cols_list,
                                 research_stage=['baseline'],
                                 min_col_present_frac=drop_frac).df
    df = df.reset_index(level=[1]).drop(['Date'], axis=1)
    name_dict = {'body_comp_total_fat_mass' : 'total_fat_mass',
                 'body_comp_android_fat_mass' : 'android_fat_mass',
                 'body_comp_gynoid_fat_mass' : 'gynoid_fat_mass',
                 'body_comp_total_lean_mass' : 'total_lean_mass'}
    df = df.rename(columns=name_dict)
    df = df.drop_duplicates()
    def g2kg(column):
        if column.name.endswith('_mass'):
            return column/1000
        return column
    df = df.apply(g2kg)
    return df

def load_abi_main(drop_frac = 0.1):
    #TODO: v tentative list, confirm.
    cols_list = ["from_r_thigh_to_r_ankle_pwv", "from_l_thigh_to_l_ankle_pwv",
                 "r_abi", "l_abi"]
    abi = ABILoader().get_data(study_ids=['10K'], research_stage=['baseline'], cols=cols_list, min_col_present_frac=drop_frac)
    df = abi.df
    df['min_abi'] = df[["r_abi", "l_abi"]].min(axis=1)
    df['max_thigh_to_ankle_pwv'] = df[["from_r_thigh_to_r_ankle_pwv", "from_l_thigh_to_l_ankle_pwv"]].max(axis=1)
    df = df.drop(columns = cols_list)
    df = df.reset_index(level=[1]).drop(['Date'], axis=1)
    return df

def load_carotid_main(drop_frac = 0.1):
    cols_list= ["intima_media_th_mm_1_intima_media_thickness","intima_media_th_mm_2_intima_media_thickness"]
    us = UltrasoundLoader().get_data(study_ids=['10K'], research_stage=['baseline'], cols=cols_list, min_col_present_frac=drop_frac)
    df = us.df.reset_index(level=[1]).drop(['Date'], axis=1)
    df['max_intima_media_thickness'] = df[cols_list].max(axis=1)
    df = df.drop(columns = cols_list)
    return df


def load_liver_main(drop_frac = 0.1):
    cols_list =  ['att_plus_ssp_plus_m_s_1_ssp_plus', 'att_plus_ssp_plus_db_cm_mhz_1_att_plus',
                  'att_plus_ssp_plus_db_cm_mhz_2_att_plus','att_plus_ssp_plus_db_cm_mhz_3_att_plus',
                  'att_plus_ssp_plus_m_s_3_ssp_plus','att_plus_ssp_plus_m_s_2_ssp_plus',
                  "q_box_mean_kpa_mean_elasticity", "q_box_mean_m_s_khz_vi_plus_mean"]
    df = UltrasoundLoader().get_data(study_ids=['10K'], research_stage=['baseline'], cols=cols_list, min_col_present_frac=drop_frac).df
    df["att_mean_m_s"] = df[['att_plus_ssp_plus_m_s_1_ssp_plus','att_plus_ssp_plus_m_s_2_ssp_plus','att_plus_ssp_plus_m_s_3_ssp_plus']].mean(axis=1)
    df["att_mean_db_cm_mhz"] = df[['att_plus_ssp_plus_db_cm_mhz_1_att_plus','att_plus_ssp_plus_db_cm_mhz_2_att_plus','att_plus_ssp_plus_db_cm_mhz_3_att_plus']].mean(axis=1)
    df = df.drop(columns=['att_plus_ssp_plus_m_s_1_ssp_plus','att_plus_ssp_plus_m_s_2_ssp_plus','att_plus_ssp_plus_m_s_3_ssp_plus','att_plus_ssp_plus_db_cm_mhz_1_att_plus','att_plus_ssp_plus_db_cm_mhz_2_att_plus','att_plus_ssp_plus_db_cm_mhz_3_att_plus'])
    df = df.rename(columns={"q_box_mean_kpa_mean_elasticity" : "elasticity_mean_kpa",
                            "q_box_mean_m_s_khz_vi_plus_mean" : "vi_mean_m_s_khz"})
    df = df.reset_index(level=[1]).drop(['Date'], axis=1)
    return df

def get_loader_vars(loader,drop_frac=.1):
    loaders_dict = {'retina':load_retina, 'microbiome_families':load_microbiome_families, 'microbiome':load_microbiome, #'mental':load_mental,
                    'sleep' : load_sleep, 'dexa' : load_dexa, 'medical_conditions' : load_medical, 'carotid': load_carotid,
                    'ecg' : load_ecg, 'cgm' : load_cgm, 'bloodtests' : load_bloodtests, 'dietnew' : load_dietnew, 'abi':load_abi, # 'diet' : load_diet,
                    'bodymeasures' : load_bodymeasures, 'age_gender_bmi' : load_age_gender_bmi,'metabolomics':load_metabolomics,
                    'liver':load_liver,'microbiome_phyla_main':load_microbiome_phyla,'cgm_main':load_cgm_main,
                    'sleep_main':load_sleep_main,'ecg_main':load_ecg_main, 'dexa_main' : load_dexa_main,'retina_main':load_retina_main,
                    'abi_main':load_abi_main,'carotid_main':load_carotid_main,'liver_main':load_liver_main, 'diet_main': load_diet_main}#'metabolomics_cluster':load_metab_centers, 'retina_main':load_retina_main}

    if loader=='medical_conditions' and drop_frac==.1:
        drop_frac = .02 # SUPER HACKY TODO: fix this

    df = loaders_dict[loader](drop_frac)

    print(f"Loaded: {loader}")
    return df



def get_system_vars(system_file):
    df = pd.read_csv(system_file, index_col=0)
    return df


def run_partial_corr(merged_df,var,method,by_gender=False):
    results = {}
    print(var)
    stats = merged_df.partial_corr(y='predictions', x=var, covar=['age', 'gender', 'bmi'], method=method)
    results[var] = {'pval': stats['p-val'].values[0], 'r': stats['r'].values[0], 'n':stats['n'].values[0]}
    if by_gender:
        m_merged = merged_df[merged_df['gender'] == 1]
        f_merged = merged_df[merged_df['gender'] == 0]
        m_stats = m_merged.partial_corr(y='predictions',x=var,covar=['age','bmi'],method=method)
        results[var].update({'m_pval': m_stats['p-val'].values[0], 'm_r': m_stats['r'].values[0], 'm_n':m_stats['n'].values[0]})
        f_stats = f_merged.partial_corr(y='predictions', x=var, covar=['age', 'bmi'], method=method)
        results[var].update(
            {'f_pval': f_stats['p-val'].values[0], 'f_r': f_stats['r'].values[0], 'f_n': f_stats['n'].values[0]})

    return results

def get_partial_corrs(outcome,loader,method='spearman'):
    #TODO: Check if already done and only run if not
    outcome_preds = load_outcome(outcome).sort_index()
    covariates = load_age_gender_bmi().sort_index()
    loader_vars = get_loader_vars(loader).sort_index()
    loader_vars.drop(list(loader_vars.filter(regex='participant_id')),axis = 1, inplace = True, errors='ignore')
 #   merged = pd.concat([outcome_preds,covariates,loader_vars],axis=1)

    results = {}
    for var in loader_vars.columns:
        var_df = pd.concat([outcome_preds,covariates,loader_vars[var]],axis=1).dropna(
                    subset=var).dropna(subset=outcome_preds.columns.to_list())

        results[var] = run_partial_corr(var_df, var, method, by_gender=True)[var]

    print(f"Computed correlations for {loader} to {outcome}")
    res_df = pd.DataFrame(results).T
    return res_df

def get_system_partial_corrs(outcome,system_file,method='spearman'):
    #TODO: Check if already done and only run if not
    outcome_preds = load_outcome(outcome).sort_index()
    covariates = load_age_gender_bmi().sort_index()
    system_vars = get_system_vars(system_file).sort_index()
    system_vars = system_vars.drop(['age','gender','bmi'],errors='ignore',axis=1)
    system_vars.drop(list(system_vars.filter(regex='participant_id')),axis = 1, inplace = True, errors='ignore')

   # merged = pd.concat([outcome_preds,covariates,system_vars],axis=1)
   # merged = merged.loc[:, ~merged.columns.duplicated()] # the systems should have gender...

    results = {}
    for var in system_vars.columns:
        var_df = pd.concat([outcome_preds,covariates,system_vars[var]],axis=1).dropna(
                    subset=var).dropna(subset=outcome_preds.columns.to_list())

        results[var] = run_partial_corr(var_df, var, method, by_gender=True)[var]

    print(f"Computed correlations for {os.path.basename(system_file)} to {outcome}")
    res_df = pd.DataFrame(results).T
    return res_df

def run_loaders_vs_outcomes(loaders, outcomes_dict, method='spearman', cancer=False):
    outcome_corrs = []
    old_cwd = os.getcwd()
    old_cwd = os.getcwd()
    os.chdir(OUTPATH)

    results = {}
    with qp(jobname='partial_corr', _delete_csh_withnoerr=False, q=['himem7.q'], _trds_def=1, max_r=500,
            _mem_def='2G') as q:
        q.startpermanentrun()

        for outcome in outcomes_dict.keys():

            pred_dir = os.path.join(topdir, str(outcome), "tenk","full_ukb")

            predfile = os.path.join(pred_dir, "tenk_predictions.csv")
            if not os.path.isfile(predfile): continue
            #  corrs = pd.DataFrame(columns = ['pval','r','corrected_outcome_vs_loader'])
            for loader in loaders:
                # corrs = pd.concat([corrs,get_partial_corrs(outcome,loader,'pearson')])

                results[(outcome, loader)] = q.method(get_partial_corrs, (outcome, loader, method))
        results = {k: q.waitforresult(v) for k, v in results.items()}

    os.chdir(old_cwd)

    loader_corrs = {loader: pd.concat([results[k] for k in results.keys() if k[1] == loader],
                                      axis=1, keys=[outcomes_dict[o[0]] for o in results.keys() if o[1] == loader]) for
                    loader in loaders}

    # all_corrs = pd.concat(outcome_corrs,axis=1,keys = [outcomes_dict[o] for o in outcomes["UKBB Field ID"]])

    # all_corrs.to_csv(os.path.join(topdir,'all_outcome_partial_corrs.csv'))
    for loader in loaders:
        mkdirifnotexists(os.path.join(topdir,method))
        filename = f'all_outcomes_{loader}_partial_corrs.csv'
        if cancer:
            filename = f'all_cancer_outcomes_{loader}_partial_corrs.csv'

        mkdirifnotexists(os.path.join(topdir, method, 'full_ukb'))
        loader_corrs[loader].to_csv(os.path.join(topdir, method, 'full_ukb', filename))

def run_body_systems_vs_outcomes(system_files, outcomes_dict, method='spearman', cancer=False):
    outcome_corrs = []
    systems = [os.path.splitext(os.path.basename(path))[0] for path in system_files]
    old_cwd = os.getcwd()
    old_cwd = os.getcwd()
    os.chdir(OUTPATH)

    results = {}
    with qp(jobname='partial_corr', _delete_csh_withnoerr=False, q=['himem7.q'], _trds_def=1, max_r=500,
            _mem_def='2G') as q:
        q.startpermanentrun()

        for outcome in outcomes_dict.keys():

            pred_dir = os.path.join(topdir, str(outcome), "tenk","full_ukb")

            predfile = os.path.join(pred_dir, "tenk_predictions.csv")
            if not os.path.isfile(predfile): continue
            #  corrs = pd.DataFrame(columns = ['pval','r','corrected_outcome_vs_loader'])
            for ind, system_file in enumerate(system_files):
                results[(outcome, systems[ind])] = q.method(get_system_partial_corrs, (outcome, system_file, method))
        results = {k: q.waitforresult(v) for k, v in results.items()}

    os.chdir(old_cwd)

    system_corrs = {system: pd.concat([results[k] for k in results.keys() if k[1] == system],
                                      axis=1, keys=[outcomes_dict[o[0]] for o in results.keys() if o[1] == system]) for
                    system in systems}

    # all_corrs = pd.concat(outcome_corrs,axis=1,keys = [outcomes_dict[o] for o in outcomes["UKBB Field ID"]])

    # all_corrs.to_csv(os.path.join(topdir,'all_outcome_partial_corrs.csv'))
    for system in systems:
        mkdirifnotexists(os.path.join(topdir, method))
        filename = f'all_outcomes_systems_partial_corrs_{system}.csv'
        if cancer:
            filename = f'all_cancer_outcomes_systems_partial_corrs_{system}.csv'

        system_corrs[system].to_csv(os.path.join(topdir, method, 'full_ukb', filename))



if __name__ == '__main__':


    #build_Xs_and_Ys()

    sethandlers()
    outcomes = pd.read_csv(OUTCOMES_LIST).astype({"UKBB Field ID": 'int'})
    outcomes_dict = dict(zip(outcomes['UKBB Field ID'].values, outcomes['UKBB Description'].values))
    for o in outcomes_dict:
        outcomes_dict[o] = '('.join(outcomes_dict[o][:-1].split('(')[1:])

    cancer_outcomes = pd.read_csv(CANCER_OUTCOMES)
    cancer_outcomes_dict = dict(zip(cancer_outcomes['ICD10'].values, cancer_outcomes['Type of cancer'].values))


    loaders = ['dietnew','diet_main']#'medical_conditions','diet', 'retina', 'microbiome', 'sleep', 'dexa', 'ecg', 'cgm', \
                #'bloodtests', 'bodymeasures','carotid','abi','metabolomics','liver']
    high_level_loaders = ['dexa_main','microbiome_phyla_main','cgm_main','sleep_main','ecg_main',\
                          'retina_main','abi_main','carotid_main','liver_main', 'diet_main'] # 'metabolomics_cluster',


    systems_files = glob(DATAPATH+'/body_systems/Xs/*.csv')
    systems_files = [f for f in systems_files if 'male' not in f]

    new_outcomes = dict((k,outcomes_dict[k]) for k in [130896,131702])

    run_loaders_vs_outcomes(loaders, cancer_outcomes_dict, cancer=True)
 #   run_body_systems_vs_outcomes(systems_files, cancer_outcomes_dict, cancer=True)

    run_loaders_vs_outcomes(loaders, outcomes_dict)
  #  run_body_systems_vs_outcomes(systems_files, outcomes_dict)







