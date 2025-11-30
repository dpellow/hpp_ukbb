# try to run validations:
# correlation/odds ratio of diabetes, renal, cardiovascular outcomes to changes over time in hba1c, egfr, and to ascvd risk score
#
#

import os, glob, re
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy
from scipy import stats
from scipy.stats import norm


import statsmodels.api as sm
from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader
from LabData.DataLoaders.Medications10KLoader import Medications10KLoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabData.DataLoaders.LifeStyleLoader import LifeStyleLoader
from LabData.DataAnalyses.UKBB_10k.UKBB_survival import *
from LabData.DataAnalyses.UKBB_10k.process_predictions import load_medical, load_outcome, load_age_gender_bmi, load_microbiome





def get_multiples(df):
    s = df.copy()       
    dates = pd.to_datetime(s.index.get_level_values(1), errors="coerce")
    mask = dates.notna()
    s = s[mask]

    s.index = pd.MultiIndex.from_arrays(
        [s.index.get_level_values(0), dates[mask]],
        names=["RegistrationCode", "Date"]
    )

    valid_ids = s.index.get_level_values("RegistrationCode").value_counts()
    valid_ids = valid_ids[valid_ids > 1].index
    s = s.loc[s.index.get_level_values("RegistrationCode").isin(valid_ids)]
    return s

def first_last_diff(x):
    x = x.sort_index(level="Date")
    first_val = x.iloc[0]
    last_val = x.iloc[-1]
    first_date = x.index.get_level_values("Date")[0]
    last_date = x.index.get_level_values("Date")[-1]
    interval_years = (last_date - first_date).days / 365.25
    return pd.Series({
        "baseline": first_val,
        "end": last_val,
        "diff": last_val - first_val,
        "interval_years": round(interval_years, 2),
        "first_date" : first_date
    })


def process_df(df):
    df = df.dropna()
    df = get_multiples(df)
    print(df.head(60))
    df = df.groupby(level="RegistrationCode").apply(first_last_diff)
    df = df.unstack()
    print(df.head(60))
    
    df = df[(df['interval_years'] > 2) & (df['interval_years']<10)]  # only keep those with more than 2 years between first and last
    df['slope'] = df['diff'] / df['interval_years']
    return df
    
def add_age_gender_bmi(df):
    age_gender_bmi=load_age_gender_bmi()
    df = df.join(age_gender_bmi)
    return df

def add_dm_meds(df):
    meds = Medications10KLoader().get_data(study_ids=['10K'])
    mdf = meds.df
    meta = meds.df_metadata
    dm_meds_list = ["סקסנדה / Saxenda",
                    "אוזמפיק / Ozempic",
                    "מטפורמין טבע / Metformin Teva",
                    "גלוקומין / Glucomin",
                    "ג'נואט / Januet",
                    "גלוקופג' / Glucophage",
                    "ג'רדיאנס / Jardiance",
                    "רפגליניד־טבע / Repaglinide-Teva",
                    "פורקסיגה / Forxiga",
                    "יוקריאס / Eucreas",
                    "נובונורם / Novonorm",
                    "ויקטוזה / Victoza",
                    "אמריל / Amaryl",
                    "אפידרה / Apidra",
                    "בסגלר / Basaglar",
                    "אינסולין הומלוג מיקס / Humalog Mix",
                    "נובורפיד / Novorapid",
                    "גלובן / Gluben",
                    "אינסולין מיקסטרד 30 / Insulin Mixtard 30",
                    "טרוליסיטי / Trulicity",
                    "ג'ארדיאנס דואו / Jardiance Duo",
                    "קסיגדו / Xigduo"
                ]

    mask = meta.index.get_level_values("medication").str.contains(
                                            "|".join(dm_meds_list),case=False,na=False)
    meta = meta[mask]
    meta = meta.dropna(subset=['start_year'])
    meta = meta.sort_values(["start_year", "start_month"])
    meta = meta.groupby(level="RegistrationCode").head(1)    
    print(meta)
    meta = meta[['start_year', 'start_month']].droplevel([-2, -1])
    print(meta)
    df = df.join(meta, how='left')
    df['first_date'] = pd.to_datetime(df['first_date'])
    df['started'] = ((df['first_date'].dt.year<df['start_year']) | \
                        (df['first_date'].dt.year == df['start_year']) & \
                                (df['first_date'].dt.month<df['start_month'])).astype(int)
    df['started'] = df['started'].fillna(0)
    df = df.drop(columns=['start_year', 'start_month','first_date'])
    
    return df

def get_htn_meds():
    meds = Medications10KLoader().get_data(study_ids=['10K'])
    mdf = meds.df
    meta = meds.df_metadata
    htn_meds_list = ["קרדילוק / Cardiloc",
                        "אמלודיפין־טבע / Amlodipine-Teva",
                        "דיזותיאזיד / Disothiazide",
                        "ולסרטן דקסל / Valsartan Dexcel",
                        "וקטור / Vector",
                        "ביסופרולול פומראט / Bisoprolol Fumarate",
                        "טריטייס / Tritace",
                        "קנדור / CANDOR",
                        "רמיפריל טבע / Ramipril Teva",
                        "אמלו / Amlow",
                        "לרקפרס / Lercapress",
                        "אנלדקס / Enaladex",
                        "אנלפריל / Enalapril",
                        "לוסרדקס / Losardex",
                        "סילריל / Cilaril",
                        "דיובן / Diovan",
                        "נורמיטן / Normiten",
                        "לופרסור / Lopresor",
                        "קונברטין / Convertin",
                        "אלדקטון / Aldactone",
                        "פרולול / Prolol",
                        "ורפרס / Verapress SR",
                        "נורמלול / Normalol",
                        "אוקסר / Ocsaar",
                        "דרלין / Deralin",
                        "ספירונולקטון טבע / Spironolactone Teva",
                        "נורמופרסן / Normopresan",
                        "קרדורל / Cardoral",
                        "סוטלול / Sotalol",
                        "פוסיד / Fusid",
                        "תוריד / Torid",
                        "איקפרס / Ikapress",
                        "אדיזם / Adizem",
                        "נאובלוק / Neobloc",
                        "טרזוסין / Terazosin",
                        "לרקאנידיפין / Lercanidipine",
                        "נורווסק / Norvasc",
                        "קונקור / Concor",
                        "סלואו דרלין / Slow Deralin",
                        "מטופרולול / Metoprolol",
                        "נורליפ / Norlip",
                        "לופרסור אורוס / Lopresor oros",
                        "פרופרנולול הידרוכלוריד רטרד / Propranolol",
                        "דופלקס 5/80 / DUPLEX 5/80",
                        "וזודיפ קומבו / Vasodip Combo",
                        "קו־דיובן / Co-Diovan",
                        "אקספורג' / Exforge",
                        "וקטור פלוס / Vector Plus",
                        "אטקנד פלוס / Atacand Plus",
                        "סילריל פלוס / Cilaril Plus",
                        "לוסרדקס פלוס / Losardex Plus",
                        "לוסרטה פלוס / Losarta Plus",
                        "לוטן פלוס / Lotan Plus",
                    ]
    mask = meta.index.get_level_values("medication").str.contains(
                                            "|".join(htn_meds_list),case=False,na=False)
    meta = meta[mask]
    meta = meta.dropna(subset=['start_year'])
    meta.to_csv('~/proj_data/htn_meta.csv')


def get_score2_vars():
    bt = BloodTestsLoader().get_data(study_ids=['10K'], research_stage=['baseline'], groupby_reg='first', min_col_present_frac=0.1)
    df = bt.df
    mdf = bt.df_metadata
    merged = pd.concat([df, mdf], axis=1)
    
    vars = merged[['bt__total_cholesterol','bt__hdl_cholesterol']].rename(columns={
        'bt__total_cholesterol':'tot_chol',
        'bt__hdl_cholesterol':'hdl'
    })
    vars = vars*0.02586 # convert to mmol/L    
    
    vars = add_age_gender_bmi(vars)
    
    bm = BodyMeasuresLoader().get_data(study_ids=['10K'], research_stage=['baseline'],groupby_reg='first').df
    bm = bm.droplevel(-1)
    bm = bm['sitting_blood_pressure_systolic']
    vars = vars.join(bm, how='left').rename(columns={'sitting_blood_pressure_systolic':'sbp'})
    
    ls = LifeStyleLoader().get_data(study_ids=['10K'], research_stage=['baseline'],groupby_reg='first').df
    ls = ls.droplevel(-1)
    ls = ls['smoke_tobacco_now']
    
    vars = vars.join(ls, how='left')
    vars['smoker'] = (vars['smoke_tobacco_now']>0).astype(int)
    vars = vars.drop(columns=['smoke_tobacco_now'])
    return vars

def transform_score2(vars):
    vars['age'] = (vars['age']-60)/5
    vars['sbp'] = (vars['sbp']-120)/20
    vars['tot_chol'] = (vars['tot_chol']-6)
    vars['hdl'] = (vars['hdl']-1.3)/0.5
    vars['smoke_age'] = vars['smoker'] * (vars['age'])
    vars['sbp_age'] = vars['sbp'] * (vars['age'])
    vars['tot_age'] = vars['tot_chol'] * (vars['age'])
    vars['hdl_age'] = vars['hdl'] * (vars['age'])
    
    return vars


def compute_score2(vars):
    m_vars = vars[vars['gender'] == 1]
    f_vars = vars[vars['gender'] == 0]
    
    m_coeffs = [0.3742,0.6012,0.2777, 0.1458, -0.2698,-0.0755, -0.0255,-0.0281,0.0426]
    m_surv = 0.9605
    
    f_coeffs = [ 0.4648, 0.7744, 0.3131, 0.1002, -0.2606, -0.1088, -0.0277, -0.0226, 0.0613]
    f_surv = 0.9776
    
    m_risk = 1-m_surv**np.exp(np.dot(m_vars[['age','smoker','sbp','tot_chol','hdl','smoke_age',
                                                'sbp_age','tot_age','hdl_age']], m_coeffs))

    
    f_risk = 1-f_surv**np.exp(np.dot(f_vars[['age','smoker','sbp','tot_chol','hdl','smoke_age',
                                                'sbp_age','tot_age','hdl_age']], f_coeffs))
    
    calibrated_m_risk = (1-np.exp(-np.exp(-0.5699+0.7476*np.log(-np.log(1-m_risk)))))*100
    calibrated_f_risk = (1-np.exp(-np.exp(-0.7380+0.7019*np.log(-np.log(1-f_risk)))))*100
    
    calibrated_risk = pd.concat([pd.Series(calibrated_m_risk, index=m_vars.index),
                                 pd.Series(calibrated_f_risk, index=f_vars.index)])
    calibrated_risk = pd.concat([calibrated_risk,pd.concat([m_vars[['age','gender','bmi']],f_vars[['age','gender','bmi']]])], axis=1)
    
    calibrated_risk['age'] = calibrated_risk['age']*5+60
    
    calibrated_risk = calibrated_risk.sort_index()
    calibrated_risk.to_csv('~/proj_data/score2.csv')

    
def get_score2():
    vars = get_score2_vars()
    vars = transform_score2(vars)
    vars = vars.dropna(how='any')
    df = compute_score2(vars)
    
    
def get_bt_changes():
    
    # get blood tests
    bt = BloodTestsLoader().get_data(study_ids=['10K'], min_col_present_frac=0.1)
    df = bt.df
    mdf = bt.df_metadata
    
    merged = pd.concat([df, mdf], axis=1)
    
    df_hba1c = df['bt__hba1c']
    df_egfr = df['bt__egfr']
    
    df_hba1c_proc = process_df(df_hba1c)
    df_egfr_proc = process_df(df_egfr)
    
    df_hba1c_proc = add_age_gender_bmi(df_hba1c_proc)
    df_egfr_proc = add_age_gender_bmi(df_egfr_proc)
    df_egfr_proc = df_egfr_proc.drop(columns=['first_date'])
    
    print(df_hba1c_proc)    
    
    df_hba1c_proc = add_dm_meds(df_hba1c_proc)
    
    
    print(df_hba1c_proc)
    print(df_egfr_proc.head())
    
    df_hba1c_proc.to_csv('~/proj_data/hba1c_changes.csv')
    df_egfr_proc.to_csv('~/proj_data/egfr_changes.csv')
    
if __name__ == "__main__":
    # get_bt_changes()
    # get_htn_meds()
    get_score2()

    
    
    
    
