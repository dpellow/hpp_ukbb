# imports
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
from LabData.DataLoaders.MedicalConditionLoader import MedicalConditionLoader
from LabData.DataAnalyses.UKBB_10k.UKBB_survival import *
from LabData.DataAnalyses.UKBB_10k.process_predictions import load_medical, load_outcome, load_age_gender_bmi, load_microbiome


# general vars

outdir = '/net/mraid20/export/jafar/UKBioBank/davidpel/UKBB_10K_outputs'
method = 'regularized_cox'
correlation = 'spearman' #'pearson'

topdir = os.path.join(outdir,method)
corrdir = os.path.join(topdir,correlation,'full_ukb')

tenk_features_short_names_dict = {"glasses/contacts_condition (For long-sightedness, i.e. for distance and near, but particularly for near tasks like reading (called 'hypermetropia'))":"hypermetropia",
                                  'occuerd_in_past_two_years (Serious illness, injury or assault to yourself)':'Serious illness, injury or assault (past 2 years)',
                                  'weight_changes_in_the_last_year (Yes - lost weight)':'lost weight in the last year',
                                  'diet_major_changes_5years (Yes, because of illness)' : 'diet_major_changes_5years due to illness',
                                  "experiences_interfering_with_usual_activities (Stomach or abdominal pain)" : "Stomach or abdominal pain",
                                  "experiences_interfering_with_usual_activities (Knee pain)" : "Knee pain",
                                  "experiences_interfering_with_usual_activities (Neck or shoulder pain)" : "Neck or shoulder pain",
                                  "experiences_interfering_with_usual_activities (Back pain)" : "Back pain",
                                  "experiences_interfering_with_usual_activities (Headache)" : "Headache",
                                  "people_living_together_retalated (Son and/or daughter (include step-children))" : "live_together (Child/ren)",
                                  "activities_type (Heavy DIY (eg: weeding, lawn mowing, carpentry, digging))" : "activities_type (Heavy DIY)",
                                 }

tenk_prs_short_names_dict = {"Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor: Hayfever, allergic rhinitis or eczema":"Hayfever, allergic rhinitis or eczema",
                             "Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor: None of the above":"No clot, dvt, bronchitis, asthma, allergy",
                             "Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor: Asthma":"Diagnosed asthma",
                             "Vascular/heart problems diagnosed by doctor: None of the above" : "No heart problems diagnosed by doctor",
                             "Vascular/heart problems diagnosed by doctor: High blood pressure" : "High blood pressure diagnosed by doctor",
                             "Non-cancer illness code, self-reported: hypertension" : "Hypertension - self-reported code",
                             "Non-cancer illness code, self-reported: heart attack/myocardial infarction" : "Heart attack/myocardial infarction - self-reported code",
                             "Non-cancer illness code, self-reported: diabetes" : "Diabetes - self-reported code",
                             "Non-cancer illness code, self-reported: high cholesterol" : "High cholestrerol - self-reported code",
                             "Non-cancer illness code, self-reported: asthma" : "Asthma - self-reported code",
                             'Diagnoses - main ICD10: I25 Chronic ischaemic heart disease' : "Chronic ischaemic heart disease",
                              }

tenk_prs_order = ['Pulse rate, automated reading', 'Non-cancer illness code, self-reported: hypertension',
                    'Non-cancer illness code, self-reported: heart attack/myocardial infarction',
                    'Illnesses of father: Heart disease', 'Illnesses of father: High blood pressure', 'Illnesses of mother: High blood pressure',
                    'Illnesses of siblings: Heart disease', 'Illnesses of siblings: High blood pressure','Chest pain or discomfort',
                    'Age high blood pressure diagnosed', 'Diastolic blood pressure, automated reading', 'Systolic blood pressure, automated reading',
                    'Pulse rate', 'Pulse wave reflection index', 'Vascular/heart problems diagnosed by doctor: Heart attack',
                    'Vascular/heart problems diagnosed by doctor: Angina', 'Vascular/heart problems diagnosed by doctor: High blood pressure',
                    'Major coronary heart disease event', 'Ischaemic heart disease, wide definition',
                    'Diseases of the circulatory system',
                    'Coronary atherosclerosis', 'Non-cancer illness code, self-reported: angina',
                    'Diagnoses - main ICD10: I25 Chronic ischaemic heart disease',
                    'Non-cancer illness code, self-reported: diabetes', 'Non-cancer illness code, self-reported: high cholesterol',
                    'Illnesses of father: Diabetes', 'Illnesses of mother: Diabetes', 'Illnesses of siblings: Diabetes',
                    'Diabetes diagnosed by doctor', 'Diagnoses - main ICD10: G56 Mononeuropathies of upper limb',
                    'Wheeze or whistling in the chest in last year', 'Non-cancer illness code, self-reported: asthma',
                    'Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor: Asthma',  
                    'Non-cancer illness code, self-reported: hayfever/allergic rhinitis', 'Doctor diagnosed hayfever or allergic rhinitis',
                    'Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor: Hayfever, allergic rhinitis or eczema',
                    'Leg pain on walking','Pain type(s) experienced in last month: Knee pain',
                    'Pain type(s) experienced in last month: Hip pain','Diagnoses - main ICD10: M54 Dorsalgia', 'Fractured/broken bones in last 5 years'
                    'Non-cancer illness code, self-reported: depression', 'Mood swings', 'Miserableness', 'Irritability',
                    'Sensitivity / hurt feelings', 'Fed-up feelings','Nervous feelings', 'Worrier / anxious feelings',
                     "Tense / 'highly strung'", 'Worry too long after embarrassment', "Suffer from 'nerves'", 'Neuroticism score',
                     'Loneliness, isolation', 'Guilty feelings', 'Risk taking', 'Ever felt worried, tense, or anxious for most of a month or longer',
                     'Ever had prolonged loss of interest in normal activities', 'Ever had prolonged feelings of sadness or depression',
                     'General happiness', 'General happiness with own health', 'Belief that own life is meaningful', 'Ever thought that life not worth living',
                     'Felt hated by family member as a child', 'Physically abused by family as a child', 'Felt loved as a child',
                     'Felt very upset when reminded of stressful experience in past month', 'Ever sought or received professional help for mental distress',
                     'Frequency of depressed mood in last 2 weeks', 'Ever suffered mental distress preventing usual activities', 'Ever had period extreme irritability',
                     'Trouble falling or staying asleep, or sleeping too much', 'Recent feelings of tiredness or low energy',
                     'Substances taken for depression: Medication prescribed to you (for at least two weeks)', 'Activities undertaken to treat depression: Talking therapies, such as psychotherapy, counselling, group therapy or CBT',
                     'Frequency of unenthusiasm / disinterest in last 2 weeks', 'Frequency of tenseness / restlessness in last 2 weeks',
                     'Frequency of tiredness / lethargy in last 2 weeks', 'Seen doctor (GP) for nerves, anxiety, tension or depression',
                     'Seen a psychiatrist for nerves, anxiety, tension or depression', 'Happiness', 'Health satisfaction', 'Family relationship satisfaction',
                     'Friendships satisfaction', 'Financial situation satisfaction', 'Ever depressed for a whole week',
                     'Pain type(s) experienced in last month: Headache',
                     'Non-cancer illness code, self-reported: hypothyroidism/myxoedema',
                     'Disorders of gallbladder, biliary tract and pancreas']
tenk_prs_order = [tenk_prs_short_names_dict.get(x,x) for x in tenk_prs_order]



ukb_disease_short_names_dict = {"disorders of lipoprotein metabolism and other lipidaemias" : "lipoprotein disorder or lipidaemia",
                                'Other specified conditions associated with the spine (intervertebral disc displacement)' : 'Spine conditions - intervertebral disc displacement',
                                "stroke, not specified as haemorrhage or infarction" : "stroke, not specified",
                                "osteoporosis with pathological fracture" : "osteoporosis"
                               }

ukb_disease_order = ["essential (primary) hypertension","obesity",
                     "disorders of lipoprotein metabolism and other lipidaemias","atherosclerosis",
                     "aortic aneurysm and dissection","other peripheral vascular diseases",
                     "other diseases of liver","fibrosis and cirrhosis of liver",
                     "other inflammatory liver diseases","non-insulin-dependent diabetes mellitus",
                     "unspecified diabetes mellitus","insulin-dependent diabetes mellitus","hyperthyroidism",
                     "other non-toxic goitre","other hypothyroidism","thyroiditis",
                     "osteoporosis with pathological fracture","nonrheumatic aortic valve disorders",
                     "nonrheumatic mitral valve disorders","atrioventricular and left bundle-branch block",
                     "atrial fibrillation and flutter","cardiomyopathy","chronic ischaemic heart disease",
                     "heart failure","other acute ischaemic heart diseases","acute myocardial infarction",
                     "other chronic obstructive pulmonary disease","other interstitial pulmonary diseases",
                     "pulmonary oedema","pulmonary embolism","asthma","chronic sinusitis",
                     "pneumonia, organism unspecified","migraine","parkinson's disease","alzheimer's disease",
                     "vascular dementia","facial nerve disorders",
                     "stroke, not specified as haemorrhage or infarction","cerebral infarction",
                     "other anxiety disorders","recurrent depressive disorder","ulcerative colitis","dyspepsia",
                     "other rheumatoid arthritis","gout","chronic renal failure",'cholelithiasis',
                     "vitamin d deficiency","acute pancreatitis","cellulitis"]
ukb_disease_order = [ukb_disease_short_names_dict.get(x,x) for x in ukb_disease_order]


prs_groups = ([    ['Pulse rate, automated reading', 'Non-cancer illness code, self-reported: hypertension',
                    'Non-cancer illness code, self-reported: heart attack/myocardial infarction',
                    'Illnesses of father: Heart disease', 'Illnesses of father: High blood pressure', 'Illnesses of mother: High blood pressure',
                    'Illnesses of siblings: Heart disease', 'Illnesses of siblings: High blood pressure','Chest pain or discomfort',
                    'Age high blood pressure diagnosed', 'Diastolic blood pressure, automated reading', 'Systolic blood pressure, automated reading',
                    'Pulse rate', 'Pulse wave reflection index', 'Vascular/heart problems diagnosed by doctor: Heart attack',
                     'Vascular/heart problems diagnosed by doctor: High blood pressure',
                    'Major coronary heart disease event', 'Ischaemic heart disease, wide definition',
                    'Diseases of the circulatory system','Coronary atherosclerosis',
                    'Non-cancer illness code, self-reported: angina', 'Vascular/heart problems diagnosed by doctor: Angina',
                    'Diagnoses - main ICD10: I25 Chronic ischaemic heart disease',],
                   ['Coronary atherosclerosis', 'Non-cancer illness code, self-reported: angina', 'Vascular/heart problems diagnosed by doctor: Angina',
                    'Diagnoses - main ICD10: I25 Chronic ischaemic heart disease','Non-cancer illness code, self-reported: diabetes', 'Non-cancer illness code, self-reported: high cholesterol',
                    'Illnesses of father: Diabetes', 'Illnesses of mother: Diabetes', 'Illnesses of siblings: Diabetes',
                    'Diabetes diagnosed by doctor',],
                   ['Non-cancer illness code, self-reported: diabetes', 'Non-cancer illness code, self-reported: high cholesterol',
                    'Illnesses of father: Diabetes', 'Illnesses of mother: Diabetes', 'Illnesses of siblings: Diabetes',
                    'Diabetes diagnosed by doctor','Diagnoses - main ICD10: G56 Mononeuropathies of upper limb',],
                   ['Wheeze or whistling in the chest in last year', 'Non-cancer illness code, self-reported: asthma',
                    'Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor: Asthma',],
                   ['Wheeze or whistling in the chest in last year', 'Non-cancer illness code, self-reported: asthma',
                    'Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor: Asthma',  
                    'Non-cancer illness code, self-reported: hayfever/allergic rhinitis', 'Doctor diagnosed hayfever or allergic rhinitis',
                    'Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor: Hayfever, allergic rhinitis or eczema',],
                   ['Leg pain on walking',],
                   ['Pain type(s) experienced in last month: Knee pain','Pain type(s) experienced in last month: Hip pain',
                    'Leg pain on walking', 'Diagnoses - main ICD10: M54 Dorsalgia',],
                   ['Pain type(s) experienced in last month: Knee pain','Pain type(s) experienced in last month: Hip pain',
                    'Leg pain on walking', 'Diagnoses - main ICD10: M54 Dorsalgia', 'Fractured/broken bones in last 5 years'],
                   ['Non-cancer illness code, self-reported: depression', 'Mood swings', 'Miserableness', 'Irritability',
                    'Sensitivity / hurt feelings', 'Fed-up feelings','Nervous feelings', 'Worrier / anxious feelings',
                     "Tense / 'highly strung'", 'Worry too long after embarrassment', "Suffer from 'nerves'", 'Neuroticism score',
                     'Loneliness, isolation', 'Guilty feelings', 'Risk taking', 'Ever felt worried, tense, or anxious for most of a month or longer',
                     'Ever had prolonged loss of interest in normal activities', 'Ever had prolonged feelings of sadness or depression',
                     'General happiness', 'General happiness with own health', 'Belief that own life is meaningful', 'Ever thought that life not worth living',
                     'Felt hated by family member as a child', 'Physically abused by family as a child', 'Felt loved as a child',
                     'Felt very upset when reminded of stressful experience in past month', 'Ever sought or received professional help for mental distress',
                     'Frequency of depressed mood in last 2 weeks', 'Ever suffered mental distress preventing usual activities', 'Ever had period extreme irritability',
                     'Trouble falling or staying asleep, or sleeping too much', 'Recent feelings of tiredness or low energy',
                     'Substances taken for depression: Medication prescribed to you (for at least two weeks)', 'Activities undertaken to treat depression: Talking therapies, such as psychotherapy, counselling, group therapy or CBT',
                     'Frequency of unenthusiasm / disinterest in last 2 weeks', 'Frequency of tenseness / restlessness in last 2 weeks',
                     'Frequency of tiredness / lethargy in last 2 weeks', 'Seen doctor (GP) for nerves, anxiety, tension or depression',
                     'Seen a psychiatrist for nerves, anxiety, tension or depression', 'Happiness', 'Health satisfaction', 'Family relationship satisfaction',
                     'Friendships satisfaction', 'Financial situation satisfaction', 'Ever depressed for a whole week',],              
                    ['Pain type(s) experienced in last month: Headache',],
                    ['Non-cancer illness code, self-reported: hypothyroidism/myxoedema',],
                    ['Disorders of gallbladder, biliary tract and pancreas']
              ],
              [    ['essential (primary) hypertension', 'nonrheumatic aortic valve disorders', 'nonrheumatic mitral valve disorders',
                    'atrioventricular and left bundle-branch block', 'atrial fibrillation and flutter', 'cardiomyopathy',
                    'chronic ischaemic heart disease', 'heart failure', 'other acute ischaemic heart diseases', 'acute myocardial infarction',
                    'obesity', 'pulmonary oedema', 'other peripheral vascular diseases',],
                   ['disorders of lipoprotein metabolism and other lipidaemias', 'atherosclerosis',
                    'non-insulin-dependent diabetes mellitus', 'unspecified diabetes mellitus',
                    'insulin-dependent diabetes mellitus', 'obesity', 'chronic renal failure', 'essential (primary) hypertension',],
                   ['non-insulin-dependent diabetes mellitus', 'unspecified diabetes mellitus',
                    'insulin-dependent diabetes mellitus',],
                   ['other chronic obstructive pulmonary disease',],
                   ['asthma','chronic_sinusitis'],
                   ['gout'],
                   ['other rheumatoid arthritis',],
                   ['osteoporosis with pathological fracture'],
                   ['recurrent depressive disorder', 'other anxiety disorders'],              
                   ['migraine',],
                   ['other hypothyroidism',],
                   ['cholelithiasis']
              ])

prs_groups = ([[tenk_prs_short_names_dict.get(x,x) for x in y] for y in prs_groups[0]] , 
              [[ukb_disease_short_names_dict.get(x,x) for x in y] for y in prs_groups[1]])


ukb_shared_diseases = ["essential (primary) hypertension","obesity","lipoprotein disorder or lipidaemia",
                       #"atherosclerosis",
                       "other diseases of liver","fibrosis and cirrhosis of liver",
                       "non-insulin-dependent diabetes mellitus",#"unspecified diabetes mellitus",
                       "other hypothyroidism","osteoporosis with pathological fracture",
                       #"nonrheumatic aortic valve disorders","nonrheumatic mitral valve disorders",
                       "asthma",
                       "chronic sinusitis","migraine","other anxiety disorders",
                       "recurrent depressive disorder","dyspepsia", "cholelithiasis"]

tenk_disease_order = ["Essential hypertension","Obesity","Hyperlipoproteinaemia  ",
                      "Non-alcoholic fatty liver disease","Obstructive sleep apnoea",
                      "Intermediate hyperglycaemia ","Thyrotoxicosis","hypothyroidism",
                      "Low bone mass disorders","Osteoarthritis",'Fracture of unspecified body region',
                      #"Heart valve diseases unspecified",
                      "Asthma","Allergic rhinitis ","Chronic rhinosinusitis",
                      "Migraine","Headache disorders","Anxiety or fear-related disorders",
                      "Depressive disorders","Insomnia disorders","Attention deficit hyperactivity disorder",
                      "Chronic widespread pain","Anal fissure","Haemorrhoids",
                      "Oral aphthae or aphtha-like ulceration","Ulcer of stomach or duodenum",
                      "Irritable bowel syndrome","Helicobacter pylori induced gastritis","Spinal pain",
                      "Other specified conditions associated with the spine (intervertebral disc displacement)",
                      "Polycystic ovary syndrome","Endometriosis","Female infertility",
                      "Hyperplasia of prostate","Urinary tract infection, site and agent not specified",
                      "Urolithiasis","Cholelithiasis","Basal cell carcinoma of skin","Atopic eczema",
                      "Psoriasis","Acne","Cataract","Episodic vestibular syndrome",
                      "Acquired hearing impairment","Anaemias or other erythrocyte disorders",
                      "Iron Deficiency Anemia","Vitamin B12 deficiency"]

tenk_shared_diseases = ["Essential hypertension","Obesity","Hyperlipoproteinaemia  ",
                        "Non-alcoholic fatty liver disease","Intermediate hyperglycaemia ","hypothyroidism",
                        "Low bone mass disorders",#'Fracture of unspecified body region',
                        #"Heart valve diseases unspecified",
                        "Asthma","Chronic rhinosinusitis","Migraine",
                        "Headache disorders","Anxiety or fear-related disorders","Depressive disorders",
                        "Ulcer of stomach or duodenum","Cholelithiasis"]

ukb_disease_to_tenk_condition_dict = {"essential (primary) hypertension" : "Essential hypertension",
                                      "obesity" : "Obesity",
                                      "lipoprotein disorder or lipidaemia" : "Hyperlipoproteinaemia  ",
                                      "other diseases of liver" : "Non-alcoholic fatty liver disease",
                                      "non-insulin-dependent diabetes mellitus" : "Intermediate hyperglycaemia ",
                                      "other hypothyroidism" : "hypothyroidism",
                                      "osteoporosis with pathological fracture" : "Low bone mass disorders",
                                      "asthma" : "Asthma",
                                      "chronic sinusitis" : "Chronic rhinosinusitis",
                                      "migraine" : ["Migraine","Headache disorders"],
                                      "other anxiety disorders" : "Anxiety or fear-related disorders",
                                      "recurrent depressive disorder" : "Depressive disorders",
                                      "dyspepsia" : "Ulcer of stomach or duodenum",
                                      "cholelithiasis" : "Cholelithiasis"
                                     }

tenk_condition_to_ukb_disease_dict = {"Essential hypertension" : "essential (primary) hypertension",
                                      "Obesity" : "obesity",
                                      "Hyperlipoproteinaemia  " : "lipoprotein disorder or lipidaemia",
                                      "Non-alcoholic fatty liver disease" : "other diseases of liver",
                                      "Intermediate hyperglycaemia " : "non-insulin-dependent diabetes mellitus",
                                      "hypothyroidism" : "other hypothyroidism",
                                      "Low bone mass disorders" : "osteoporosis with pathological fracture",
                                      "Asthma" : "asthma",
                                      "Chronic rhinosinusitis" : "chronic sinusitis",
                                      "Migraine" : "migraine",
                                      "Headache disorders" : "migraine",
                                      "Anxiety or fear-related disorders" : "other anxiety disorders",
                                      "Depressive disorders" : "recurrent depressive disorder",
                                      "Ulcer of stomach or duodenum" : 'dyspepsia',
                                      "Cholelithiasis" : "cholelithiasis"
                                     }


drop_outcomes = [130736,132054,132072,132152,131620,131612,131636] # These were not run or are gender specific

drop_cancer_outcomes = ["C61","C56","C20"]

m_conditions = ["Hyperplasia of prostate"]
f_conditions = ['Polycystic ovary syndrome',"Endometriosis","Female infertility"]#,'haemolytic anaemia due to G6PD deficiency']

loaders = ['retina', 'microbiome', 'sleep',  'ecg', 'cgm', 'dexa', # 'bloodtests', 'bodymeasures', \ 
                'carotid','abi', 'diet', 'metabolomics','liver','age_gender_bmi'] # , 'diet'

# features that basically overlap with an input feature
drop_features_dict = {'sleep' : ['MeanPRSleep'],
                      'ecg' : ['hr_bpm'],
                      'abi' : ['l_brachial_pressure', 'r_brachial_pressure'], # 'l_ankle_pressure', 'r_ankle_pressure'
                      'cgm' : ['eA1C'],
                      'dexa' : ['body_comp_trunk_region_percent_fat', 'body_comp_total_region_percent_fat'],
                      'diet' : [],
                      'cgm_main' : ['eA1C']
                     }

outcomes_df = pd.read_csv(os.path.join(os.path.dirname(outdir),'UKBB_data','outcomes.csv'))

outcomes_dict = dict(zip(outcomes_df['UKBB Field ID'].values,outcomes_df['UKBB Description'].values))
for o in outcomes_dict:
    outcomes_dict[o] = '('.join(outcomes_dict[o][:-1].split('(')[1:])
for k in drop_outcomes:
    outcomes_dict.pop(k,None)
    
outcomes_inv_dict = {ukb_disease_short_names_dict.get(v,v):k for k,v in outcomes_dict.items()}
    


def get_odds_by_gender(ukb_preds, tenk_condition, age_gender_bmi, res_df,
                       ukb_disease, tenk_cond, total_tests):
    # Bonferroni-adjusted z-score
    alpha = 0.05
    z_bonf = norm.ppf(1 - alpha / (2 * total_tests))

    # Merge inputs
    merged = pd.concat([tenk_condition, ukb_preds * -1, age_gender_bmi], axis=1).dropna(
        subset=ukb_preds.columns.to_list()).dropna(subset=age_gender_bmi.columns.to_list())
    m_merged = merged[merged['gender'] == 1].drop(columns='gender')
    f_merged = merged[merged['gender'] == 0].drop(columns='gender')

    # Initialize placeholders
    m_or, m_pval, m_ci = np.nan, np.nan, (np.nan, np.nan)
    f_or, f_pval, f_ci = np.nan, np.nan, (np.nan, np.nan)

    # Male model
    if tenk_cond not in f_conditions:
        m_y = m_merged.iloc[:, 0]
        m_X = sm.add_constant(m_merged.iloc[:, 1:])
        m_model = sm.Logit(m_y, m_X, missing='drop').fit(disp=0)

        m_beta = m_model.params['predictions']
        m_se = m_model.bse['predictions']
        m_or = np.exp(m_beta)
        m_ci = (np.exp(m_beta - z_bonf * m_se), np.exp(m_beta + z_bonf * m_se))
        m_pval = m_model.pvalues['predictions']

    # Female model
    if tenk_cond not in m_conditions:
        f_y = f_merged.iloc[:, 0]
        f_X = sm.add_constant(f_merged.iloc[:, 1:])
        f_model = sm.Logit(f_y, f_X, missing='drop').fit(disp=0)

        f_beta = f_model.params['predictions']
        f_se = f_model.bse['predictions']
        f_or = np.exp(f_beta)
        f_ci = (np.exp(f_beta - z_bonf * f_se), np.exp(f_beta + z_bonf * f_se))
        f_pval = f_model.pvalues['predictions']

    # Add result to dataframe
    new_row = {
        "ukbb disease": ukb_disease,
        "tenk cond": tenk_cond,
        "m_or": m_or,
        "m_ci_lower": m_ci[0],
        "m_ci_upper": m_ci[1],
        "m_pval": m_pval,
        "f_or": f_or,
        "f_ci_lower": f_ci[0],
        "f_ci_upper": f_ci[1],
        "f_pval": f_pval
    }

    res_df = res_df.append(new_row, ignore_index=True)
    return res_df

def plot_ors(df, alpha = 0.05, prefix=None):

    df["pval"]*=(2*len(df)) # bonferroni correct for both male and female
    
   # df["sig"] = df["pval"]<alpha
    print(df[df["pval"]>=alpha])
    df = df[df["pval"]<alpha]
    
    sorted_df = df.sort_values(by="or",ascending=True)
    print(sorted_df)
    print(sorted_df['or'].mean())
    
    
    
    plt.clf()
    
    fig, ax1 = plt.subplots(figsize=(16, 7))
    ax2 = ax1.twinx()
    ax1.set_xlabel('Odds ratio', fontsize=24)

    ax1.axvline(x=1, color='black', linestyle='--', linewidth=1)

    sns.scatterplot(data=sorted_df,y="tenk cond",x="or", s=80, ax=ax1)#,hue="sig",palette={True:"slategray",False:"lightgray"})
    ax1.errorbar(
        sorted_df["or"],
        range(len(sorted_df)),
        xerr=[
            sorted_df["or"] - sorted_df["ci_lower"],
            sorted_df["ci_upper"] - sorted_df["or"]
        ],
        fmt='none',
        color='black',
        capsize=3,
        elinewidth=1
    )

    ax1.set(ylabel=None)
    ax2.set(ylabel=None)
    ax1.tick_params(axis='y', labelsize=19)
    ax1.tick_params(axis='x', labelsize=19)
    ax2.tick_params(axis='y', labelsize=19)

    ax2.set_yticks(range(len(sorted_df)))
    ax2.set_yticklabels(sorted_df['ukbb disease'])
    ax1.set_yticks(range(len(sorted_df)))
    ax1.set_yticklabels(sorted_df['tenk cond'])
    
    ax1.set_ylim(-0.5, len(sorted_df) - 0.5)
    ax2.set_ylim(-0.5, len(sorted_df) - 0.5)
    
    

    ax1.annotate(
        'Condition',
        xy=(-0.1, 1.01),
        xycoords='axes fraction',
        ha='right',
        fontsize=22,
    )
    
    ax2.annotate(
        'Pseudo-outcome',
        xy=(1.1, 1.01),
        xycoords='axes fraction',
        ha='left',
        fontsize=22,
    )
    
    if prefix is not None:
        ax1.set_title(prefix,fontsize=24)


#    for index, row in sorted_df.iterrows():
 #       ax1.text(row['or'], -0.04, row['ukbb disease'], va='center', ha='center', fontsize=14, rotation='vertical')
    
    plt.tight_layout()
    
    fname = "condition_ors.png"
    if prefix is not None:
        fname = f"condition_ors_{prefix}_new_2025_2.png"
    plt.savefig(os.path.join('/home/davidpel/proj_data',fname),dpi=300)
    plt.clf()
    
    
    
def plot_ors_by_gender(df):
    
    m_df = df[["ukbb disease", "tenk cond", "m_or", "m_pval", "m_ci_lower", "m_ci_upper"]]
    
    plot_ors(m_df.rename(columns = {"m_or":"or", "m_pval":"pval", "m_ci_lower":"ci_lower","m_ci_upper":"ci_upper"}),prefix='Male')
    
    f_df = df[["ukbb disease", "tenk cond", "f_or", "f_pval", "f_ci_lower", "f_ci_upper"]]
    
    plot_ors(f_df.rename(columns = {"f_or":"or", "f_pval":"pval", "f_ci_lower": "ci_lower", "f_ci_upper":"ci_upper"}),prefix='Female')



conditions = load_medical()
age_gender_bmi = load_age_gender_bmi()

results_cols = ["ukbb disease", "tenk cond", "m_or", "m_pval", "m_ci_lower", "m_ci_upper", "f_or", "f_pval", "f_ci_lower", "f_ci_upper"]


df = pd.DataFrame(columns = results_cols)

num_tests = 0
for k,v in ukb_disease_to_tenk_condition_dict.items():
        
    if isinstance(v, list):
        num_tests += len(v)
        print(v,len(v))
    else:
        num_tests += 1
        print(v)
num_tests*=2 # male and female
print(num_tests)
for disease in ukb_disease_to_tenk_condition_dict:
    
    disease_id = outcomes_inv_dict[ukb_disease_short_names_dict.get(disease,disease)]
    preds = load_outcome(disease_id)
    preds /= 365
    condition = ukb_disease_to_tenk_condition_dict[disease]
    condition_cols = conditions[condition]
    
    if isinstance(condition_cols,pd.Series):
        df = get_odds_by_gender(preds, condition_cols, age_gender_bmi, df, disease, condition, num_tests)
    else:
        for cond in condition_cols:
            df = get_odds_by_gender(preds, condition_cols[cond], age_gender_bmi, df, disease, cond, num_tests)
            
plot_ors_by_gender(df)


