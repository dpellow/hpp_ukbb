

from LabData.DataLoaders.LifeStyleLoader import LifeStyleLoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabData.DataLoaders.GutMBLoader import GutMBLoader
from LabData.DataAnalyses.UKBB_10k.process_predictions import load_medical, load_outcome, load_age_gender_bmi, load_microbiome

# from scipy.stats import mannwhitneyu
# import statsmodels.stats.multitest as smm
import statsmodels.api as sm

import numpy as np
import pandas as pd
import pingouin as pg

bm_md = BodyMeasuresLoader().get_data(study_ids=['10K'], research_stage=['baseline']).df_metadata
reg_ids = bm_md.index.get_level_values(0).tolist() # ids of everyone who has come in for a visit

data = LifeStyleLoader().get_data(study_ids=['10K'], research_stage=['baseline'],groupby_reg='first',reg_ids=reg_ids).df

data = data[['tobacco_past_how_often','smoke_tobacco_now']]
data = data.reset_index(level=[1]).drop(['Date'], axis=1)
def add_age_gender_bmi(df):
    age_gender_bmi=load_age_gender_bmi()
    df = df.join(age_gender_bmi)
    return df

data = add_age_gender_bmi(data)

# data['tobacco_past_how_often'] = data['tobacco_past_how_often'].fillna(0)
# data['smoke_tobacco_now'] = data['smoke_tobacco_now'].fillna(0)
data.loc[data['tobacco_past_how_often']<0, 'tobacco_past_how_often'] = np.nan
data.loc[data['smoke_tobacco_now']<0, 'smoke_tobacco_now'] = np.nan


mb = load_microbiome()

preds = load_outcome(131492)

print('loaded_microbiome')

merged = preds.join(data, how='inner')
merged = merged.join(mb, how='inner')

target = "predictions"
covars = ["age","bmi","tobacco_past_how_often","smoke_tobacco_now"]

# All other columns except target + covars
others = [col for col in merged.columns if col not in [target,'gender'] + covars]

m_merged = merged[merged['gender']==1].drop(columns='gender')
f_merged = merged[merged['gender']==0].drop(columns='gender')
# Compute partial correlations
results = []
for col in others:
    print(col)
    pcorr = pg.partial_corr(data=m_merged,
                            x=target,
                            y=col,
                            covar=covars,
                            method='spearman')
    pcorr['var'] = col
    results.append(pcorr)

results_df = pd.concat(results, ignore_index=True)

# Keep the key columns
results_df = results_df[['var', 'r', 'p-val', 'CI95%']]
print(results_df)
results_df.to_csv('/home/davidpel/proj_data/male_smoking_microbiome_association_partial_corr.csv')

results = []
for col in others:
    print(col)
    pcorr = pg.partial_corr(data=f_merged,
                            x=target,
                            y=col,
                            covar=covars,
                            method='spearman')
    pcorr['var'] = col
    results.append(pcorr)

results_df = pd.concat(results, ignore_index=True)

# Keep the key columns
results_df = results_df[['var', 'r', 'p-val', 'CI95%']]
print(results_df)
results_df.to_csv('/home/davidpel/proj_data/female_smoking_microbiome_association_partial_corr.csv')



# smokers = ((data['tobacco_past_how_often'] > 1) | (data['smoke_tobacco_now'] > 0)).astype(int)
# ids = smokers.index.intersection(mb.index)
# smokers = smokers.loc[ids]
# mb = mb.loc[ids]

# mb_smoke = mb[smokers[0]==1]
# mb_nonsmoke = mb[smokers[0]==0]

# p_values = []
# for microbe in mb.columns:
#     stat, p = mannwhitneyu(mb_smoke[microbe], mb_nonsmoke[microbe], alternative='two-sided')
#     p_values.append(p)
# adjusted_p_values = smm.multipletests(p_values, method='bonferroni')[1]
# results = pd.DataFrame({
#     'microbe': mb.columns,
#     'p_value': p_values,
#     'adjusted_p_value': adjusted_p_values,
#     'smokers_mean':mb_smoke.mean(),
#     'nonsmokers_mean': mb_nonsmoke.mean()
# })

# results['enriched_in'] = results.apply(
#     lambda row: 'smokers' if row['smokers_mean'] > row['nonsmokers_mean'] else 'nonsmokers',
#     axis=1
# )

# significant_results = results[results['adjusted_p_value'] < 0.05]
# significant_results = significant_results.sort_values(by='adjusted_p_value')

# significant_results.to_csv('/home/davidpel/proj_data/smoking_microbiome_association.csv', index=False)