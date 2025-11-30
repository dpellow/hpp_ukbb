from LabData.DataAnalyses.UKBB_10k.UKBB_survival import *
from LabData.DataAnalyses.UKBB_10k.process_predictions import *

from LabData.DataLoaders.PRSLoader import PRSLoader
import pandas as pd
import numpy as np
from LabData.DataLoaders.SubjectLoader import SubjectLoader
import os

from LabQueue.qp import fakeqp as qp # qp #

from LabUtils.addloglevels import sethandlers

model = "regularized_cox"
topdir = os.path.join(OUTPATH,model)


#From Zach
def getsigunique(cached=True, dir="/net/mraid08/export/jafar/UKBioBank/davidpel/UKBB_data/",
                 fileName="getsigunique_cache.csv"):
    # Load all the PRS from 10K
    if cached:  ##Much faster
        cached_list = list(pd.read_csv(dir + fileName).iloc[:,
                           1])  ##column 0 holds an aribitrary index, column 1 has the list we care about
        return cached_list
    mydata = PRSLoader().get_data(study_ids=['10K'])#.df.columns
    metadata_table = pd.read_excel(
        '/net/mraid20/export/genie/10K/genetics/PRSice/SummaryStatistics/Nealelab/v3/UKBB_GWAS_Imputed_v3-File_Manifest_Release_20180731.xlsx',
      #  '/net/mraid08/export/jafar/UKBioBank/davidpel/UKBB_data/UKBB_GWAS_Imputed_v3-File_Manifest_Release_20180731.xlsx',
        sheet_name='Manifest 201807', engine='openpyxl')
    metadata_table = metadata_table.loc[metadata_table['Phenotype Code'].notnull()]
    metadata_table = metadata_table.loc[metadata_table['Sex'].eq('both_sexes')]
    metadata_table.set_index('Phenotype Code', inplace=True)
    # metadata of the PRS that are related to genetics
    high_sig = mydata.df_columns_metadata[mydata.df_columns_metadata['h2_h2_sig'].eq('z7')]
    high_sig = high_sig.merge(metadata_table, left_index=True, right_index=True, how='inner')
    # dividing the PRS into groups according to their descriptions
    prefix_group = {'Illnesses of siblings': 'Family history',
                    'Illnesses of mother': 'Family history',
                    'Illnesses of father': 'Family history',
                    'Non-cancer illness code, self-reported': 'Medical conditions',
                    'Treatment/medication code': 'Medication',
                    'Diagnoses - main ICD10': 'Summary Diagnoses',
                    'Pain type(s) experienced in last month': 'Pain',
                    'Leisure/social activities': 'Social Support',
                    'Types of physical activity in last 4 weeks': 'Physical activity',
                    'Qualifications': 'Education',
                    'Medication for pain relief, constipation, heartburn': 'Medication',
                    'Types of transport used (excluding work)': 'Physical activity',
                    'Reason for glasses/contact lenses': 'Eyesight',
                    'How are people in household related to participant': 'Household',
                    'Mouth/teeth dental problems': 'Mouth',
                    'Destinations on discharge from hospital (recoded)': 'Summary Administration',
                    'Vascular/heart problems diagnosed by doctor': 'Medical conditions',
                    'Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones': 'Medication',
                    'Smoking status': 'Smoking',
                    'Mineral and other dietary supplements': 'Medication',
                    'Medication for cholesterol, blood pressure or diabetes': 'Medication',
                    'Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor': 'Medical conditions',
                    'Hair/balding pattern': 'Male-specific factors',
                    'Illness, injury, bereavement, stress in last 2 years': 'Mental health',
                    'Spread type': 'Diet',
                    'Major dietary changes in the last 5 years': 'Diet',
                    'Never eat eggs, dairy, wheat, sugar': 'Diet',
                    'Milk type used': 'Diet',
                    'Tobacco smoking': 'Smoking',
                    'Bread type': 'Diet',
                    'Cereal type': 'Diet',
                    'Coffee type': 'Diet',
                    'Attendance/disability/mobility allowance': 'Other sociodemographic factors',
                    'Hearing difficulty/problems': 'Hearing', 'Vitamin and mineral supplements': 'Medication',
                    'Body mass index (BMI)': 'Body size measures',
                    'Weight': 'Body size measures',
                    'Own or rent accommodation lived in': 'Household',
                    'High light scatter reticulocyte percentage': 'Blood count',
                    'IGF-1 (quantile)': 'Blood biochemistry',
                    'Average total household income before tax': 'Household',
                    'Time spend outdoors in summer': 'Sun exposure',
                    'Time spent outdoors in winter': 'Sun exposure',
                    'Time spent watching television (TV)': 'Physical activity',
                    'Time spent using computer': 'Physical activity',
                    'Time spent driving': 'Physical activity',
                    'Duration of light DIY': 'Physical activity',
                    'Pulse rate, automated reading': 'Blood pressure',
                    'Cooked vegetable intake': 'Diet',
                    'Salad / raw vegetable intake': 'Diet',
                    'Fresh fruit intake': 'Diet',
                    'Dried fruit intake': 'Diet',
                    'Oily fish intake': 'Diet',
                    'Non-oily fish intake': 'Diet',
                    'Processed meat intake': 'Diet',
                    'Poultry intake': 'Diet',
                    'Beef intake': 'Diet',
                    'Pork intake': 'Diet',
                    'Cheese intake': 'Diet',
                    'Bread intake': 'Diet',
                    'Cereal intake': 'Diet',
                    'Lamb/mutton intake': 'Diet',
                    'Salt added to food': 'Diet',
                    'Tea intake': 'Diet',
                    'Coffee intake': 'Diet',
                    'Water intake': 'Diet',
                    'Variation in diet': 'Diet',
                    'Current tobacco smoking': 'Smoking',
                    'Past tobacco smoking': 'Smoking',
                    'Exposure to tobacco smoke outside home': 'Smoking',
                    'Average weekly red wine intake': 'Alcohol',
                    'Average weekly champagne plus white wine intake': 'Alcohol',
                    'Average weekly spirits intake': 'Alcohol',
                    'Average weekly beer plus cider intake': 'Alcohol',
                    'Alcohol intake frequency.': 'Alcohol',
                    'Frequency of friend/family visits': 'Social support',
                    'Length of mobile phone use': 'Electronic device use',
                    'Drive faster than motorway speed limit': 'Physical activity',
                    'Weekly usage of mobile phone in last 3 months': 'Electronic device use',
                    'Invitation to complete online 24-hour recall dietary questionnaire, acceptance': 'Diet by 24-hour recall',
                    'Sleep duration': 'Sleep',
                    'Getting up in morning': 'Sleep',
                    'Morning/evening person (chronotype)': 'Sleep',
                    'Nap during day': 'Sleep',
                    'Sleeplessness / insomnia': 'Sleep',
                    'Snoring': 'Sleep',
                    'Daytime dozing / sleeping (narcolepsy)': 'Sleep',
                    'Number of self-reported non-cancer illnesses': 'Medical conditions',
                    'Number of operations, self-reported': 'Operations',
                    'Number of treatments/medications taken': 'Medication',
                    'Hot drink temperature': 'Diet',
                    'Alcohol usually taken with meals': 'Alcohol',
                    'Alcohol intake versus 10 years previously': 'Alcohol',
                    'Comparative body size at age 10': 'Early life factors',
                    'Comparative height size at age 10': 'Early life factors',
                    'Facial ageing': 'Sun exposure',
                    'Maternal smoking around birth': 'Early life factors',
                    'Father\'s age at death': 'Family history',
                    'Townsend deprivation index at recruitment': 'Baseline characteristics',
                    'Mood swings': 'Mental health',
                    'Miserableness': 'Mental health',
                    'Irritability': 'Mental health',
                    'Sensitivity / hurt feelings': 'Mental health',
                    'Fed-up feelings': 'Mental health',
                    'Nervous feelings': 'Mental health',
                    'Worrier / anxious feelings': 'Mental health',
                    "Tense / 'highly strung'": 'Mental health',
                    'Worry too long after embarrassment': 'Mental health',
                    'Sitting height': 'Body size measures',
                    'Fluid intelligence score': 'Fluid intelligence / reasoning',
                    'Birth weight': 'Early life factors',
                    'Mean time to correctly identify matches': 'Reaction time',
                    "Suffer from 'nerves'": 'Mental health',
                    'Alcohol drinker status': 'Alcohol', 'Neuroticism score': 'Mental health',
                    'Number of fluid intelligence questions attempted within time limit': 'Fluid intelligence / reasoning',
                    'Forced expiratory volume in 1-second (FEV1), Best measure': 'Spirometry',
                    'Forced vital capacity (FVC), Best measure': 'Spirometry',
                    'Forced expiratory volume in 1-second (FEV1), predicted': 'Spirometry',
                    'Forced expiratory volume in 1-second (FEV1), predicted percentage': 'Spirometry',
                    'Ever smoked': 'Smoking', 'Loneliness, isolation': 'Mental health',
                    'Guilty feelings': 'Mental health',
                    'Risk taking': 'Mental health', 'Frequency of drinking alcohol': 'Alcohol',
                    'Frequency of consuming six or more units of alcohol': 'Alcohol',
                    'Ever felt worried, tense, or anxious for most of a month or longer': 'Mental health',
                    'Ever had prolonged loss of interest in normal activities': 'Mental health',
                    'General happiness': 'Mental health',
                    'Ever had prolonged feelings of sadness or depression': 'Mental health',
                    'Ever taken cannabis': 'Cannabis use',
                    'General happiness with own health': 'Mental health',
                    'Belief that own life is meaningful': 'Mental health',
                    'Ever thought that life not worth living': 'Mental health',
                    'Felt hated by family member as a child': 'Mental health',
                    'Physically abused by family as a child': 'Mental health',
                    'Felt very upset when reminded of stressful experience in past month': 'Mental health',
                    'Felt loved as a child': 'Mental health',
                    'Frequency of depressed mood in last 2 weeks': 'Mental health',
                    'Ever sought or received professional help for mental distress': 'Mental health',
                    'Ever suffered mental distress preventing usual activities': 'Mental health',
                    'Ever had period extreme irritability': 'Mental health',
                    'Frequency of unenthusiasm / disinterest in last 2 weeks': 'Mental health',
                    'Frequency of tenseness / restlessness in last 2 weeks': 'Mental health',
                    'Frequency of tiredness / lethargy in last 2 weeks': 'Mental health',
                    'Seen doctor (GP) for nerves, anxiety, tension or depression': 'Mental health',
                    'Seen a psychiatrist for nerves, anxiety, tension or depression': 'Mental health',
                    'Trouble falling or staying asleep, or sleeping too much': 'Depression',
                    'Recent feelings of tiredness or low energy': 'Depression',
                    'Substances taken for depression': 'Depression',
                    'Activities undertaken to treat depression': 'Depression',
                    'Able to confide': 'Social support',
                    'Answered sexual history questions': 'Sexual factors',
                    'Age first had sexual intercourse': 'Sexual factors',
                    'Lifetime number of sexual partners': 'Sexual factors',
                    'Overall health rating': 'General health', 'Impedance of whole body': 'Body composition',
                    'Impedance of leg (right)': 'Body composition',
                    'Impedance of leg (left)': 'Body composition', 'Impedance of arm (right)': 'Body composition',
                    'Impedance of arm (left)': 'Body composition',
                    'Leg fat percentage (right)': 'Body composition', 'Leg fat mass (right)': 'Body composition',
                    'Leg fat-free mass (right)': 'Body composition', 'Leg predicted mass (right)': 'Body composition',
                    'Leg fat percentage (left)': 'Body composition', 'Leg fat mass (left)': 'Body composition',
                    'Leg fat-free mass (left)': 'Body composition',
                    'Leg predicted mass (left)': 'Body composition', 'Arm fat percentage (right)': 'Body composition',
                    'Arm fat mass (right)': 'Body composition',
                    'Arm fat-free mass (right)': 'Body composition',
                    'Arm predicted mass (right)': 'Body composition',
                    'Arm fat percentage (left)': 'Body composition',
                    'Arm fat mass (left)': 'Body composition',
                    'Arm fat-free mass (left)': 'Body composition',
                    'Arm predicted mass (left)': 'Body composition',
                    'Trunk fat percentage': 'Body composition',
                    'Trunk fat mass': 'Body composition',
                    'Trunk fat-free mass': 'Body composition',
                    'Trunk predicted mass': 'Body composition',
                    'Doctor diagnosed hayfever or allergic rhinitis': 'Medical information',
                    'Age started wearing glasses or contact lenses': 'Eyesight',
                    'Long-standing illness, disability or infirmity': 'General health',
                    'Plays computer games': 'Electronic device use',
                    'Year ended full time education': 'Work environment',
                    'Hearing difficulty/problems with background noise': 'Hearing',
                    'Home location - north co-ordinate (rounded)': 'Home locations',
                    'Body fat percentage': 'Body composition',
                    'Whole body fat mass': 'Body composition',
                    'Whole body fat-free mass': 'Body composition',
                    'Whole body water mass': 'Body composition',
                    'Falls in the last year': 'General health',
                    'Basal metabolic rate': 'Body composition',
                    'Wheeze or whistling in the chest in last year': 'Breathing',
                    'Chest pain or discomfort': 'Chest pain',
                    'Relative age of first facial hair': 'Male-specific factors',
                    'Relative age voice broke': 'Male-specific factors',
                    'Diabetes diagnosed by doctor': 'Medical conditions',
                    'Fractured/broken bones in last 5 years': 'Medical conditions',
                    'Other serious medical condition/disability diagnosed by doctor': 'Medical conditions',
                    'Taking other prescription medications': 'Medication',
                    'Light smokers, at least 100 smokes in lifetime': 'Smoking',
                    'Reason for reducing amount of alcohol drunk': 'Alcohol',
                    'Age when periods started (menarche)': 'Female-specific factors',
                    'Birth weight of first child': 'Female-specific factors',
                    'Age at first live birth': 'Female-specific factors',
                    'Age at last live birth': 'Female-specific factors',
                    'Age started oral contraceptive pill': 'Female-specific factors',
                    'Ever used hormone-replacement therapy (HRT)': 'Female-specific factors',
                    'Age started smoking in former smokers': 'Smoking',
                    'Age high blood pressure diagnosed': 'Medical conditions',
                    'White blood cell (leukocyte) count': 'Blood count',
                    'Red blood cell (erythrocyte) count': 'Blood count',
                    'Haemoglobin concentration': 'Blood count',
                    'Haematocrit percentage': 'Blood count',
                    'Mean corpuscular volume': 'Blood count',
                    'Mean corpuscular haemoglobin': 'Blood count',
                    'Mean corpuscular haemoglobin concentration': 'Blood count',
                    'Red blood cell (erythrocyte) distribution width': 'Blood count',
                    'Platelet count': 'Blood count',
                    'Platelet crit': 'Blood count',
                    'Mean platelet (thrombocyte) volume': 'Blood count',
                    'Lymphocyte count': 'Blood count',
                    'Monocyte count': 'Blood count',
                    'Neutrophill count': 'Blood count',
                    'Eosinophill count': 'Blood count',
                    'Lymphocyte percentage': 'Blood count',
                    'Neutrophill percentage': 'Blood count',
                    'Eosinophill percentage': 'Blood count',
                    'Reticulocyte percentage': 'Blood count',
                    'Reticulocyte count': 'Blood count',
                    'Mean reticulocyte volume': 'Blood count',
                    'Mean sphered cell volume': 'Blood count',
                    'Immature reticulocyte fraction': 'Blood count',
                    'High light scatter reticulocyte count': 'Blood count',
                    'Creatinine (enzymatic) in urine': 'Urine assays',
                    'Potassium in urine': 'Urine assays',
                    'Sodium in urine': 'Urine assays',
                    '3mm weak meridian (left)': 'Autorefraction',
                    '6mm weak meridian (left)': 'Autorefraction',
                    '6mm weak meridian (right)': 'Autorefraction',
                    '3mm weak meridian (right)': 'Autorefraction',
                    '3mm strong meridian (right)': 'Autorefraction',
                    '6mm strong meridian (right)': 'Autorefraction',
                    '6mm strong meridian (left)': 'Autorefraction',
                    '3mm strong meridian (left)': 'Autorefraction',
                    '3mm strong meridian angle (left)': 'Autorefraction',
                    'Standing height': 'Body measures',
                    'Hip circumference': 'Body measures',
                    'Albumin (quantile)': 'Blood biochemistry',
                    'Alanine aminotransferase (quantile)': 'Blood biochemistry',
                    'Aspartate aminotransferase (quantile)': 'Blood biochemistry',
                    'Urea (quantile)': 'Blood biochemistry',
                    'Calcium (quantile)': 'Blood biochemistry',
                    'Creatinine (quantile)': 'Blood biochemistry',
                    'Gamma glutamyltransferase (quantile)': 'Blood biochemistry',
                    'Glycated haemoglobin (quantile)': 'Blood biochemistry',
                    'Phosphate (quantile)': 'Blood biochemistry',
                    'Testosterone (quantile)': 'Blood biochemistry',
                    'Total protein (quantile)': 'Blood biochemistry',
                    'Forced vital capacity (FVC)': 'Spirometry',
                    'Forced expiratory volume in 1-second (FEV1)': 'Spirometry',
                    'Peak expiratory flow (PEF)': 'Spirometry',
                    'Ankle spacing width': 'Bone-densitometry of heel',
                    'Heel Broadband ultrasound attenuation, direct entry': 'Bone-densitometry of heel',
                    'Heel quantitative ultrasound index (QUI), direct entry': 'Bone-densitometry of heel',
                    'Heel bone mineral density (BMD)': 'Bone-densitometry of heel',
                    'Age at menopause (last menstrual period)': 'Female-specific factors',
                    'Number of incorrect matches in round': 'Pairs matching',
                    'Time to complete round': 'Pairs matching',
                    'Duration to first press of snap-button in each round': 'Reaction time',
                    'Diastolic blood pressure, automated reading': 'Blood pressure',
                    'Systolic blood pressure, automated reading': 'Blood pressure',
                    'Ankle spacing width (left)': 'Bone-densitometry of heel',
                    'Heel broadband ultrasound attenuation (left)': 'Bone-densitometry of heel',
                    'Heel quantitative ultrasound index (QUI), direct entry (left)': 'Bone-densitometry of heel',
                    'Heel bone mineral density (BMD) (left)': 'Bone-densitometry of heel',
                    'Heel bone mineral density (BMD) T-score, automated (left)': 'Bone-densitometry of heel',
                    'Ankle spacing width (right)': 'Bone-densitometry of heel',
                    'Heel broadband ultrasound attenuation (right)': 'Bone-densitometry of heel',
                    'Hospital episode type': 'Summary administration',
                    'Spells in hospital': 'Summary administration',
                    'Heel quantitative ultrasound index (QUI), direct entry (right)': 'Bone-densitometry of heel',
                    'Heel bone mineral density (BMD) (right)': 'Bone-densitometry of heel',
                    'Heel bone mineral density (BMD) T-score, automated (right)': 'Bone-densitometry of heel',
                    'Pulse rate': 'Arterial stiffness',
                    'Pulse wave reflection index': 'Arterial stiffness',
                    'Duration screen displayed': 'Prospective memory',
                    'Happiness': 'Mental health',
                    'Health satisfaction': 'Mental health',
                    'Family relationship satisfaction': 'Mental health',
                    'Friendships satisfaction': 'Mental health',
                    'Financial situation satisfaction': 'Mental health',
                    'Ever depressed for a whole week': 'Mental health',
                    'Hand grip strength (left)': 'Hand grip strength',
                    'Leg pain on walking': 'Claudication and peripheral artery disease',
                    'Hand grip strength (right)': 'Hand grip strength',
                    'Tinnitus': 'Hearing',
                    'Noisy workplace': 'Hearing',
                    'Waist circumference': 'Body size measures',
                    'FI3 ': 'Fluid intelligence / reasoning',
                    'FI4 ': 'Fluid intelligence / reasoning',
                    'FI6 ': 'Fluid intelligence / reasoning',
                    'FI8 ': 'Fluid intelligence / reasoning',
                    'Spherical power (right)': 'Autorefraction',
                    'Spherical power (left)': 'Autorefraction',
                    '3mm cylindrical power (right)': 'Autorefraction',
                    '3mm cylindrical power (left)': 'Autorefraction',
                    'Intra-ocular pressure, corneal-compensated (right)': 'Intraocular pressure',
                    'Intra-ocular pressure, Goldmann-correlated (right)': 'Intraocular pressure',
                    'Corneal hysteresis (right)': 'Intraocular pressure',
                    'Corneal resistance factor (right)': 'Intraocular pressure',
                    'Intra-ocular pressure, corneal-compensated (left)': 'Intraocular pressure',
                    'Intra-ocular pressure, Goldmann-correlated (left)': 'Intraocular pressure',
                    'Corneal hysteresis (left)': 'Intraocular pressure',
                    'Corneal resistance factor (left)': 'Intraocular pressure',
                    'Gas or solid-fuel cooking/heating': 'Household',
                    'Current employment status': 'Employment',
                    'Length of time at current address': 'Household',
                    'Number of vehicles in household': 'Household',
                    'Heel bone mineral density (BMD) T-score, automated': 'Bone-densitometry of heel',
                    'Job involves mainly walking or standing': 'Employment',
                    'Job involves heavy manual or physical work': 'Employment',
                    'Job involves shift work': 'Employment',
                    'Age completed full time education': 'Education',
                    'Number of days/week walked 10+ minutes': 'Physical activity',
                    'Duration of walks': 'Physical activity',
                    'Number of days/week of moderate physical activity 10+ minutes': 'Physical activity',
                    'Duration of moderate activity': 'Physical activity',
                    'Number of days/week of vigorous physical activity 10+ minutes': 'Physical activity',
                    'Duration of vigorous activity': 'Physical activity',
                    'Usual walking pace': 'Physical activity',
                    'Frequency of stair climbing in last 4 weeks': 'Physical activity',
                    'Frequency of walking for pleasure in last 4 weeks': 'Physical activity',
                    'Carpal tunnel syndrome': 'Nervous system disorders',
                    'Nerve, nerve root and plexus disorders': 'Nervous system disorders',
                    'Disorders of lens': 'Eyesight',
                    'Major coronary heart disease event': 'Circulatory system disorders',
                    'Coronary atherosclerosis': 'Circulatory system disorders',
                    'Diseases of veins, lymphatic vessels and lymph nodes, not elsewhere classified': 'Circulatory system disorders',
                    'Ischaemic heart disease, wide definition': 'Circulatory system disorders',
                    'Any ICDMAIN event in hilmo or causes of death': 'Circulatory system disorders',
                    'Diseases of the circulatory system': 'Circulatory system disorders',
                    'Disorders of gallbladder, biliary tract and pancreas': 'Circulatory system disorders',
                    'Gonarthrosis [arthrosis of knee](FG)': 'Circulatory system disorders',
                    '#Arthrosis': 'Circulatory system disorders',
                    'Dorsalgia': 'Summary diagnoses',
                    '#Other joint disorders': 'Summary diagnoses',
                    'Diseases of the nervous system': 'Nervous system disorders',
                    'Diseases of the musculoskeletal system and connective tissue': 'Circulatory system disorders',
                    'Injury, poisoning and certain other consequences of external causes': 'Summary diagnoses',
                    'Diseases of the digestive system': 'Summary diagnoses',
                    'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified': 'Summary diagnoses',
                    }
    # adding the general group of each PRS
    high_sig['phen_group'] = high_sig['Phenotype Description'].apply(lambda x: prefix_group.get(x.split(':')[0], None))
    # list of our interesting PRS
    my_phengroup = ['Family history', 'Medical conditions',
                    'Circulatory system disorders', 'Intraocular pressure', 'Summary Diagnoses',
                    'Pain', 'Blood pressure','Hearing', 'Mental health', 'Depression',
                    'General health', 'Nervous system disorders',
                    'Arterial stiffness', 'Chest pain', 'Breathing', 'Claudication and peripheral artery disease',
                    'Medical information']

    # filtering the PRS to only relevant ones which are also segnificant to genetics
    phen_group_bool = []
    for i in high_sig['phen_group']:
        phen_group_bool.append(i in my_phengroup)
    high_sig['phen_group_bool'] = phen_group_bool
    # list of the ralvant and corralted to genetics PRS
    sig = high_sig.index[high_sig['phen_group_bool']].to_list()
    # removing duplicated of PRS's name
    siguniqe = list(dict.fromkeys(sig))
    return (siguniqe)

def return_comp_dict(outcomes_dict):
    '''
    The purpose of this function is to define the PRSes to test for association with each
    UKBB disease
    '''

    comp_dict = {o : [] for o in outcomes_dict.keys()}

    cvd_prs_list = ['Pulse rate, automated reading', 'Non-cancer illness code, self-reported: hypertension',
                    'Non-cancer illness code, self-reported: angina', 'Non-cancer illness code, self-reported: heart attack/myocardial infarction',
                    'Illnesses of father: Heart disease', 'Illnesses of father: High blood pressure', 'Illnesses of mother: High blood pressure',
                    'Illnesses of siblings: Heart disease', 'Illnesses of siblings: High blood pressure','Chest pain or discomfort',
                    'Age high blood pressure diagnosed', 'Diastolic blood pressure, automated reading', 'Systolic blood pressure, automated reading',
                    'Pulse rate', 'Pulse wave reflection index', 'Vascular/heart problems diagnosed by doctor: Heart attack',
                    'Vascular/heart problems diagnosed by doctor: Angina', 'Vascular/heart problems diagnosed by doctor: High blood pressure',
                    'Major coronary heart disease event', 'Coronary atherosclerosis', 'Ischaemic heart disease, wide definition',
                    'Diagnoses - main ICD10: I25 Chronic ischaemic heart disease', 'Diseases of the circulatory system',

                    ]
    osteo_RA_prs_list = ['Pain type(s) experienced in last month: Knee pain','Pain type(s) experienced in last month: Hip pain',
                   'Leg pain on walking', 'Diagnoses - main ICD10: M54 Dorsalgia']
    metab_prs_list  = ['Non-cancer illness code, self-reported: diabetes', 'Non-cancer illness code, self-reported: high cholesterol',
                       'Illnesses of father: Diabetes', 'Illnesses of mother: Diabetes', 'Illnesses of siblings: Diabetes',
                       'Diabetes diagnosed by doctor', 'Coronary atherosclerosis', 'Non-cancer illness code, self-reported: angina',
                       'Diagnoses - main ICD10: I25 Chronic ischaemic heart disease', 'Vascular/heart problems diagnosed by doctor: Angina',
                      ]

    resp_prs_list = ['Wheeze or whistling in the chest in last year', 'Non-cancer illness code, self-reported: asthma',
                     'Non-cancer illness code, self-reported: hayfever/allergic rhinitis', 'Doctor diagnosed hayfever or allergic rhinitis',
                     'Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor: Asthma',
                     'Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor: Hayfever, allergic rhinitis or eczema',
                     ]
    pulm_prs_list = ['Wheeze or whistling in the chest in last year', 'Non-cancer illness code, self-reported: asthma',
                     'Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor: Asthma',]

    mental_prs_list = ['Non-cancer illness code, self-reported: depression', 'Mood swings', 'Miserableness', 'Irritability',
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
                      ]
    cvd_ukb_list = ['essential (primary) hypertension', 'nonrheumatic aortic valve disorders', 'nonrheumatic mitral valve disorders',
                    'atrioventricular and left bundle-branch block', 'atrial fibrillation and flutter', 'cardiomyopathy',
                    'chronic ischaemic heart disease', 'heart failure', 'other acute ischaemic heart diseases', 'acute myocardial infarction',
                    'obesity', 'pulmonary oedema', 'other peripheral vascular diseases',
                   ]

    metab_ukb_list = ['disorders of lipoprotein metabolism and other lipidaemias', 'atherosclerosis',
                      'non-insulin-dependent diabetes mellitus', 'unspecified diabetes mellitus',
                      'insulin-dependent diabetes mellitus', 'obesity', 'chronic renal failure', 'essential (primary) hypertension',

                      ]
    diabetes_ukb_list = ['non-insulin-dependent diabetes mellitus', 'unspecified diabetes mellitus',
                      'insulin-dependent diabetes mellitus',]
    resp_ukb_list = ['asthma','chronic_sinusitis']

    for n, o in comp_dict.items():
        if outcomes_dict[n] == 'osteoporosis with pathological fracture':
            comp_dict[n] += (['Fractured/broken bones in last 5 years']+ osteo_RA_prs_list)
        if outcomes_dict[n] == 'gout':
            comp_dict[n] += ['Leg pain on walking']
        if outcomes_dict[n] == 'other rheumatoid arthritis':
            comp_dict[n] += osteo_RA_prs_list
        if outcomes_dict[n] in cvd_ukb_list:
            comp_dict[n] += cvd_prs_list
        if outcomes_dict[n] in metab_ukb_list:
            comp_dict[n] += metab_prs_list
        if outcomes_dict[n] in diabetes_ukb_list:
            comp_dict[n] +=['Diagnoses - main ICD10: G56 Mononeuropathies of upper limb']
        if outcomes_dict[n] == 'other chronic obstructive pulmonary disease':
            comp_dict[n] += pulm_prs_list
        if outcomes_dict[n] == 'obesity':
            comp_dict[n] += ['Pain type(s) experienced in last month: Knee pain']
        if outcomes_dict[n] in ['recurrent depressive disorder', 'other anxiety disorders']:
            comp_dict[n] += mental_prs_list
        if outcomes_dict[n] == 'migraine':
            comp_dict[n] += ['Pain type(s) experienced in last month: Headache']
        if outcomes_dict[n] in resp_ukb_list:
            comp_dict[n] += resp_prs_list
        if outcomes_dict[n] == 'other hypothryroidism':
            comp_dict[n] += ['Non-cancer illness code, self-reported: hypothyroidism/myxoedema']
        if outcomes_dict[n] == 'cholelithiasis':
            comp_dict[n] += ['Disorders of gallbladder, biliary tract and pancreas']


    for o in comp_dict:
        comp_dict[o] = list(set(comp_dict[o]))

    return comp_dict


def write_getsigunique_cache(dir="/net/mraid08/export/jafar/UKBioBank/davidpel/UKBB_data/",
                             fileName="getsigunique_cache_new.csv"):
    pd.Series(getsigunique(cached=False)).to_csv(dir + fileName)

def load_sigunique_data(dir="/net/mraid08/export/jafar/UKBioBank/davidpel/UKBB_data/",
                             fileName="getsigunique_cache_new.csv"):

    sigunique = pd.read_csv(os.path.join(dir,fileName),index_col=0,header=0,names=['prs'])
    prs_data = PRSLoader().get_data(study_ids=['10K']).df
    prs_data.columns = prs_data.columns.astype(str)
    prs_data = prs_data[sigunique.iloc[:,0].tolist()]
    prs_data = prs_data.loc[:,~prs_data.columns.duplicated(keep='first')]
    prs_col_md = PRSLoader().get_data(study_ids=['10K']).df_columns_metadata['h2_description'].to_dict()
   # prs_col_md = {v:k for k,v in prs_col_md.items() if not pd.isna(v)}
    prs_data = prs_data.rename(columns=prs_col_md)
    return prs_data


def get_outcome_corrs(outcome, prs_df, method):

    outcome_preds = load_outcome(outcome).sort_index()
    covariates = load_age_gender_bmi().sort_index()
    prs_df = prs_df.sort_index()

    results = {}
    for var in prs_df.columns:
        var_df = pd.concat([outcome_preds, covariates, prs_df[var]], axis=1).dropna(
            subset=var).dropna(subset=outcome_preds.columns.to_list())

        results[var] = run_partial_corr(var_df, var, method, by_gender=True)[var]
    res_df = pd.DataFrame(results).T
    print(res_df)
    return res_df


def run_vs_predictions(prs_df,method='spearman'):
    sethandlers()
    outcomes = pd.read_csv(OUTCOMES_LIST).astype({"UKBB Field ID": 'int'})
    outcomes_dict = dict(zip(outcomes['UKBB Field ID'].values, outcomes['UKBB Description'].values))
    for o in outcomes_dict:
        outcomes_dict[o] = '('.join(outcomes_dict[o][:-1].split('(')[1:])

    comp_dict = return_comp_dict(outcomes_dict)
    old_cwd = os.getcwd()
    old_cwd = os.getcwd()
    os.chdir(OUTPATH)

    results = {}
    with qp(jobname='partial_corr', _delete_csh_withnoerr=False, q=['himem7.q'], _trds_def=1, max_r=500,
            _mem_def='2G') as q:
        q.startpermanentrun()

        for outcome in outcomes_dict.keys():

            pred_dir = os.path.join(topdir, str(outcome), "tenk", "full_ukb")

            predfile = os.path.join(pred_dir, "tenk_predictions.csv")
            if not os.path.isfile(predfile): continue
            comp_df = prs_df[comp_dict[outcome]]
            results[outcome] = q.method(get_outcome_corrs, (outcome, comp_df, method))

        results = {k: q.waitforresult(v) for k, v in results.items()}

    os.chdir(old_cwd)
    prs_corrs = pd.concat([results[k] for k in results.keys()], axis=1, keys=[outcomes_dict[o] for o in results.keys()])
    prs_corrs.to_csv(os.path.join(topdir, method, 'full_ukb', "all_outcomes_partial_corrs_prs_new.csv"))


def main():
    use_cached = False
    if not use_cached:
        write_getsigunique_cache()
    prs_data = load_sigunique_data()
    run_vs_predictions(prs_data)


if __name__ == "__main__":
    main()

