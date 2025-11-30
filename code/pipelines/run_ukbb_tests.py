from LabData.DataAnalyses.UKBB_10k.UKBB_model_pipeline import UKBB_model
from LabData.DataAnalyses.UKBB_10k.UKBB_survival import *
import os
from LabUtils.addloglevels import sethandlers

from LabQueue.qp import qp # fakeqp as qp #


def run_ukbb_test(model,outcome):
    logger.info("Applying {} for {} on ukbb test".format(model,outcome))
    clf = UKBB_model(model,use_cache=True,test=True,outcome_field = outcome,parquet_file='test_ukb669318.parquet')#, feature_fields=[21003,31,21001])#, imputation='iter_mode_RF') # age,gender,bmi
    cind = clf.apply_pipeline_to_ukbb_test()
    return cind

def main():

    print(run_ukbb_test('regularized_cox',132072))


    # old_cwd = os.getcwd()
    # os.chdir(OUTPATH)
    #
    # sethandlers()
    #
    #
    # outcomes = pd.read_csv(OUTCOMES_LIST).astype({"UKBB Field ID": 'int'})
    # outcomes_dict = dict(zip(outcomes['UKBB Field ID'].values, outcomes['UKBB Description'].values))
    # for o in outcomes_dict:
    #     outcomes_dict[o] = '('.join(outcomes_dict[o][:-1].split('(')[1:])
    #
    # cancer_outcomes = pd.read_csv(CANCER_OUTCOMES)
    # cancer_outcomes_dict = dict(zip(cancer_outcomes['ICD10'].values, cancer_outcomes['Type of cancer'].values))
    #
    # with qp(jobname='apply_baseline', _delete_csh_withnoerr=False, q=['himem7.q'], _trds_def=6, max_r=200,
    #         _mem_def='15G') as q:
    #     q.startpermanentrun()
    #     res = {}
    #     for outcome in list(outcomes_dict.keys())+list(cancer_outcomes_dict.keys()):
    #         if not os.path.exists(os.path.join(OUTPATH,'regularized_cox',str(outcome),'full_data')):
    #             continue
    #         res[outcome] = q.method(run_ukbb_test, ('regularized_cox',outcome))
    #     res = {k: q.waitforresult(v) for k, v in res.items()}
    #
    #     cind_df = pd.DataFrame.from_dict(res,orient='index',columns=['cind'])
    #     cind_df.to_csv(os.path.join(OUTPATH, 'regularized_cox', 'ukbb_test_cinds.csv'))
    # logger.info("Done")
    # os.chdir(old_cwd)



if __name__ == "__main__":
    main()