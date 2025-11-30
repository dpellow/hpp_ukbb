from LabData.DataAnalyses.UKBB_10k.UKBB_model_pipeline import UKBB_model
from LabData.DataAnalyses.UKBB_10k.UKBB_survival import *
import os
from LabUtils.addloglevels import sethandlers

from LabQueue.qp import qp # fakeqp as qp #
import time

def run_gridsearch(model,outcome):
    logger.info("Running {} for {}".format(model,outcome))
    clf = UKBB_model(model=model,use_cache=True,outcome_field = outcome)#, feature_fields=[21003,31,21001])#, imputation='iter_mode_RF') # age,gender,bmi
    clf.run_gridsearch_pipeline()

def main():

    old_cwd = os.getcwd()
    os.chdir(OUTPATH)

    sethandlers()

    with qp(jobname='search', _delete_csh_withnoerr=False, q=['himem7.q'], _trds_def=40, max_r=500,
            _mem_def='20G') as q:
        q.startpermanentrun()
        #outcomes = pd.read_csv(CANCER_OUTCOMES)#.astype({"UKBB Field ID": 'int'})
        outcomes = pd.read_csv(OUTCOMES_LIST).astype({"UKBB Field ID": 'int'})
        run_res = []
        for outcome in outcomes["UKBB Field ID"]:#"ICD10"]:#
            run_res.append(q.method(run_gridsearch, ('regularized_cox',outcome)))
        q.waitforresults(run_res)
    logger.info("Done search")
    os.chdir(old_cwd)


if __name__ == "__main__":
    main()