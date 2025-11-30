from LabData.DataAnalyses.UKBB_10k.UKBB_model_pipeline import UKBB_model
from LabData.DataAnalyses.UKBB_10k.UKBB_survival import *
import os
from LabUtils.addloglevels import sethandlers

from LabQueue.qp import qp # fakeqp as qp  #


def run_apply(model):

    outcome = 131702
    logger.info("Applying {} for {} on tenk".format(model,outcome))
    clf = UKBB_model(model,use_cache=True,tenk=True,full=True,outcome_field = outcome)#, feature_fields=[21003,31,21001])#, imputation='iter_mode_RF') # age,gender,bmi
    clf.apply_pipeline_to_tenk()

def main():

    old_cwd = os.getcwd()
    os.chdir(OUTPATH)

    sethandlers()


    with qp(jobname='apply', _delete_csh_withnoerr=False, q=['himem7.q'], _trds_def=6, max_r=200,
            _mem_def='10G') as q:
        q.startpermanentrun()
        qmethod = q.method(run_apply, ('regularized_cox',))#model.run_gridsearch_pipeline)
        q.waitforresult(qmethod)  # use waitforresult*s* if qmethod is a list
    logger.info("Done apply")
    os.chdir(old_cwd)



if __name__ == "__main__":
    main()