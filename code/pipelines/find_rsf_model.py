from LabData.DataAnalyses.UKBB_10k.UKBB_model_pipeline import UKBB_model
from LabData.DataAnalyses.UKBB_10k.UKBB_survival import *
import os
from LabUtils.addloglevels import sethandlers

from LabQueue.qp import fakeqp as qp# qp#
import time

def run_gridsearch(model):
    clf = UKBB_model(model=model,use_cache=True)
    clf.run_gridsearch_pipeline()

def main():
  #  model = UKBB_model(model='regularized_cox', use_cache=True)

    old_cwd = os.getcwd()
    os.chdir(OUTPATH)

    sethandlers()

    with qp(jobname='gridsearch', _delete_csh_withnoerr=False, q=['himem7.q'], _trds_def=8, max_r=200,
            _mem_def='100G') as q:
        q.startpermanentrun()
        qmethod = q.method(run_gridsearch, ('rsf',))#model.run_gridsearch_pipeline)
        q.waitforresult(qmethod)  # use waitforresult*s* if qmethod is a list
    logger.info("Done search")
    os.chdir(old_cwd)


    # model.load_cohort()
    # model._plot_joint_features()


if __name__ == "__main__":
    main()