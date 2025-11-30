import os

#######################################################################
# TODO !!
os.environ["OMP_NUM_THREADS"] = "16" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "16" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "16" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "16" # export NUMEXPR_NUM_THREADS=1

from LabData.DataAnalyses.UKBB_10k.UKBB_model_pipeline import UKBB_model
from LabData.DataAnalyses.UKBB_10k.UKBB_survival import *

from LabUtils.addloglevels import sethandlers

from LabQueue.qp import qp # # fakeqp as qp
import time

def run_gridsearch(model):
    outcome = 132152
    print(outcome)
    logger.info("Running {} for {}".format(model, outcome))
    clf = UKBB_model(model=model,use_cache=True,outcome_field = outcome)#, feature_fields=[21003,31,21001])#, imputation='iter_mode_RF') # age,gender,bmi
    clf.run_gridsearch_pipeline()

def main():
  #  model = UKBB_model(model='regularized_cox', use_cache=True)

    old_cwd = os.getcwd()
    os.chdir(OUTPATH)

    sethandlers()

    with qp(jobname='search', _delete_csh_withnoerr=False, q=['himem7.q'], _trds_def=16, max_r=600,
            _mem_def='25G', _max_secs=-1) as q:
        q.startpermanentrun()
        qmethod = q.method(run_gridsearch, ('regularized_cox',))#model.run_gridsearch_pipeline)
        q.waitforresult(qmethod)  # use waitforresult*s* if qmethod is a list
    logger.info("Done search")
    os.chdir(old_cwd)


    # model.load_cohort()
    # model._plot_joint_features()


if __name__ == "__main__":
    main()