import os

#######################################################################
# TODO !!
os.environ["OMP_NUM_THREADS"] = "12" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "12" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "12" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "12" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "12" # export NUMEXPR_NUM_THREADS=1



from LabData.DataAnalyses.UKBB_10k.UKBB_model_pipeline import UKBB_model
from LabData.DataAnalyses.UKBB_10k.UKBB_survival import *

from LabUtils.addloglevels import sethandlers

from LabQueue.qp import qp # fakeqp as qp # #


def run_train(pipe_dict,params_dict):
    outcome = 132072
    logger.info("Training {} for {}".format(pipe_dict['model'],outcome))
    # Full model on all ukb data:
    clf = UKBB_model(pipe_dict['model'],use_cache=True,outcome_field = outcome,parquet_file='ukb669318.parquet',full=True)#,train=True#, feature_fields=[21003,31,21001]) # age,gender,bmi
    # Train specific (best) model on training data
   # clf = UKBB_model(pipe_dict['model'], use_cache=True, outcome_field=outcome, parquet_file='train_ukb669318.parquet', train=True)#, feature_fields=[21003,31,21001]) # age,gender,bmi
    clf.train_model(pipe_dict,params_dict)

def main():

    old_cwd = os.getcwd()
    os.chdir(OUTPATH)

    sethandlers()

    pipe_dict = {'imp': 'iter_mode', 'prep' : "standard", 'model': "regularized_cox"}
    params_dict = {"imp__num_imp__tol": 0.001,
                   'model__penalizer': 0.001,
                   'model__l1_ratio': 0.697
                  }

    with qp(jobname='train_full', _delete_csh_withnoerr=False, q=['himem7.q'], _trds_def=12, max_r=500,
            _mem_def='10G') as q:
        q.startpermanentrun()
        qmethod = q.method(run_train, (pipe_dict,params_dict))#model.run_gridsearch_pipeline)
        q.waitforresult(qmethod)  # use waitforresult*s* if qmethod is a list
    logger.info("Done train")
    os.chdir(old_cwd)



if __name__ == "__main__":
    main()