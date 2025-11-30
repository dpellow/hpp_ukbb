import pickle

from LabData.DataAnalyses.UKBB_10k.UKBB_model_pipeline import UKBB_model
from LabData.DataAnalyses.UKBB_10k.UKBB_survival import *
import os
from LabUtils.addloglevels import sethandlers

from sklearn.ensemble import RandomForestRegressor

from LabQueue.qp import qp#
import time


def impute(method):
    clf = UKBB_model(model="regularized_cox",use_cache=True)
    data = clf.load_raw_train_features()
    clf.define_training_pipeline_elems()

    if method == 'simple':
        imp = clf.simple_impute
    # TODO: add missforest (ie randomforestregressor instead of bayesianridge)
    if method == 'iter_zero':
        imp = clf.iterative_zero_impute
        imp.set_params(**{"num_imp__estimator":RandomForestRegressor(n_estimators=50, max_depth=10,
                                                                           max_samples=0.5)})
    if method == 'iter_mode':
        imp = clf.iterative_mode_impute
        imp.set_params(**{"num_imp__estimator":RandomForestRegressor(n_estimators=50, max_depth=10,
                                                                           max_samples=0.5)})

    imp_data = imp.fit_transform(data)
    # save imputed data
    data_path = os.path.join(CACHEPATH,"{}_RF_imputed_ukbb_features.csv".format(method))
    imp_data.to_csv(data_path)
    # save imputer
    imp_path = os.path.join(CACHEPATH, "{}_RF_imputer".format(method))
    with open(imp_path,'wb') as o:
        pickle.dump(imp,o)
    return True



def main():
  #  model = UKBB_model(model='regularized_cox', use_cache=True)

    imp_methods = ['simple','iter_zero','iter_mode']

    old_cwd = os.getcwd()
    os.chdir(OUTPATH)

    sethandlers()

    with qp(jobname='imputation', _delete_csh_withnoerr=False, q=['himem7.q'], _trds_def=12, max_r=200,
            _mem_def='50G') as q:
        q.startpermanentrun()
        run_res = []
        for m in imp_methods:
            run_res.append(q.method(impute, (m,)))#model.run_gridsearch_pipeline)
        res = q.waitforresults(run_res)  # use waitforresult*s* if qmethod is a list
    logger.info("Done search")
    os.chdir(old_cwd)


if __name__ == "__main__":
    main()