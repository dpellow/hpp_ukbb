import sys
import warnings

import numpy as np
import pandas as pd

from pathlib import Path

#warnings.filters("ignore")

import autoprognosis.logger as log
from autoprognosis.studies.risk_estimation import RiskEstimationStudy

#from UKBB_survival import *
from UKBB_model_pipeline import *

log.add(sink=sys.stderr,level="INFO")


def create_study(dataset, field=131306):
    workspace = Path(OUTPATH)
    workspace.mkdir(parents=True, exist_ok=True)

    study_name = "autoprognosis_{}".format(field)

    eval_time_horizons = [
        int(dataset[dataset['event']==1]['durations'].quantile(0.25)),
        int(dataset[dataset['event'] == 1]['durations'].quantile(0.50)),
        int(dataset[dataset['event'] == 1]['durations'].quantile(0.75)),
    ]
    print(eval_time_horizons)

    study = RiskEstimationStudy(
        study_name=study_name,
        dataset=dataset,
        target='event',
        time_to_event="durations",
        time_horizons= eval_time_horizons,
        num_iter= 100,
        num_study_iter=10,
        timeout=3000,
        risk_estimators=['cox_ph', 'coxnet', 'survival_xgboost', 'weibull_aft'],# 'deephit', 'loglogistic_aft', 'lognormal_aft', 'survival_xgboost', 'weibull_aft'],
        imputers=['mean','missforest', 'mice', 'most_frequent','nop'],#, 'median'
        feature_scaling=['scaler','normal_transform', 'nop'],#, 'minmax_scaler', 'feature_normalizer'],
        score_threshold=0.5,
        workspace=workspace
    )
    return study


def run_study():
    clf = UKBB_model(model=model, use_cache=True)
    ukbb_data = clf.load_train_data()
    study = create_study(ukbb_data)
    study.run()

if __name__ == "__main__":
  run_study()