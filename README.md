This repo contains code and some of the raw data for **Analysis of biomarkers in the Human Phenotype Project using disease models from UK Biobank**

`code/plots.ipynb' is the code to replicate the plots for the paper. The environment needed for this notebook can be recreated from requirements.txt.

Raw data needed for these plots and other results is in `data`. Note that individual level data cannot be made publicly available and can be obtained via MTA as described in the paper.

`code/pipelines/` contains raw code for the anaylsis pipelines to generate the models, correlations, and other raw results. These are part of a larger codebase for accessing and processing the individual level data that cannot be made publicly available.
