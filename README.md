#Mitigating Covariate Shift for Low Dimensional Machine Learning Problems via Lattice Based Models 
##Bachelor Thesis 

This directory contains all the necessary files to reproduce my bachelor thesis experiment. Simply clone the repo and create a condo environment with `conda env create -f environment.yml`; alternatively the main .R script contains an alternative method (creating a virtualenv) or set up python and the required packages manually. Besides this the enitire experiment process is contained in the tfl_experiment.R file.

The finished thesis can be found in the paper directory. Furthermore I supply my trained models. The entire project has the following structure:

```
$ pwd
/path/to/project/directory/ba_vwl

$ ls
|- ba_vwl.Rproj
|- data/
   |- raw/
	|- german/
	|- gmc/
	|- pak/
	|- taiwanese/
   |- README.md
|- environment.yma
|- paper/
   |- bachelor_thesis_j_schmitz.pdf
|- README.me
|- results/
   |- eda/
   |- exp_results/
	|- results.RDS
   |- lat_models/
	|- ger_model/
	|- gmc_model/
	|- pak_model/
	|- tcd_model/
|- scripts/
   |- feature_configs.py
   |- helpers.R
   |- tfl_experiment.R

``