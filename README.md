## Efficient adjustment for complex covariates
Reproduces the experiment from `Efficient adjustment for complex covariates:
Gaining efficiency with DOPE'.
[arXiv:2402.12980](https://arxiv.org/abs/2402.12980).

## Setup
To run the simulation study
> python run_simulations.py

Optionally, specify `--file_save_path`, `--repetitions`, `--crossfit` or `--seed`.

The plots in the main manuscript are generated by running:
> python create_plots.py

The NHANES dataset(s) was loaded and cleaned using
> cd NHANES_application

> python clean_data.py

> python clean_data.py --impute_all

and the NHANES analysis was run with
> python analyze_nhanes.py --tune_hyperparameters

> python analyze_nhanes.py --use_imputed --tune_hyperparameters

