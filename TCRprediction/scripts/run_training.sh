#!/bin/bash

EPI="$1"
SPLIT="$2"

conda activate tcrsat

python ../tcrsat/predictor_optuna.py --data_config_split_col group_stratified_split_${SPLIT} --data_config_path ../data/${EPI}_with_background.csv --data_config_data_col full --train_config_early_stopping 5
