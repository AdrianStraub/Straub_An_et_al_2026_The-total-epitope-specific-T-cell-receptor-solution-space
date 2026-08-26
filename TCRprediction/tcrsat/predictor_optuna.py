import comet_ml
import optuna
import argparse
import randomname
import os
import torch
from importlib import import_module
from time import strftime

from predictor_train import train
from utils import convert_str_to_bool

import sys
sys.path.append('../')

parser = argparse.ArgumentParser()

# ESM configs
parser.add_argument('--config', type=str, default='config.predictor.example_optuna', help='Path to optuna config file')
parser.add_argument('--pretrained', type=str, default=None,
					choices=['facebook/esm2_t6_8M_UR50D', 'facebook/esm2_t12_35M_UR50D', 'facebook/esm2_t30_150M_UR50D', 'False'],
					help='Name of pretrained model as str, None to let Optuna propose, False to not use pretrained model')
parser.add_argument('--esm_config', type=str, default='False',
					help='True to manually configure model, False to use pretrained model')

# Head configs
parser.add_argument('--head_config_dropout', type=float, default=None,
					help='Dropout probability for head, None for Optuna to propose')
parser.add_argument('--head_config_hidden_neurons', type=int, default=None,
					help='Number of neurons per hidden layer in head, None for Optuna to propose')
parser.add_argument('--head_config_num_hidden_layers', type=int, default=None,
					help='Number of hidden layers in head, None for Optuna to propose')
parser.add_argument('--head_config_embedding_dim', type=int, default=None,
					help='Embedding dimension of amino acids, used only if not using pretrained model, None for Optuna to propose')
parser.add_argument('--head_config_pool_type', type=str, default=None,
					choices=['start_token', 'mean', 'max', 'attention'],
					help='Pooling type from [start_token, mean, max, attention], None for Optuna to propose')
parser.add_argument('--head_config_type', type=str, default='binary', 
                                        choices=['binary', 'multi_class', 'multi_label'],
                                        help='Predict between two classes, multiple classes with one true choice, or multiple classes with sevaral trues')

# Lora configs
parser.add_argument('--lora_config', type=str, default='True', help='True to use LoRA, False to not use LoRA')
parser.add_argument('--lora_config_r', type=int, default=None,
					help='Number of hops, None for Optuna to propose')
parser.add_argument('--lora_config_lora_alpha', type=int, default=None,
					help='Alpha for Lora, None for Optuna to propose')
parser.add_argument('--lora_config_lora_dropout', type=float, default=None,
					help='Dropout probability for Lora, None for Optuna to propose')

# Training configs
parser.add_argument('--train_config_batch_size', type=int, default=None, help='Batch Size, None for Optuna to propose')
parser.add_argument('--train_config_lr', type=float, default=None, help='Learning rate, None for Optuna to propose')
parser.add_argument('--train_config_max_epochs', type=int, default=10000, help='Max epochs')
parser.add_argument('--train_config_early_stopping', type=int, default=100, help='Early stopping patience')
parser.add_argument('--train_config_ignore_first', type=int, default=0, help='Do not save first n models')
parser.add_argument('--train_config_unfreeze_at_epoch', type=int, default=25, help='Unfreeze Esm at epoch')

# Data configs
parser.add_argument('--data_config_path', type=str, default='../data/full.pickle', help='Path to data')
parser.add_argument('--data_config_sample_size', type=int, default=None, help='Downsample data size, None to use all data')
parser.add_argument('--data_config_split_col', type=str, default='group_stratified_split_0', help='Column name for split')
parser.add_argument('--data_config_data_col', type=str, default='full', help='Column name for data')

# Comment
parser.add_argument('--comment', type=str, default='', help='Comment for CometML')

args = parser.parse_args()

# convert String 'True'/'False' to Boolean True/False
args = convert_str_to_bool(args)

def objective(trial: optuna.trial.Trial, args: argparse.Namespace) -> float:
	print('#' * 40 + ' NEW RUN ' + '#' * 40)
	print("Trial number: {}".format(trial.number))

	optuna_proposer = import_module(args.config, package=None)

	config = optuna_proposer.propose(args, trial)
	config['trial_number'] = trial.number

	trainer = train(config)
	best_model_score = trainer.checkpoint_callbacks[0].best_model_score

	return best_model_score


if __name__ == "__main__":
	start_time = strftime('%Y%m%d-%H%M%S')
	study_name = f'{start_time}_' + randomname.get_name()

	e = args.data_config_path.replace('../data', '').replace('_with_background.pickle', '')
	m = args.data_config_data_col
	s = args.data_config_split_col.replace('group_stratified_', '')

	study_name = f'{e}_{m}_{s}'

	os.makedirs(os.path.dirname('optuna/'), exist_ok=True)

	study = optuna.create_study(storage=f'sqlite:///optuna/{study_name}.db', load_if_exists=True,
								sampler=optuna.samplers.TPESampler(seed=42),
								study_name=study_name, direction='maximize')

	timeout = 47.5 * 3600  # 47.5 hours
	study.optimize(lambda trial: objective(trial, args), n_trials=1000, timeout=timeout, n_jobs=1)

	print("Number of finished trials: {}".format(len(study.trials)))

	print("Best trial:")
	trial = study.best_trial

	print("  Value: {}".format(trial.value))

	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))
