import optuna

# On V100 with 32GB GPU and 1024 max_seq_len grows quadratically
max_batch_size_dict = {'facebook/esm2_t6_8M_UR50D': 9,  # 32  TODO
				  	   'facebook/esm2_t12_35M_UR50D': 8,  # 16
					   'facebook/esm2_t30_150M_UR50D': 6}  # 4


def propose(args, trial: optuna.trial.Trial) -> dict:
	if args.pretrained is None:
		pretrained = trial.suggest_categorical('pretrained', ['facebook/esm2_t6_8M_UR50D', 'facebook/esm2_t12_35M_UR50D', 'facebook/esm2_t30_150M_UR50D'])
	elif not args.pretrained:
		pretrained = None
		hidden_size = 2**trial.suggest_int('esm_config/hidden_size', 6, 9)
	else:
		pretrained = args.pretrained

	if pretrained is not None:
		max_batch_size = max_batch_size_dict[pretrained]
	else:
		max_batch_size = 6  # TODO

	if args.data_config_data_col == 'full':
		data_config_max_seq_len = 463
	elif args.data_config_data_col == 'cdr3':
		data_config_max_seq_len = 38
	elif args.data_config_data_col == 'IR_VDJ_1_junction_aa':
		data_config_max_seq_len = 20
	elif args.data_config_data_col == 'IR_VJ_1_junction_aa':
		data_config_max_seq_len = 24
	elif args.data_config_data_col == 'full_alpha_aa_1':
		data_config_max_seq_len = 228
	elif args.data_config_data_col == 'full_beta_aa_1':
		data_config_max_seq_len = 296

	config = {
		'pretrained': pretrained,
		'esm_config': None if not args.esm_config else {
			'attention_probs_dropout_prob': 0.0,
			'emb_layer_norm_before': False,
			'esmfold_config': None,
			'hidden_dropout_prob': 0.0,
			'hidden_size': hidden_size,  # 320-640 in original ESM, here 64-512
			'initializer_range': 0.02,
			'intermediate_size': 4 * hidden_size,  # 1280-2560, here 256-2048
			'is_folding_model': False,
			'layer_norm_eps': 1e-05,
			'mask_token_id': 32,
			'max_position_embeddings': 1026,
			'num_attention_heads': 16,
			'num_hidden_layers': trial.suggest_int('esm_config/num_hidden_layers', 2, 5),  # 6-30, here 4-32
			'pad_token_id': 1,
			'position_embedding_type': 'rotary',
			'token_dropout': True,
			'use_cache': True,
			'vocab_list': None,
			'vocab_size': 33,
		},
		'head_config': {
			'activation': 'relu',
			'batch_norm': True,
			'dropout': trial.suggest_float('head_config/dropout', 0.0, 0.25, step=0.05) if args.head_config_dropout is None else args.head_config_dropout,
			'hidden_neurons': 2**trial.suggest_int('head_config/hidden_neurons', 5, 9) if args.head_config_hidden_neurons is None else args.head_config_hidden_neurons,
			'num_hidden_layers': trial.suggest_int('head_config/num_hidden_layers', 1, 4) if args.head_config_num_hidden_layers is None else args.head_config_num_hidden_layers,
			'pool_type': 'start_token' if args.head_config_pool_type is None else args.head_config_pool_type,
			'pool_num_heads': 4,
			'embedding_dim': None,
		},
		'lora_config': None if not args.lora_config else {
			'peft_type': None,
			'base_model_name_or_path': None,
			'task_type': None,
			'inference_mode': False,
			'r': 2**trial.suggest_int('lora_config/r', 3, 7) if args.lora_config_r is None else args.lora_config_r,
			'target_modules': None,
			'lora_alpha': 2**trial.suggest_int('lora_config/lora_alpha', 3, 7) if args.lora_config_lora_alpha is None else args.lora_config_lora_alpha,
			'lora_dropout': trial.suggest_float('lora_config/lora_dropout', 0.0, 0.25, step=0.05) if args.lora_config_lora_dropout is None else args.lora_config_lora_dropout,
			'fan_in_fan_out': False,
			'bias': 'none',
			'modules_to_save': None,
			'init_lora_weights': True,
		},
		'train_config': {
			'batch_size': 2 if args.train_config_batch_size is None else args.train_config_batch_size,
			'lr': 10**(-trial.suggest_int('train_config/lr', 4, 6)) if args.train_config_lr is None else args.train_config_lr,
			'max_epochs': args.train_config_max_epochs,
			'early_stopping': args.train_config_early_stopping,
			'ignore_first': args.train_config_ignore_first,
			'unfreeze_at_epoch': args.train_config_unfreeze_at_epoch,
		},
		'data_config': {
			'path': args.data_config_path,
			'sample_size': args.data_config_sample_size,
			'split_col': args.data_config_split_col,
			'data_col': args.data_config_data_col,
			'max_seq_len': data_config_max_seq_len,
		},
		'comment': args.comment,
	}

	return config
