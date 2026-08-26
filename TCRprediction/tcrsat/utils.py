from urllib.request import urlopen
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from pynvml import *
from platform import system


def print_gpu_utilization():
	if system() == 'Darwin':
		print('GPU utilization not available on Mac.')
		return
	nvmlInit()
	handle = nvmlDeviceGetHandleByIndex(0)
	info = nvmlDeviceGetMemoryInfo(handle)
	print(f"GPU memory occupied: {info.used//1024**2} MB.")


def get_gpu_utilization():
	if system() == 'Darwin':
		print('GPU utilization not available on Mac.')
		return 0
	nvmlInit()
	handle = nvmlDeviceGetHandleByIndex(0)
	info = nvmlDeviceGetMemoryInfo(handle)
	return info.used//1024**2

class CustomModelCheckpointCallback(pl.callbacks.ModelCheckpoint):
	def __init__(self, ignore_first=0, *args, **kwargs):
		super(CustomModelCheckpointCallback, self).__init__(*args, **kwargs)
		self.ignore_first = ignore_first

	def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
		if trainer.current_epoch + 1 > self.ignore_first:
			super().on_train_epoch_end(trainer, pl_module)

	def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
		if trainer.current_epoch + 1 > self.ignore_first:
			super().on_validation_end(trainer, pl_module)


def init_model(config):
	if config['model_type'] == 'autoregressive':
		from models.autoregressive import Autoregressive
		if config['train_config']['load_pretrained'] is not None:
			model = Autoregressive.load_from_checkpoint(config['train_config']['load_pretrained'],
														train_config=config['train_config'],
														# progen_config=config['progen_config'],  Overwrite all configs except for progen aka model_config
														generation_config=config['generation_config'],
														data_config=config['data_config'],
														lora_config=config['lora_config'])
			# model.train_config = config['train_config']
			# model.progen_config = config['progen_config']
			# model.generation_config = config['generation_config']
			# model.data_config = config['data_config']
			# model.lora_config = config['lora_config']
		else:
			model = Autoregressive(**config)
	else:
		raise NotImplementedError(f'Model {config["model"]} not implemented.')

	return model


def convert_str_to_bool(args):
	def str_to_bool(s):
		if not isinstance(s, str):
			return s
		if s.lower() == 'true':
			return True
		elif s.lower() == 'false':
			return False
		else:
			return s
	for key, value in vars(args).items():
		setattr(args, key, str_to_bool(value))

	return args


def check_offline():
	try:
		urlopen('https://www.google.com/', timeout=10)
		return False
	except:
		return True
