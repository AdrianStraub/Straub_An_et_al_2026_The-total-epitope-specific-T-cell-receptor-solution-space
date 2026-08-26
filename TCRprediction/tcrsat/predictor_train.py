import comet_ml
import torch
import pytorch_lightning as pl
import randomname
import yaml
import argparse
import os
from pprint import pprint
import platform

from time import strftime
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.tuner import Tuner

from data import TCRSatDataset
from models.predictor import Predictor, EsmFreezeUnfreeze
from utils import check_offline, CustomModelCheckpointCallback


torch.set_float32_matmul_precision('medium')


def train(config):
    start_time = strftime('%Y%m%d-%H%M%S')
    experiment_id = f'{start_time}_' + randomname.get_name()
    seed_everything(42)

    if isinstance(config, str):
        print(f'Loading config from {config}')
        config = yaml.safe_load(open(config, 'r'))
        pprint(config)
    elif isinstance(config, dict):
        print('Using config dict')
        pprint(config)
    else:
        raise ValueError('config must be either a path to a yaml file or a dict')

    model = Predictor(**config)

    train_dataset = TCRSatDataset(config=config['data_config'], split='train', tokenizer=model.tokenizer)
    model.set_train_dataset(train_dataset)
    val_dataset = TCRSatDataset(config=config['data_config'], split='val', tokenizer=model.tokenizer)
    model.set_val_dataset(val_dataset)

    os.makedirs(f'logs/{experiment_id}', exist_ok=True)
    os.makedirs(f'saved_models/{experiment_id}', exist_ok=True)
    workspace = 'deepozean' if os.environ["USER"] != 'felixrupert.drost' else 'saturationbusch'
    comet_logger = CometLogger(api_key=open(f'../../API Keys/comet-ml', 'r').read().rstrip('\n'), workspace=workspace,
                               project_name=f'tcrsat-predictor', experiment_name=experiment_id, save_dir=f'logs/{experiment_id}/',
                               offline=check_offline())
    callbacks = []
    callbacks.append(CustomModelCheckpointCallback(ignore_first=config['train_config']['ignore_first'],
                                                   dirpath=f'saved_models/{experiment_id}', monitor='val_auc', mode='max', 
                                                   filename='auc_{epoch}', save_last=True, save_top_k=1, verbose=False))
    callbacks[0].CHECKPOINT_NAME_LAST = 'last-{epoch}'
    callbacks.append(CustomModelCheckpointCallback(ignore_first=config['train_config']['ignore_first'],
                                                   dirpath=f'saved_models/{experiment_id}', monitor='val_aps', mode='max',
                                                   filename='aps_{epoch}', save_last=False, save_top_k=1, verbose=False))
    if 'type' not in config['head_config'] or config['head_config']['type']=='binary':
        callbacks.append(CustomModelCheckpointCallback(ignore_first=config['train_config']['ignore_first'],
                                                       dirpath=f'saved_models/{experiment_id}', monitor='val_f1_thresh_f1', mode='max',
                                                       filename='f1_{epoch}', save_last=False, save_top_k=1, verbose=False))

    callbacks.append(EarlyStopping(monitor='val_auc', patience=config['train_config']['early_stopping'], 
                                   mode='max', verbose=True))
    callbacks.append(EsmFreezeUnfreeze(unfreeze_at_epoch=config['train_config']['unfreeze_at_epoch']))

    trainer = pl.Trainer(accelerator='cuda' if torch.cuda.is_available() else 'cpu',
                         max_epochs=config['train_config']['max_epochs'], log_every_n_steps=1, logger=comet_logger,
                         callbacks=callbacks, num_sanity_val_steps=0, max_time='01:23:50:00')
    tuner = Tuner(pl.Trainer(accelerator='cuda' if torch.cuda.is_available() else 'cpu', num_sanity_val_steps=0,
                callbacks=[EsmFreezeUnfreeze(unfreeze_at_epoch=-1)]))
    tuner.scale_batch_size(model, mode='power', init_val=2, max_trials=3 if platform.system() == 'Darwin' else 8)
    model.batch_size = int(model.batch_size / 2)  # TODO sometimes scale_batch_size overestimates, so we divide by 2

    
    trainer.fit(model)

    return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/predictor/example.yaml')
    parser.add_argument('--comment', type=str, default='', help='Comment for CometML')
    args = parser.parse_args()

    train(args.config)
