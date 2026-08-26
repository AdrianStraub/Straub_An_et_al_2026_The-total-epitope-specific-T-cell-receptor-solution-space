from transformers import EsmModel, EsmConfig, EsmTokenizer
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from peft import get_peft_model, LoraConfig
from torchmetrics.classification import BinaryAccuracy, BinaryAveragePrecision, BinaryAUROC, BinaryMatthewsCorrCoef, \
        BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix
from torchmetrics.classification import MulticlassAccuracy, MulticlassAveragePrecision, MulticlassAUROC, MulticlassMatthewsCorrCoef, \
        MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix
from collections import defaultdict
import sklearn.metrics as metrics
import numpy as np
import gc

from plots import plot_confusion_matrix_wrapper, plot_roc_prc_curves, plot_predicted_scores


class EsmFreezeUnfreeze(pl.callbacks.BaseFinetuning):
        def __init__(self, unfreeze_at_epoch=0):
                super().__init__()
                self._unfreeze_at_epoch = unfreeze_at_epoch

        def freeze_before_training(self, pl_module):
                self.freeze(pl_module.model)
                print('Model frozen')

        def finetune_function(self, pl_module, current_epoch, optimizer):
                if current_epoch == self._unfreeze_at_epoch:
                        print('Unfreezing model at epoch ', current_epoch)
                        self.unfreeze_and_add_param_group(
                                 modules=pl_module.model,
                                 optimizer=optimizer,
                                 train_bn=True,
                        )

        def setup(self, trainer, pl_module, stage):
                super().setup(trainer, pl_module, stage)
                if self._unfreeze_at_epoch == -1:
                        self.make_trainable(pl_module.model)



class AttentionPooler(nn.Module):
        def __init__(self, input_dim, pool_num_heads=1):
                super().__init__()
                self.pool_num_heads = pool_num_heads
                self.weights = nn.Linear(input_dim, pool_num_heads)
                self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
                weights = self.weights(x)
                weights = self.softmax(weights)
                weighted_x = torch.bmm(weights.transpose(1, 2), x)
                weighted_x = weighted_x.mean(dim=1)
                return weighted_x


class Predictor(pl.LightningModule):
        def __init__(self,
                                 pretrained: str = None,
                                 esm_config: dict = None,
                                 head_config: dict = None,
                                 lora_config: dict = None,
                                 train_config: dict = None,
                                 data_config: dict = None,
                                 comment: str = '',
                                 batch_size: int = 2,
                                 trial_num: int = -1,
                                 **kwargs):
                super().__init__()
                self.save_hyperparameters()

                self.train_dataset = None
                self.val_dataset = None
                self.test_dataset = None
                self.batch_size = batch_size

                if pretrained is not None and esm_config is None:
                        self.model = EsmModel.from_pretrained(pretrained)
                        self.tokenizer = EsmTokenizer.from_pretrained(pretrained)
                        esm_config = self.model.config.to_dict()
                        self.model.embeddings.token_dropout = False  # TODO
                elif esm_config is not None and pretrained is None:
                        self.model = EsmModel(EsmConfig(**esm_config))
                        self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
                        self.model.embeddings.token_dropout = False  # TODO
                # MLP only
                elif esm_config is None and pretrained is None:
                        self.model = nn.Embedding(32, head_config['embedding_dim'])
                        self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
                else:
                        raise ValueError('pretrained and esm_config cannot be specified at the same time')

                if lora_config is not None and (pretrained is not None or esm_config is not None):
                        lora_config['target_modules'] = [name.replace('base_model.model.', '')
                                                                                         for name, module in self.model.named_modules()
                                                                                         if isinstance(module, nn.Linear) and 'attention' in name]
                        self.model = get_peft_model(self.model, LoraConfig(**lora_config))
                        print('Using LoRA')
                        self.model.print_trainable_parameters()

                if head_config['pool_type'] == 'start_token':
                        self.pooler = lambda x: x  # use "pooler_output" instead of "last_hidden_state" from ESM output
                elif head_config['pool_type'] == 'mean':
                        self.pooler = lambda x: torch.mean(x, dim=1)
                elif head_config['pool_type'] == 'max':
                        self.pooler = lambda x: torch.max(x, dim=1)[0]
                elif head_config['pool_type'] == 'attention':
                        self.pooler = AttentionPooler(self.model.embeddings.word_embeddings.embedding_dim, head_config['pool_num_heads'])
                else:
                        raise ValueError(f'Invalid pool_type: {head_config["pool_type"]}')

                self.pretrained = pretrained
                self.esm_config = esm_config
                self.head_config = head_config
                self.lora_config = lora_config
                self.train_config = train_config
                self.data_config = data_config

                hidden_neurons = [head_config['hidden_neurons']] * head_config['num_hidden_layers']

                if esm_config is None and pretrained is None:
                        hidden_neurons = [self.model.embedding_dim * (data_config['max_seq_len'] + 2)] + hidden_neurons + [1]
                else:
                        hidden_neurons = [self.model.embeddings.word_embeddings.embedding_dim] + hidden_neurons + [1]
                if 'type' in head_config and head_config['type'] in ['multi_class', 'multi_label']:
                        self.classification_type = head_config['type']
                        hidden_neurons[-1] = head_config['n_classes']
                        n_class = head_config['n_classes']
                else:
                        self.classification_type = 'binary'

                layers = []
                for i in range(len(hidden_neurons) - 2):
                        layers.append(nn.Linear(hidden_neurons[i], hidden_neurons[i + 1]))
                        if head_config['batch_norm']:
                                layers.append(nn.BatchNorm1d(hidden_neurons[i + 1]))

                        if head_config['activation'] == 'relu':
                                layers.append(nn.ReLU())
                        elif head_config['activation'] == 'tanh':
                                layers.append(nn.Tanh())
                        elif head_config['activation'] == 'sigmoid':
                                layers.append(nn.Sigmoid())
                        elif head_config['activation'] == 'linear':
                                pass
                        else:
                                raise ValueError(f'Invalid activation: {head_config["activation"]}')
                        if head_config['dropout'] > 0:
                                layers.append(nn.Dropout(head_config['dropout']))

                layers.append(nn.Linear(hidden_neurons[-2], hidden_neurons[-1]))

                self.head = nn.Sequential(*layers)
                self.transform_predict = nn.Sigmoid()  # not used in training for better stability, but needed for predict
                if self.classification_type == 'multi_class':
                    self.transform_predict = nn.Softmax(dim=1)

                self.loss = nn.BCEWithLogitsLoss()
                if self.classification_type == 'multi_class':
                    self.loss = nn.CrossEntropyLoss()

                self.outputs = {'train': defaultdict(list), 'val': defaultdict(list), 'test': defaultdict(list)}

                self.evaluation = nn.ModuleDict({'accuracy': BinaryAccuracy(), 'precision': BinaryPrecision(),
                                                                                 'recall': BinaryRecall(), 'f1': BinaryF1Score(), 'auc': BinaryAUROC(),
                                                                                 'aps': BinaryAveragePrecision(), 'mcc': BinaryMatthewsCorrCoef()})
                if self.classification_type == 'multi_class':
                    self.evaluation = nn.ModuleDict({
                                    'accuracy': MulticlassAccuracy(n_class), 'precision': MulticlassPrecision(n_class),
                                    'recall': MulticlassRecall(n_class), 'f1': MulticlassF1Score(n_class), 'auc': MulticlassAUROC(n_class),
                                    'aps': MulticlassAveragePrecision(n_class), 'mcc': MulticlassMatthewsCorrCoef(n_class),
                                    'accuracy_sep': MulticlassAccuracy(n_class, average=None), 'precision_sep': MulticlassPrecision(n_class, average=None),
                                    'recall_sep': MulticlassRecall(n_class, average=None), 'f1_sep': MulticlassF1Score(n_class, average=None), 
                                    'auc_sep': MulticlassAUROC(n_class, average=None), 'aps_sep': MulticlassAveragePrecision(n_class, average=None) 
                                }) 
                self.confusion_matrix = BinaryConfusionMatrix()

        def forward(self, x, attention_mask=None):
                if self.esm_config is None and self.pretrained is None:
                        x = self.model(x)
                        x = x.flatten(start_dim=1)
                        x = self.head(x).squeeze(1)
                else:
                        x = self.model(x, attention_mask=attention_mask)
                        if self.head_config['pool_type'] == 'start_token':
                                x = x['pooler_output']
                        else:
                                x = x['last_hidden_state']
                                x = self.pooler(x)
                        x = self.head(x).squeeze(1)

                return x

        def predict(self, x, attention_mask=None):
                x = self.transform_predict(self(x, attention_mask))
                return x

        def set_train_dataset(self, train_dataset):
                self.train_dataset = train_dataset

        def train_dataloader(self):
                if self.train_dataset is None:
                        raise ValueError('train_dataset is None, please first set it using set_train_dataset()')
                return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        def set_val_dataset(self, val_dataset):
                self.val_dataset = val_dataset

        def val_dataloader(self):
                if self.val_dataset is None:
                        raise ValueError('val_dataset is None, please first set it using set_val_dataset()')
                return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        def set_test_dataset(self, test_dataset):
                self.test_dataset = test_dataset

        def test_dataloader(self):
                if self.test_dataset is None:
                        raise ValueError('test_dataset is None, please first set it using set_train_dataset()')
                return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        def on_train_start(self) -> None:
                self.logger.log_hyperparams({'num_params_backbone': sum(p.numel() for p in self.model.parameters()),
                                                                         'num_params_head': sum(p.numel() for p in self.head.parameters()),
                                                                         'batch_size': self.batch_size,})

        def basic_step(self, batch, batch_idx, mode):
                x, attention_mask, y = batch
                y_hat_logits = self(x, attention_mask)

                if self.classification_type == 'multi_label':
                    y = nn.functional.one_hot(y, num_classes=self.head_config['n_classes']).float()
                loss = self.loss(y_hat_logits, y)

                # log metrics
                y_hat = self.transform_predict(y_hat_logits)
                self.log(f'{mode}_loss', loss, on_step=False, on_epoch=True, logger=True)
                self.outputs[mode]['y_hat'].append(y_hat.detach())
                self.outputs[mode]['y'].append(y.detach())

                return loss

        def basic_epoch_end(self, mode):
                outputs = self.outputs[mode]
                y_hat = torch.cat(outputs['y_hat'])
                y = torch.cat(outputs['y'])
                for name, metric in self.evaluation.items():
                        if self.classification_type == 'multi_label':
                            scores_sum = 0. 
                            for i, binder in enumerate(self.train_dataset.spec):
                                score = metric(y_hat[:, i], y.int()[:, i])
                                scores_sum += score
                                self.log(f'{mode}_{name}_sep_{binder}', score.float(), on_step=False, on_epoch=True, logger=True,
                                         prog_bar=True if mode == 'val' else False)

                            self.log(f'{mode}_{name}', (scores_sum/(i+1)).float(), on_step=False, on_epoch=True, logger=True,
                                     prog_bar=True if mode == 'val' else False)
                            
                            continue
                        score = metric(y_hat, y.int())
                        if not name.endswith('_sep'):
                            self.log(f'{mode}_{name}', score.float(), on_step=False, on_epoch=True, logger=True,
                                             prog_bar=True if mode == 'val' else False)
                        else:
                            for binder, s in zip(self.train_dataset.spec, score):
                                self.log(f'{mode}_{name}_{binder}', s.float(), on_step=False, on_epoch=True, logger=True,
                                         prog_bar=True if mode == 'val' else False)
                                

                y_hat = y_hat.cpu()
                y = y.cpu()
                
                if self.classification_type == 'binary':
                    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)
                    optimal_threshold_auc = thresholds[np.argmax(tpr - fpr)]
                    precision, recall, thresholds = metrics.precision_recall_curve(y, y_hat)
                    optimal_threshold_prc = thresholds[np.nanargmax(2 * precision * recall / (precision + recall))]

                    self.log_dict({f'{mode}_auc_thresh': optimal_threshold_auc,
                                               f'{mode}_auc_thresh_accuracy': metrics.accuracy_score(y, y_hat > optimal_threshold_auc),
                                               f'{mode}_auc_thresh_precision': metrics.precision_score(y, y_hat > optimal_threshold_auc),
                                               f'{mode}_auc_thresh_recall': metrics.recall_score(y, y_hat > optimal_threshold_auc),
                                               f'{mode}_auc_f1_thresh': metrics.f1_score(y, y_hat > optimal_threshold_auc),
                                               f'{mode}_auc_thresh_mcc': metrics.matthews_corrcoef(y, y_hat > optimal_threshold_auc),
                                               f'{mode}_f1_thresh': optimal_threshold_prc,
                                               f'{mode}_f1_thresh_accuracy': metrics.accuracy_score(y, y_hat > optimal_threshold_prc),
                                               f'{mode}_f1_thresh_precision': metrics.precision_score(y, y_hat > optimal_threshold_prc),
                                               f'{mode}_f1_thresh_recall': metrics.recall_score(y, y_hat > optimal_threshold_prc),
                                               f'{mode}_f1_thresh_f1': metrics.f1_score(y, y_hat > optimal_threshold_prc),
                                               f'{mode}_f1_thresh_mcc': metrics.matthews_corrcoef(y, y_hat > optimal_threshold_prc)},
                                      on_step=False, on_epoch=True, logger=True,)

                    if y.mean() != 0 and y.mean() != 1:
                            plot_roc_prc_curves(fpr, tpr, precision, recall, y_hat, y, mode, self.logger.experiment, self.current_epoch)
                            plot_confusion_matrix_wrapper(y_hat, y, optimal_threshold_auc, optimal_threshold_prc, mode, self.logger.experiment, self.current_epoch)
                            plot_predicted_scores(y_hat, y, optimal_threshold_auc, optimal_threshold_prc, mode, self.logger.experiment, self.current_epoch)

                    # Prediction value distribution
                    pos_predictions = y_hat[y == 1]
                    neg_predictions = y_hat[y == 0]
                    self.log(f'{mode}_pos_preds_mean', pos_predictions.mean().float(), on_step=False, on_epoch=True, logger=True)
                    self.log(f'{mode}_pos_preds_std', pos_predictions.std().float(), on_step=False, on_epoch=True, logger=True)
                    self.log(f'{mode}_neg_preds_mean', neg_predictions.mean().float(), on_step=False, on_epoch=True, logger=True)
                    self.log(f'{mode}_neg_preds_std', neg_predictions.std().float(), on_step=False, on_epoch=True, logger=True)

                self.outputs[mode] = defaultdict(list)  # reset outputs to free memory
                torch.cuda.empty_cache()


        def training_step(self, batch, batch_idx):
                return self.basic_step(batch, batch_idx, 'train')

        def on_train_epoch_end(self):
                self.basic_epoch_end('train')

        def validation_step(self, batch, batch_idx):
                return self.basic_step(batch, batch_idx, 'val')

        def on_validation_epoch_end(self):
                self.basic_epoch_end('val')

        def test_step(self, batch, batch_idx):
                return self.basic_step(batch, batch_idx, 'test')

        def on_test_epoch_end(self):
                self.basic_epoch_end('test')

        def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=self.train_config['lr'])


if __name__ == '__main__':
        config = yaml.safe_load(open('../../config/predictor/example.yaml', 'r'))
        model = Predictor(**config)
        # model = Predictor(pretrained="Rostlab/prot_bert")
        print(model(torch.randint(0, 24, (16, 10))))
