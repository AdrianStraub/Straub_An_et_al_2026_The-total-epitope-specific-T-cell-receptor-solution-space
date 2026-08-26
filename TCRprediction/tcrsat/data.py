import numpy as np
import torch
import pandas as pd
import yaml
import warnings


class TCRSatDataset(torch.utils.data.Dataset):
        def __init__(self, config, split='train', tokenizer=None, positive_only=False, data=None):
                """
                Dataset for TCR Saturation data, outputs tokenized amino acid sequences (label encoding) and labels
                """
                super().__init__()
                if data is None:
                        if '.pickle' in config['path']:
                                data = pd.read_pickle(config['path'])
                        elif '.csv' in config['path']:
                                data = pd.read_csv(config['path'], index_col=0)

                max_length = data[config['data_col']].str.len().max()
                if split is not None:
                        data = data[data[config['split_col']] == split].copy()
                if positive_only:
                        data = data[data['specificity'] != 'not_identified']

                if tokenizer is not None:
                        seq_data = tokenizer(data[config['data_col']].to_list(), padding='max_length', truncation=True,
                                                                 add_special_tokens=True, max_length=max_length+2)  # need to add 2 for < and >
                        data['seq_data'] = seq_data['input_ids']
                        data['attention_mask'] = seq_data['attention_mask']
                else:
                        unique_aa = ['<', '-', '>'] + sorted(set(''.join(data['TRA+TRB'].to_list())))  # padding token + unique amino acids
                        self.aa2label = dict(zip(unique_aa, range(len(unique_aa))))
                        data['seq_data'] = '<' + data['TRA+TRB'] + '>'
                        data['seq_data'] = data['seq_data'].apply(lambda x: np.array([self.aa2label[c] for c in [*x]]))
                        data['seq_data'] = data['seq_data'].apply(
                                lambda x: np.pad(x, (0, max(0, config['max_seq_len'] - x.shape[0])), 'constant', constant_values=0)[
                                                  :config['max_seq_len']].T)
                        data['attention_mask'] = data['seq_data'].apply(
                                lambda x: np.pad(x, (0, max(0, config['max_seq_len'] - x.shape[0])), 'constant', constant_values=0)[
                                                  :config['max_seq_len']].T)
                        data['attention_mask'] = data['seq_data'].apply(lambda x: np.where(x == 1, 0, 1))
                
                self.spec = data['specificity'].unique().tolist()
                if len(self.spec) <= 2:
                    if 'not_identified' in self.spec:
                        self.spec.remove('not_identified')
                    data['label'] = (data['specificity'] == self.spec[0]).astype(float)
                else:
                    data['specificity'] = pd.Categorical(data['specificity'])
                    if 'not_identified' in self.spec:
                        self.spec.remove('not_identified')
                        self.spec = ['not_identified'] + sorted(self.spec) 
                        data['specificity'] = data['specificity'].cat.reorder_categories(self.spec)
                    data['label'] = data['specificity'].cat.codes

                if config['sample_size'] is not None:
                        sample_size = config['sample_size']
                        if sample_size > len(data):
                                warnings.warn(f'Selected sample_size {sample_size} is greater then length of dataset {len(data)}. Default to dataset size.')
                                sample_size = len(data)
                        data = data.sample(sample_size, random_state=42).reset_index(drop=True)

                self.x = torch.tensor(data['seq_data'].to_list())
                self.attention_mask = torch.tensor(data['attention_mask'].to_list())
                if len(self.spec) > 2:
                    self.y = torch.tensor(data['label'].values).to(torch.int64)
                else:
                    self.y = torch.tensor(data['label'].values)

        def __len__(self):
                return len(self.x)

        def __getitem__(self, idx):
                return self.x[idx], self.attention_mask[idx], self.y[idx]


if __name__ == '__main__':
        from transformers import EsmTokenizer
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        config = yaml.safe_load(open('../config/predictor/example.yaml', 'r'))
        dataset = TCRSatDataset(config=config['data_config'], split='train', tokenizer=tokenizer)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        for i, batch in enumerate(dataloader):
                if (i == 0) or (i == 100) or (i == 223):
                        print(batch)
