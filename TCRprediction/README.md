# TCRprediction
This directory contains the training and evaluation code of the binding specificity models.

![Workflow](figures/sketch.png)

## Install
```
git clone https://github.com/AdrianStraub/Straub_An_et_al_2026_The-total-epitope-specific-T-cell-receptor-solution-space.git
cd Straub_An_et_al_2026_The-total-epitope-specific-T-cell-receptor-solution-space
cd TCRprediction
conda create --name tcrPrediction -y
conda activate tcrPrediction
conda install python=3.10.13
conda install nb_conda_kernels -y
pip install -r requirements.txt
```

## Reproduce results
- Download the preprocessed supplementary data and extract to `./data`
- Download models from [zenodo](XXX) and extract to `./saved_models`
- Reproducing the Performance metrics, figures, and binding values with notebook 01, 02 and 03, respectively in `./analysis`

## Training
To train the predictors a machine with GPU support is required. The hyperparameter optimization shown in our manuscript was performed on a one single A100 or H100 GPU machine with 40 or 80GB of memory, for each epitope and split invididually.

Exectute `.scripts/run_training.sh --epitope <choose_epitope> --split <choose_split>`
