# The epitope-specific T cell receptor solution space utilizes the entire set of germline-encoded V(D)J genes
This repository contains the code to reproduce the results from the manuscript
**The epitope-specific T cell receptor solution space utilizes the entire set
of germline-encoded V(D)J genes**.

The **TCR specificity predictor** code and README is separated into
`TCRprediction/`, as it contains a different suite of libraries. All
descriptions in this README is only for the main repo, in particular for the
three notebooks.

## Steps to reproduce the results and figures

### 1. Data

The TCR datasets are part of this submission and are not yet publicly released.
Reviewers and editors can access them through the supplementary material of the
submitted manuscript (Supplementary Tables 2-5). Download the supplementary
tables and extract them into `tcr_data/` directory at the root of this
repository, keeping the original file names:

The IMGT and Cell Ranger reference files the notebooks use are already included
in `10X_mouse_vdj/`.

### 2. Installation

```bash
git clone https://github.com/AdrianStraub/Straub_An_et_al_2026_The-total-epitope-specific-T-cell-receptor-solution-space.git
cd Straub_An_et_al_2026_The-total-epitope-specific-T-cell-receptor-solution-space
conda env create -f environment.yml
```

Tested on macOS 26.6 (Apple Silicon). A standard desktop or laptop is
sufficient, no GPU is required. All dependencies and their pinned versions are
declared in `environment.yml`. Installing takes 2-5 minutes on a normal desktop
computer. On macOS with Apple Silicon, iNEXT needs one additional install step,
see the first cell of notebook 03.

### 3. Notebooks

Run the notebooks with the installed `tcr` kernel. The notebooks are
independent of each other and can be run in any order.

- **`01_manuscript_figures.ipynb`**: V(D)J gene usage, Gini indices, CDR3 k-mer
  entropy, generation probabilities and germline sharedness across the OVA,
  GP33 and M45 repertoires; produces Figures 1-5.
- **`02_GLIPH2_kmer_analysis.ipynb`**: local CDR3 motif enrichment against the
  antigen-unselected naive repertoire.
- **`03_richness_estimates.ipynb`**: metaTCR definition and total
  epitope-specific TCR richness estimated with iNEXT (Chao2, ICE and
  Michaelis-Menten). The recomputation with iNEXT at `NBOOT=100` can take 10 hours. For a faster run, please set `NBOOT=3`, which should produce identical point estimates and different (more unreliable) confidence intervals.
