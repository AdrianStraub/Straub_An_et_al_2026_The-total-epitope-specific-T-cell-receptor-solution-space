import ast

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.special import gammaln
from rapidfuzz import process, distance

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# INEXT version pinned to avoid differences in estimates due to changes in the R package.
INEXT_VERSION = '3.0.2'
ICE_CUTOFF = 10     # incidence cutoff separating "infrequent" from "frequent" species


def calculate_ice(df, col_name='CDR3ab', mouse_col='mouse', cut_off=10):
    """
    Calculates the ICE (Incidence-based Coverage Estimator) for species richness.
    ICE is generally preferred over Chao2 when there is high heterogeneity (mixture of frequent and rare species).
    
    A rough guideline for reliability: 
    1. Sample Coverage (C_ice) should be high (> 0.5).
    2. CV_gamma should not be excessively high, as it indicates extreme variation in detection probabilities.

    Input: Pandas DataFrame df[col_name] = seq, df[mouse_col] = mouse_id
    """
    # Convert to incidence matrix
    inc_matrix = pd.crosstab(df[col_name], df[mouse_col])
    inc_matrix = (inc_matrix > 0).astype(int)
    
    incidence_counts = inc_matrix.sum(axis=1)  # Vector showing in its element e the number of incidences found for species e

    # Default cutoff for rare/infrequent species is 10
    freq_mask = incidence_counts > cut_off
    infr_mask = incidence_counts <= cut_off
    
    S_freq = freq_mask.sum()
    S_infr = infr_mask.sum()
    
    if S_infr == 0:
        print("No rare species found")
        return

    Q1 = (incidence_counts == 1).sum()
    N_infr = incidence_counts[infr_mask].sum()
    
    C_ice = 1 - (Q1 / N_infr) if N_infr > 0 else 0  # Sample Coverage
    
    infr_matrix = inc_matrix[infr_mask]
    samples_with_infr = (infr_matrix.sum(axis=0) > 0).sum()  # samples_with_infr: Number of samples that contain at least one infrequent species
    
    # Gamma Formula
    sum_j_j_minus_1 = sum(c * (c - 1) for c in incidence_counts[infr_mask])
    term1 = (S_infr / C_ice)
    term2 = (samples_with_infr / (samples_with_infr - 1)) if samples_with_infr > 1 else 1
    term3 = (sum_j_j_minus_1 / (N_infr**2)) if N_infr > 0 else 0
    
    gamma_sq = (term1 * term2 * term3) - 1
    gamma_sq = max(gamma_sq, 0) # Gamma cannot be negative

    ice_est = S_freq + (S_infr / C_ice) + ((Q1 / C_ice) * gamma_sq)
    return ice_est


def rep_sat(x, bmax, kd):
    """
    Saturation function: bmax * x / (kd + x)

    Parameters
    ----------
    x     : number of cumulative repertoires
    bmax  : maximum clonotypes at saturation
    kd    : repertoire size at half-saturation
    """
    return bmax * x / (kd + x)


def _rarefaction_matrix(T):
    t = np.arange(1, T + 1)[:, None]
    k = np.arange(1, T + 1)[None, :]
    ok = t <= T - k
    A = np.zeros((T, T))
    tt, kk = t * np.ones_like(k), k * np.ones_like(t)
    A[ok] = np.exp(gammaln(T - kk[ok] + 1) + gammaln(T - tt[ok] + 1)
                   - gammaln(T - kk[ok] - tt[ok] + 1) - gammaln(T + 1))
    return A


def _ice_from_counts(Y, T_infr, cut_off=ICE_CUTOFF):
    """calculate_ice's maths, but driven by a vector of per-species incidence counts."""
    Y = Y[Y > 0]
    infr = Y <= cut_off
    Yi = Y[infr]
    N_infr, Q1 = Yi.sum(), (Y == 1).sum()
    if N_infr == 0 or Q1 == N_infr:      # C_ice would be 0 -> division by zero
        return np.nan
    C = 1 - Q1 / N_infr
    term2 = T_infr / (T_infr - 1) if T_infr > 1 else 1.0
    gamma_sq = max((infr.sum() / C) * term2 * (np.sum(Yi * (Yi - 1)) / N_infr ** 2) - 1, 0)
    return (~infr).sum() + infr.sum() / C + (Q1 / C) * gamma_sq


def _bmax_from_counts(Y, T, A, p0=(20000, 500)):
    """Refit the MM curve to the incidence rarefaction curve S_rare(t) = S_obs - A @ Q, t = 1..T."""
    Q = np.bincount(Y[Y > 0], minlength=T + 1)[1:T + 1]
    curve = (Y > 0).sum() - A @ Q
    try:
        popt_b, _ = curve_fit(rep_sat, np.arange(1, T + 1), curve, p0=p0, maxfev=10000)
    except RuntimeError:
        return np.nan, curve
    return popt_b[0], curve




def compute_inext_richness(df_tcr, valid_mouse, chains, x_lims, nboot,
                           ice_cutoff=ICE_CUTOFF, seed=42):
    """Run iNEXT + MM/ICE richness estimation for every (epitope, chain).

    `x_lims` maps epitope -> n_model (extrapolation size). Returns one long
    DataFrame with the iNEXT size-based curves plus the Chao2/MM/ICE columns.
    """
    inext = importr('iNEXT')
    _inext_loaded = ro.r('as.character(packageVersion("iNEXT"))')[0]
    if _inext_loaded != INEXT_VERSION:
        raise ImportError(f'WARNING: loaded iNEXT {_inext_loaded} != pinned {INEXT_VERSION}; estimates may differ.')

    ro.r['set.seed'](seed)
    rng = np.random.default_rng(seed)

    results_all = []
    for e, n_model in tqdm(x_lims.items(), desc="Epitope loop"):
        for chain in chains:
            print(f"Processing epitope: {e}, chain: {chain}, n_model: {n_model}")
            #### all annotated TCRs
            df_annotated_epi = df_tcr[df_tcr.annotated_specificity==e].copy()
            df_annotated_epi = df_annotated_epi[df_annotated_epi[chain].notna()].copy()

            # Mice are stored as strings of lists, so we need to convert them back to lists before exploding
            df_annotated_epi['mouse'] = df_annotated_epi['mouse'].apply(ast.literal_eval).apply(list)
            df_annotated_epi = df_annotated_epi.explode('mouse')#.reset_index(drop=True)
            df_annotated_epi['mouse'] = df_annotated_epi['mouse'].astype(str) + '_' + df_annotated_epi['isolated_specificity'].astype(str)
            df_annotated_epi = df_annotated_epi[df_annotated_epi['mouse'].isin(valid_mouse)].copy()

            incidence_matrix = pd.crosstab(df_annotated_epi[chain], df_annotated_epi['mouse'])
            incidence_matrix = (incidence_matrix > 0).astype(int)  # Makes sure that duplicate chains are counted only once per mouse

            total_units = incidence_matrix.shape[1] # Number of columns (mice)
            incidence_counts = incidence_matrix.sum(axis=1) # Row sums (how many mice had each TCR)

            T = incidence_matrix.shape[1] 
            counts = incidence_matrix.sum(axis=1).sort_values(ascending=False)
            input_data_list = [T] + counts.tolist()

            r_input = ro.FloatVector(input_data_list)
            target_sizes = ro.IntVector(range(1, n_model + 1,))
            r_output = inext.iNEXT(r_input, q=ro.FloatVector([0.0, ]), datatype="incidence_freq", size=target_sizes, nboot=nboot)
            
            r_est_data = r_output.rx2('iNextEst')
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_dataframe = r_est_data[0]
                size_based_df = ro.conversion.rpy2py(r_dataframe)

                r_asy_est = r_output.rx2('AsyEst')
                chao2_df = ro.conversion.rpy2py(r_asy_est)

            final_df = size_based_df
            
            obs = final_df[final_df['Method'].isin(['Rarefaction', 'Observed'])].copy()
            obs = obs[obs['Order.q'] == 0].copy()  # Only consider q=0 for species richness
            popt, _ = curve_fit(rep_sat, obs['t'].values, obs['qD'].values, p0=(20000, 500), maxfev=10000)

            xx = final_df['t'].values
            final_df['MM'] = rep_sat(xx, *popt)
            final_df['MM_estimate'], final_df['MM_kd'] = popt[0], popt[1]

            # Store results
            final_df['epitope'] = e
            final_df['n_model'] = n_model
            final_df['chain'] = chain

            # Rename column name
            col = 'Species Richness'
            name = 'Chao2'
            final_df[name] = chao2_df['Estimator'].loc[col]
            final_df[f'{name}.LCL'] = chao2_df['95% Lower'].loc[col]
            final_df[f'{name}.UCL'] = chao2_df['95% Upper'].loc[col]
            final_df[f'{name}_SE'] = chao2_df['Est_s.e.'].loc[col]
            final_df[f'{name}_observed'] = chao2_df['Observed'].loc[col]
            ice_point = calculate_ice(df_annotated_epi, col_name=chain, mouse_col='mouse', cut_off=ice_cutoff)
            final_df['ICE'] = ice_point

            # 95% CIs for ICE and MM, by bootstrap over mice 
            M = incidence_matrix.values
            Y_obs = incidence_counts.values
            A = _rarefaction_matrix(T)

            assert np.allclose(_bmax_from_counts(Y_obs, T, A)[1],
                               obs.sort_values('t')['qD'].values, atol=1e-6), (e, chain)
            T_infr_obs = (M[(Y_obs >= 1) & (Y_obs <= ice_cutoff)].sum(axis=0) > 0).sum()
            assert np.isclose(_ice_from_counts(Y_obs, T_infr_obs, ice_cutoff), ice_point, rtol=1e-9), (e, chain)

            ice_b, bmax_b = [], []
            for _ in range(nboot):
                c = rng.multinomial(T, np.full(T, 1 / T))   # how many times each mouse was drawn
                Y = M @ c                                   # resampled incidence counts
                infr = (Y >= 1) & (Y <= ice_cutoff)
                ice_b.append(_ice_from_counts(Y, c[M[infr].sum(axis=0) > 0].sum(), ice_cutoff))
                bmax_b.append(_bmax_from_counts(Y, T, A)[0])

            ice_sd, bmax_sd = np.nanstd(ice_b, ddof=1), np.nanstd(bmax_b, ddof=1)
            final_df['ICE.LCL'], final_df['ICE.UCL'] = ice_point - 1.96 * ice_sd, ice_point + 1.96 * ice_sd
            final_df['MM.LCL'], final_df['MM.UCL'] = popt[0] - 1.96 * bmax_sd, popt[0] + 1.96 * bmax_sd

            results_all.append(final_df)

    results_all_df = pd.concat(results_all, ignore_index=True)

    _chk = results_all_df.drop_duplicates(['epitope', 'chain'])
    assert _chk[['ICE.LCL', 'ICE.UCL', 'MM.LCL', 'MM.UCL']].notna().all().all()
    assert (_chk['ICE.LCL'] < _chk['ICE']).all() and (_chk['ICE'] < _chk['ICE.UCL']).all()
    assert (_chk['MM.LCL'] < _chk['MM_estimate']).all() and (_chk['MM_estimate'] < _chk['MM.UCL']).all()

    return results_all_df


def compute_diversity_metrics(df_tcr, epitopes, chains_reduced=('TRA', 'TRB', 'TCR', 'CDR3a', 'CDR3b', 'CDR3ab'),
                              n_trials=500, subset_size=100):
    """Mean pairwise and mean nearest-neighbour Levenshtein distance per (epitope, chain)."""
    avg_dist = {}
    min_dist_df = {}

    diversity_metrics = []
    for e in epitopes:
        for chain in tqdm(chains_reduced):
            df_epi = df_tcr[df_tcr['annotated_specificity'] == e]
            df_epi = df_epi.drop_duplicates(chain)
            df_epi = df_epi[df_epi[chain].notna()].copy()                
            
            chain_to_seq = {'TRA': 'TRA_seq', 'TRB': 'TRB_seq', 'TCR': 'TCR_seq', 'CDR3a': 'CDR3a', 'CDR3b': 'CDR3b', 'CDR3ab': 'CDR3ab',}
            ##################### Edit distance
            df_epi_ = df_epi.copy()
            col = chain_to_seq[chain]
            df_epi_ = df_epi_.dropna(subset=[col])
            df_epi_ = df_epi_.drop_duplicates(col)
            seqs = df_epi_[col].tolist()
            dist = process.cdist(seqs, seqs, scorer=distance.Levenshtein.distance, dtype=np.uint8, workers=-1,)
            upper_diag = dist[np.triu_indices(len(df_epi_), k=1)]
            
            avg_dist[f'{e}_{chain}'] = upper_diag.mean()
        
            min_dist = dist.copy()
            np.fill_diagonal(min_dist, 255)
            
            min_dist_df[f'{e}_{chain}'] = []
            for i in range(n_trials):
                rng = np.random.default_rng(seed=i)
                rand_idx = rng.choice(len(df_epi_), size=subset_size, replace=False)
                min_dist_sub = min_dist[:, rand_idx]
                min_dist_sub = min_dist_sub.min(axis=1)
                min_dist_df[f'{e}_{chain}'].append(min_dist_sub.mean())

            diversity_metrics.append({'epitope': e, 'chain': chain, 'N': df_epi.shape[0],
                                    'min_dist': np.mean(min_dist_df[f'{e}_{chain}']), 
                                    'avg_dist': avg_dist[f'{e}_{chain}'], })

    diversity_metrics = pd.DataFrame(diversity_metrics)

    # Each full-length chain pulls distances from its CDR3 counterparts
    partners = {
        'TRA': {'CDR3': 'CDR3a'},
        'TRB': {'CDR3': 'CDR3b'},
        'TCR': {'CDR3': 'CDR3ab'},
    }

    lookup = diversity_metrics.set_index(['epitope', 'chain'])[['min_dist', 'avg_dist']]

    def _dist(row, kind, which):
        src = partners.get(row['chain'], {}).get(kind)
        key = (row['epitope'], src)
        if src is None or key not in lookup.index:
            return np.nan
        return lookup.loc[key, which]

    for kind in ['CDR3']:
        diversity_metrics[f'min_dist_{kind}'] = diversity_metrics.apply(lambda r: _dist(r, kind, 'min_dist'), axis=1)
        diversity_metrics[f'avg_dist_{kind}'] = diversity_metrics.apply(lambda r: _dist(r, kind, 'avg_dist'), axis=1)

    return diversity_metrics
