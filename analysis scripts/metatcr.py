import pandas as pd
import numpy as np
import math
from collections import Counter


def _greedy_centered_chain(df, spec_col, chain_col, id_col, seed_col, id_counter):
    """Greedy, frequency-centered HD<=1 clustering of one chain (TRA or TRB).

    Uses the aggregated "V_J_CDR3" string in `chain_col`. Blocks are
    (epitope, V+J). Returns df with `id_col` + `seed_col` added and the next
    free integer id.
    """
    s = df[chain_col].astype(str)
    vj = s.str.rsplit('_', n=1).str[0]           # V+J key
    cdr3 = s.str.rsplit('_', n=1).str[1]          # CDR3 aa sequence

    work = pd.DataFrame({'spec': df[spec_col].values, 'vj': vj.values,
                         'cdr3': cdr3.values})
    # clone frequency = number of rows carrying each unique chain string
    freq = s.value_counts()
    work['freq'] = s.map(freq).values

    block_cols = ['spec', 'vj']
    # unique (block, cdr3) with their frequency, most frequent first
    uniq = (work.groupby(block_cols + ['cdr3'], as_index=False)['freq'].max()
                .sort_values('freq', ascending=False))

    key_to_id = {}      # (block..., cdr3) -> cluster id
    id_to_seed = {}     # cluster id -> center CDR3

    for block_vals, block in uniq.groupby(block_cols, sort=False):
        bk = block_vals if isinstance(block_vals, tuple) else (block_vals,)
        seqs = block['cdr3'].tolist()
        wild = {}
        for seq in seqs:
            for k in range(len(seq)):
                wild.setdefault(seq[:k] + '*' + seq[k + 1:], []).append(seq)
        assigned = set()
        for center in seqs:
            if center in assigned:
                continue
            cid = id_counter
            id_counter += 1
            id_to_seed[cid] = center
            # claim every still-unassigned CDR3 within HD<=1 of the center
            for k in range(len(center)):
                for cand in wild[center[:k] + '*' + center[k + 1:]]:
                    if cand not in assigned:
                        assigned.add(cand)
                        key_to_id[bk + (cand,)] = cid

    keys = list(zip(work['spec'], work['vj'], work['cdr3']))
    df[id_col] = pd.Series([key_to_id.get(k) for k in keys], index=df.index).astype('Int64')
    df[seed_col] = df[id_col].map(id_to_seed)
    return df, id_counter


def define_metaTCR(df, spec_col='annotated_specificity',
                   tra_col='TRA', trb_col='TRB'):
    counter = 1
    df, counter = _greedy_centered_chain(df, spec_col, tra_col,
                                         'metaTCRa', 'metaTCRa_seed', counter)
    df, counter = _greedy_centered_chain(df, spec_col, trb_col,
                                         'metaTCRb', 'metaTCRb_seed', counter)

    # Paired metaTCR = combination of alpha-meta x beta-meta,
    both = df['metaTCRa'].notna() & df['metaTCRb'].notna()
    pair_key = (df[spec_col].astype(str) + '|'
                + df['metaTCRa'].astype(str) + '|' + df['metaTCRb'].astype(str))
    codes = pd.Series(pd.factorize(pair_key.where(both))[0], index=df.index)
    df['metaTCRab'] = (codes + 1).where(both).astype('Int64')   # 1-based ids
    df['metaTCRab_seed'] = (df['metaTCRa_seed'] + '-' + df['metaTCRb_seed']).where(both)

    return df
