import numpy as np
import pandas as pd
from typing import Literal
from Bio.Seq import Seq
from Bio.Data import CodonTable
import networkx as nx
import parmap
from tcrdist.repertoire import TCRrep
from tcrdist.pgen import OlgaModel
import networkx as nx


    
def preprocess_tcr_vj_vdj(df_obs: pd.DataFrame) -> pd.DataFrame:
    
    """
    Preprocesses a single cell obs DataFrame (cellranger -> scanpy -> scirpy processed) by reassigning 'extra VDJ' and 'extra VJ' chain pairings
    based on single-pair lookup sets, and updating corresponding columns.
    Extra VJ and Extra VDJ clonotypes are checked: when a TCR pair was also identified as a single pair TCR (in a different cell)
    then the extra chain is dropped, and the value is set to "assigned_single_pair", and the VJ_1/VJ_2 (VDJ_1/VDJ_2) colums are updated
    to contain the new values only in VJ_1/VDJ_1. If no match was found as a single pair, all original information is retained.

    Adds column "TCR": consistent clonotype annotation based on V/J/CDR3aa of TRA + _ + TRB
    "TCR_expansion": Calculates clone size per mouse (amino acid)
    """
    
    df = df_obs.copy()

    # === Step 1: Build set of full TCR strings from single-pair rows ===
    single_mask = df['chain_pairing'] == 'single pair'

    df['__TRA_key'] = df['IR_VJ_1_v_call'].astype(str) + "_" + df['IR_VJ_1_j_call'].astype(str) + "_" + df['IR_VJ_1_junction_aa'].astype(str)
    df['__TRB_key'] = df['IR_VDJ_1_v_call'].astype(str) + "_" + df['IR_VDJ_1_j_call'].astype(str) + "_" + df['IR_VDJ_1_junction_aa'].astype(str)
    df['__fullTCR_single'] = df['__TRA_key'] + "::" + df['__TRB_key']
    fullTCR_single_set = set(df.loc[single_mask, '__fullTCR_single'].unique())

    # === Step 2: Create candidate keys for switching ===
    df['__TRA_cand1'] = df['IR_VJ_1_v_call'].astype(str) + "_" + df['IR_VJ_1_j_call'].astype(str) + "_" + df['IR_VJ_1_junction_aa'].astype(str)
    df['__TRA_cand2'] = df['IR_VJ_2_v_call'].astype(str) + "_" + df['IR_VJ_2_j_call'].astype(str) + "_" + df['IR_VJ_2_junction_aa'].astype(str)
    df['__TRB_cand1'] = df['IR_VDJ_1_v_call'].astype(str) + "_" + df['IR_VDJ_1_j_call'].astype(str) + "_" + df['IR_VDJ_1_junction_aa'].astype(str)
    df['__TRB_cand2'] = df['IR_VDJ_2_v_call'].astype(str) + "_" + df['IR_VDJ_2_j_call'].astype(str) + "_" + df['IR_VDJ_2_junction_aa'].astype(str)

    # === Step 3: Define pickers ===
    def pick_TRA_if_fullTCR(row):
        trb1 = row['__TRB_cand1']
        if row['__TRA_cand1'] + "::" + trb1 in fullTCR_single_set:
            return row['__TRA_cand1']
        if row['__TRA_cand2'] + "::" + trb1 in fullTCR_single_set:
            return row['__TRA_cand2']
        return np.nan

    def pick_TRB_if_fullTCR(row):
        tra1 = row['__TRA_cand1']
        if tra1 + "::" + row['__TRB_cand1'] in fullTCR_single_set:
            return row['__TRB_cand1']
        if tra1 + "::" + row['__TRB_cand2'] in fullTCR_single_set:
            return row['__TRB_cand2']
        return np.nan

    # === Step 4: Apply pickers to extra chains ===
    mask_extra_vj = df['chain_pairing'] == 'extra VJ'
    mask_extra_vdj = df['chain_pairing'] == 'extra VDJ'

    df.loc[mask_extra_vj, 'assigned_TRA'] = df.loc[mask_extra_vj].apply(pick_TRA_if_fullTCR, axis=1)
    df.loc[mask_extra_vdj, 'assigned_TRB'] = df.loc[mask_extra_vdj].apply(pick_TRB_if_fullTCR, axis=1)

    # Step 5: Mark rows as assigned_single_pair, exclude original 'single pair'
    mask_assigned_vj = mask_extra_vj & df['assigned_TRA'].notna()
    mask_assigned_vdj = mask_extra_vdj & df['assigned_TRB'].notna()

    mask_to_assign = (mask_assigned_vj | mask_assigned_vdj) & (df['chain_pairing'] != 'single pair')
    df.loc[mask_to_assign, 'chain_pairing'] = 'assigned_single_pair'

    # === Step 6: Swap _2 → _1 if _2 was assigned ===
    vdj_fields = ['v_call', 'd_call', 'j_call', 'junction_aa', 'junction']
    vj_fields = ['v_call', 'j_call', 'junction_aa', 'junction']

    mask_matched_vdj2 = mask_assigned_vdj & (df['assigned_TRB'] == df['__TRB_cand2'])
    mask_matched_vj2 = mask_assigned_vj & (df['assigned_TRA'] == df['__TRA_cand2'])

    for suf in vdj_fields:
        col1 = f'IR_VDJ_1_{suf}'
        col2 = f'IR_VDJ_2_{suf}'
        df.loc[mask_matched_vdj2, col1] = df.loc[mask_matched_vdj2, col2]
        df.loc[mask_matched_vdj2, col2] = np.nan

    for suf in vj_fields:
        col1 = f'IR_VJ_1_{suf}'
        col2 = f'IR_VJ_2_{suf}'
        df.loc[mask_matched_vj2, col1] = df.loc[mask_matched_vj2, col2]
        df.loc[mask_matched_vj2, col2] = np.nan

    # === Step 7: Clean helper columns ===
    df.drop(columns=[
        '__TRA_key', '__TRB_key', '__fullTCR_single',
        '__TRA_cand1', '__TRA_cand2', '__TRB_cand1', '__TRB_cand2'
    ], inplace=True)

    # === Step 8: Compute final TRA/TRB and TCR ===
    df['TRB'] = df['IR_VDJ_1_v_call'] + "_" + df['IR_VDJ_1_j_call'] + "_" + df['IR_VDJ_1_junction_aa']
    df['TRA'] = df['IR_VJ_1_v_call'] + "_" + df['IR_VJ_1_j_call'] + "_" + df['IR_VJ_1_junction_aa']
    df['TCR'] = df['TRA'] + "__" + df['TRB']

    df['TCR_expansion'] = df.groupby(['mouse', 'TCR'], observed=True)['TCR'].transform('count')
    df['TCR_expansion_full_data'] = df.groupby('TCR', observed=True)['TCR'].transform('count')

    # === Step 9: Construct TRA2 and TRB2 for dual TCR ===
    df['TRA2'] = df['IR_VJ_2_v_call'].astype(str) + "_" + df['IR_VJ_2_j_call'].astype(str) + "_" + df['IR_VJ_2_junction_aa'].astype(str)
    df['TRB2'] = df['IR_VDJ_2_v_call'].astype(str) + "_" + df['IR_VDJ_2_j_call'].astype(str) + "_" + df['IR_VDJ_2_junction_aa'].astype(str)

    df['TRA2'] = df['TRA2'].replace("nan_nan_nan", np.nan)
    df['TRB2'] = df['TRB2'].replace("nan_nan_nan", np.nan)

    # === Step 10: Construct dual_tcr ===
    df['dual_tcr'] = df['TRA'] + "__" + df['TRB']

    mask_extra_vj = df['chain_pairing'] == 'extra VJ'
    mask_extra_vdj = df['chain_pairing'] == 'extra VDJ'
    mask_two_full = df['chain_pairing'] == 'two full chains'
    mask_assigned = df['chain_pairing'] == 'assigned_single_pair'

    df.loc[mask_extra_vj, 'dual_tcr'] = df.loc[mask_extra_vj, 'TRA'] + "__" + df.loc[mask_extra_vj, 'TRB'] + "__" + df.loc[mask_extra_vj, 'TRA2']
    df.loc[mask_extra_vdj, 'dual_tcr'] = df.loc[mask_extra_vdj, 'TRA'] + "__" + df.loc[mask_extra_vdj, 'TRB'] + "__" + df.loc[mask_extra_vdj, 'TRB2']
    df.loc[mask_two_full, 'dual_tcr'] = df.loc[mask_two_full, 'TRA'] + "__" + df.loc[mask_two_full, 'TRB'] + "__" + df.loc[mask_two_full, 'TRA2'] + "__" + df.loc[mask_two_full, 'TRB2']

    # assigned_single_pair with remaining TRA2 or TRB2
    df.loc[mask_assigned & df['TRA2'].notna(), 'dual_tcr'] = df.loc[mask_assigned & df['TRA2'].notna(), 'TRA'] + "__" + df.loc[mask_assigned & df['TRA2'].notna(), 'TRB'] + "__" + df.loc[mask_assigned & df['TRA2'].notna(), 'TRA2']
    df.loc[mask_assigned & df['TRB2'].notna(), 'dual_tcr'] = df.loc[mask_assigned & df['TRB2'].notna(), 'TRA'] + "__" + df.loc[mask_assigned & df['TRB2'].notna(), 'TRB'] + "__" + df.loc[mask_assigned & df['TRB2'].notna(), 'TRB2']

    # === Step 11: Calculate single cell size per mouse ===
    df["n_cells_mouse"] = df.groupby("mouse")["mouse"].transform("count")

    return df


if __name__ == '__main__':
    # Example usage:
    # import pandas as pd
    # df_obs = pd.read_csv('input.csv')
    # df_processed = preprocess_tcr_vj_vdj(df_obs)
    # df_processed.to_csv('output.csv', index=False)
    pass
    
    
def translate_nt_to_aa(
    nt_seq: str,
    trim_to_codon: bool = True,
    to_stop: bool = False,
    table: str = "Standard"
) -> str:
    """
    Translate a nucleotide sequence (nt_seq) into amino acids.
    
    Parameters:
    -----------
    nt_seq : str
        The input nucleotide sequence (can include lowercase, whitespace, or newline).
    trim_to_codon : bool, default True
        If True, drop any extra bases at the end so that len(nt_seq) % 3 == 0.
        If False and length is not a multiple of 3, Biopython will raise an error.
    to_stop : bool, default False
        If True, translation will stop at the first in‐frame stop codon (won’t include "*").
        If False, any internal stop codons become "*" in the output.
    table : str or int, default "Standard"
        Which codon table to use (e.g. "Standard", "Vertebrate Mitochondrial", or an NCBI table number).
    
    Returns:
    --------
    str
        The translated amino‐acid sequence.
    
    Raises:
    -------
    ValueError
        If the input contains invalid characters or if trim_to_codon is False and length % 3 != 0.
    """
    # 1. Clean input: remove whitespace/newlines, uppercase
    seq_clean = "".join(nt_seq.split()).upper()
    
    # 2. Check for valid nucleotides
    valid_nt = set("ATGCU")  # allow U/T interchangeably
    if any(ch not in valid_nt for ch in seq_clean):
        bad = sorted({ch for ch in seq_clean if ch not in valid_nt})
        raise ValueError(f"Invalid base(s) found: {bad}. Sequence should only contain A/T/G/C (or U).")
    
    # 3. (Optional) Convert U → T for DNA translation
    seq_clean = seq_clean.replace("U", "T")
    
    # 4. Trim trailing bases if needed
    if trim_to_codon and (len(seq_clean) % 3 != 0):
        trim_len = len(seq_clean) - (len(seq_clean) % 3)
        seq_clean = seq_clean[:trim_len]
    elif not trim_to_codon and (len(seq_clean) % 3 != 0):
        raise ValueError("Sequence length is not a multiple of 3; set trim_to_codon=True if you want to auto‐trim.")
    
    # 5. Perform translation
    seq_obj = Seq(seq_clean)
    try:
        aa_seq = seq_obj.translate(table=table, to_stop=to_stop)
    except CodonTable.TranslationError as e:
        raise ValueError(f"Translation error: {e}")
    
    return str(aa_seq)


def stitch_nt_from_cellranger(
    df: pd.DataFrame,
    v_info: pd.DataFrame,
    j_info: pd.DataFrame,
    v_gene_col: str,
    j_gene_col: str,
    cdr3_col: str,
    c_col: str,
    chain: Literal["alpha", "beta"],
) -> pd.DataFrame:
    """
    For each row in df, look up leader and framework/CDR segments from v_info and j_info,
    then concatenate to build the full V(D)J transcript, constant region, and amino acid translation.
    """
    
    if chain not in ("alpha", "beta"):
        raise ValueError(f"Invalid chain {chain!r}; must be 'alpha' or 'beta'")
    
    df_out = df.copy()

    # 1) Drop overlapping columns before merging v_info
    v_overlap = [
        "v_gene", "leader", "fwr1_nt", "cdr1_nt", "fwr2_nt", "cdr2_nt", "fwr3_nt",
        "fwr1", "cdr1", "fwr2", "cdr2", "fwr3"
    ]
    df_out.drop(columns=[col for col in v_overlap if col in df_out.columns], inplace=True)

    # 2) Merge v_info and rename
    df_out = df_out.merge(
        v_info,
        left_on=v_gene_col,
        right_on="v_gene",
        how="left"
    )
    df_out.rename(columns={
        "leader":  f"leader_{chain}",
        "fwr1_nt": f"fwr1_nt_{chain}",
        "cdr1_nt": f"cdr1_nt_{chain}",
        "fwr2_nt": f"fwr2_nt_{chain}",
        "cdr2_nt": f"cdr2_nt_{chain}",
        "fwr3_nt": f"fwr3_nt_{chain}",
        "fwr1":    f"fwr1_{chain}",
        "cdr1":    f"cdr1_{chain}",
        "fwr2":    f"fwr2_{chain}",
        "cdr2":    f"cdr2_{chain}",
        "fwr3":    f"fwr3_{chain}",
    }, inplace=True)

    # 3) Drop overlapping columns before merging j_info
    j_overlap = ["j_gene", "fwr4_nt", "fwr4"]
    df_out.drop(columns=[col for col in j_overlap if col in df_out.columns], inplace=True)

    # 4) Merge j_info and rename
    df_out = df_out.merge(
        j_info,
        left_on=j_gene_col,
        right_on="j_gene",
        how="left"
    )
    df_out.rename(columns={
        "fwr4_nt": f"fwr4_nt_{chain}",
        "fwr4":    f"fwr4_{chain}"
    }, inplace=True)

    # 5) Fill NaNs
    needed_cols = [
        f"leader_{chain}",
        f"fwr1_nt_{chain}", f"cdr1_nt_{chain}",
        f"fwr2_nt_{chain}", f"cdr2_nt_{chain}",
        f"fwr3_nt_{chain}",
        cdr3_col,
        f"fwr4_nt_{chain}"
    ]
    for col in needed_cols:
        if col not in df_out.columns:
            df_out[col] = ""
        df_out[col] = df_out[col].fillna("")

    # 6) Build full sequence
    concat_cols = [
        f"leader_{chain}",
        f"fwr1_nt_{chain}", f"cdr1_nt_{chain}",
        f"fwr2_nt_{chain}", f"cdr2_nt_{chain}",
        f"fwr3_nt_{chain}",
        cdr3_col,
        f"fwr4_nt_{chain}"
    ]
    df_out[f"cellranger_{chain}_nt"] = df_out[concat_cols].agg("".join, axis=1)

    # 7) Add constant regions
    mTRAC = 'ACATCCAGAACCCAGAACCTGCTGTGTACCAATTGAAAGATCCTCGGTCTCAGGACAGCACCCTCTGCCTGTTCACCGACTTTGACTCCCAAATCAATGTGCCGAAAACAATGGAATCTGGAACGTTCATCACTGACAAAACTGTGCTGGACATGAAAGCTATGGATTCCAAGAGCAATGGAGCTATTGCCTGGAGCAACCAGACAAGCTTCACCTGCCAAGATATCTTCAAAGAGACCAACGCCACCTACCCCAGTTCAGACGTTCCCTGTGATGCCACGTTGACTGAGAAAAGCTTTGAAACAGATATGAACCTAAACTTTCAAAACCTGTCAGTTATGGGACTCCGAATCCTCCTGCTGAAAGTAGCCGGATTTAACCTGCTCATGACGCTGAGGCTGTGGTCCAGT'
    mTRBC1 = 'AGGATCTGAGAAATGTGACTCCACCCAAGGTCTCCTTGTTTGAGCCATCAAAAGCAGAGATTGCAAACAAACAAAAGGCTACCCTCGTGTGCTTGGCCAGGGGCTTCTTCCCTGACCACGTGGAGCTGAGCTGGTGGGTGAATGGCAAGGAAGTCCACAGTGGGGTCAGCACGGACCCTCAGGCCTACAAGGAGAGCAATTATAGCTACTGCCTGAGCAGCCGCCTGAGGGTCTCTGCTACCTTCTGGCACAATCCTCGCAACCACTTCCGCTGCCAAGTGCAGTTCCACGGGCTTTCAGAGGAGGACAAGTGGCCAGAGGGCTCACCCAAACCTGTCACACAGAACATCAGTGCAGAGGCCTGGGGCCGAGCAGACTGTGGGATTACCTCAGCATCCTATCAACAAGGGGTCTTGTCTGCCACCATCCTCTATGAGATCCTGCTAGGGAAAGCCACCCTGTATGCTGTGCTTGTCAGTACACTGGTGGTGATGGCTATGGTCAAAAGAAAGAACTCA'
    mTRBC2 = 'AGGATCTGAGAAATGTGACTCCACCCAAGGTCTCCTTGTTTGAGCCATCAAAAGCAGAGATTGCAAACAAACAAAAGGCTACCCTCGTGTGCTTGGCCAGGGGCTTCTTCCCTGACCACGTGGAGCTGAGCTGGTGGGTGAATGGCAAGGAAGTCCACAGTGGGGTCAGCACGGACCCTCAGGCCTACAAGGAGAGCAATTATAGCTACTGCCTGAGCAGCCGCCTGAGGGTCTCTGCTACCTTCTGGCACAATCCTCGAAACCACTTCCGCTGCCAAGTGCAGTTCCACGGGCTTTCAGAGGAGGACAAGTGGCCAGAGGGCTCACCCAAACCTGTCACACAGAACATCAGTGCAGAGGCCTGGGGCCGAGCAGACTGTGGAATCACTTCAGCATCCTATCATCAGGGGGTTCTGTCTGCAACCATCCTCTATGAGATCCTACTGGGGAAGGCCACCCTATATGCTGTGCTGGTCAGTACCCTGGTGCTGATGGCTATGGTCAAGAAAAAAAATTCC'

    if chain == 'alpha':
        df_out[f"cellranger_{chain}_const_nt"] = df_out[f"cellranger_{chain}_nt"] + mTRAC
    else:  # beta
        df_out[f"cellranger_{chain}_const_nt"] = ""
        mask1 = df[c_col] == "TRBC1"
        df_out.loc[mask1, f"cellranger_{chain}_const_nt"] = (
            df_out.loc[mask1, f"cellranger_{chain}_nt"] + mTRBC1
        )
        mask2 = df[c_col] == "TRBC2"
        df_out.loc[mask2, f"cellranger_{chain}_const_nt"] = (
            df_out.loc[mask2, f"cellranger_{chain}_nt"] + mTRBC2
        )

    # 8) Translate to amino acid sequences
    df_out[f"cellranger_{chain}_aa"] = df_out[f"cellranger_{chain}_nt"].apply(translate_nt_to_aa)
    df_out[f"cellranger_{chain}_const_aa"] = df_out[f"cellranger_{chain}_const_nt"].apply(translate_nt_to_aa)
    
    return df_out
    
    
def define_tcrdist_clusters_any_v(
    clone_df,
    df_dist,
    chain,
    cutoff,
    sort_by_size=True,
    seq_col = None,
):
    """
    Adds a column 'cc_aa_cdr3{chain}' to clone_df with integer cluster IDs.
    
    Parameters
    ----------
    clone_df : pd.DataFrame
        Must contain a column f'cdr3_{chain}_aa' listing each sequence.
    df_dist : pd.DataFrame
        Square distance matrix, indexed & columned by the same CDR3 strings.
    chain : str
        'a' or 'b'.
    cutoff : float
        Distance threshold to draw edges.
    sort_by_size : bool
        If True, cluster 1 is the largest component, etc.
    
    Returns
    -------
    clone_df : pd.DataFrame
        Same as input but with an extra int column f'cc_aa_cdr3{chain}'.
    """
    if seq_col is None:
        seq_col = f'cdr3_{chain}_aa'
        
    else:
        seq_col = seq_col
      
    
    new_col = f'cc_aa_cdr3{chain}'
    
    # 1) build the graph
    G = nx.Graph()
    seqs = df_dist.index.tolist()
    G.add_nodes_from(seqs)
    
    # 2) add edges for pairs ≤ cutoff
    #    we can speed up by only looping i<j
    dist_mat = df_dist.values
    n = len(seqs)
    for i in range(n):
        for j in range(i+1, n):
            if dist_mat[i, j] <= cutoff:
                G.add_edge(seqs[i], seqs[j])
    
    # 3) get connected components
    comps = list(nx.connected_components(G))
    
    # 4) optionally sort by component size (largest first)
    if sort_by_size:
        comps.sort(key=len, reverse=True)
    
    # 5) assign cluster IDs
    seq2clust = {}
    for cid, comp in enumerate(comps, start=1):
        for seq in comp:
            seq2clust[seq] = cid
    
    # 6) map back into clone_df
    clone_df[new_col] = clone_df[seq_col].map(seq2clust).fillna(0).astype(int)
    
    print(f"Assigned {len(seq2clust)} sequences to clusters")
    print(f"Unique cluster IDs: {clone_df[new_col].unique()}")
    print(f"Connected components found: {len(comps)}")
    print(f"First few component sizes: {[len(c) for c in comps[:5]]}")
    
    
    return clone_df
    

def format_for_tcrdist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame `df` containing at least the columns:
      ['tcr_id', 'mouse', 'va', 'ja', 'cdr3a_aa', 'cdr3a_nt',
       'vb', 'jb', 'cdr3b_aa', 'cdr3b_nt', 'clone_size_mouse', 'specificity'],
    this function will:
      1. Group by 'tcr_id' and aggregate the necessary fields.
      2. Force all columns to object dtype (required by TCRdist).
      3. Append "*01" to each V/J gene call.
      4. Replace certain α‐chain V calls so that they match IMGT formatting.
      5. Reset index and reorder columns to the TCRdist‐compatible format.
    Returns a new DataFrame ready for use with TCRdist.
    """
    # 1) Aggregate by tcr_id
    df_clonotypes_all = df.groupby('tcr_id').agg(
        subject       = ('mouse',           'first'),
        v_a_gene      = ('va',              'first'),
        j_a_gene      = ('ja',              'first'),
        cdr3_a_aa     = ('cdr3a_aa',        'first'),
        cdr3a_nt      = ('cdr3a_nt',        'first'),
        v_b_gene      = ('vb',              'first'),
        j_b_gene      = ('jb',              'first'),
        cdr3_b_aa     = ('cdr3b_aa',        'first'),
        cdr3b_nt      = ('cdr3b_nt',        'first'),
        clone_id      = ('tcr_id',          'first'),
        epitope       = ('annotated_specificity',     'first'),
        count         = ('clone_size_mouse','max'),
        count_by_mouse= ('clone_size_mouse','max'),
    )

    # 2) Force subject column to a single value (required by TCRdist)
    df_clonotypes_all["subject"] = "mouse"

    # 3) Keep original tcr_id as 'clonotype_origin'
    df_clonotypes_all['clonotype_origin'] = df_clonotypes_all.index

    # 4) Make every column object dtype
    df_clonotypes_all = df_clonotypes_all.astype(object)

    # 5) Append "*01" to each V/J gene call
    df_clonotypes_all['v_a_gene'] = df_clonotypes_all['v_a_gene'].apply(lambda x: f"{x}*01")
    df_clonotypes_all['j_a_gene'] = df_clonotypes_all['j_a_gene'].apply(lambda x: f"{x}*01")
    df_clonotypes_all['v_b_gene'] = df_clonotypes_all['v_b_gene'].apply(lambda x: f"{x}*01")
    df_clonotypes_all['j_b_gene'] = df_clonotypes_all['j_b_gene'].apply(lambda x: f"{x}*01")

    # 6) Fix known IMGT formatting issues for certain α‐chain V calls
    replacements = {
        'TRAV4-4-DV10*01'     : 'TRAV4-4/DV10*01',
        'TRAV14D-3-DV8*01'    : 'TRAV14D-3/DV8*01',
        'TRAV16D-DV11*01'     : 'TRAV16D/DV11*01',
        'TRAV6-7-DV9*01'      : 'TRAV6-7/DV9*01',
        'TRAV13-4-DV7*01'     : 'TRAV13-4/DV7*01',
        'TRAV15D-2-DV6D-2*01' : 'TRAV15D-2/DV6D-2*01',
        'TRAV15-2-DV6-2*01'   : 'TRAV15-2/DV6-2*01',
        'TRAV15-1-DV6-1*01'   : 'TRAV15-1/DV6-1*01',
        'TRAV15D-1-DV6D-1*01' : 'TRAV15D-1/DV6D-1*01',
        'TRAV21-DV12*01'      : 'TRAV21/DV12*01',
    }
    for old, new in replacements.items():
        df_clonotypes_all['v_a_gene'] = df_clonotypes_all['v_a_gene'].replace(old, new)

    # 7) Apply any known β‐chain V call fixes (example shown)
    df_clonotypes_all['v_b_gene'] = df_clonotypes_all['v_b_gene'].replace(
        'TRBV12-2+TRBV13-2*01', 'TRBV13-2*01'
    )

    # 8) Reset index and keep only the columns TCRdist expects, in the correct order
    df_clonotypes_all = (
        df_clonotypes_all
        .reset_index(drop=True)
        [['subject',    # always "mouse"
          'epitope',    # originally 'specificity'
          'cdr3_a_aa',
          'v_a_gene',
          'j_a_gene',
          'cdr3_b_aa',
          'v_b_gene',
          'j_b_gene',
          'cdr3a_nt',
          'cdr3b_nt',
          'count',
          'clonotype_origin',
          'count_by_mouse'
         ]]
        .copy()
    )

    return df_clonotypes_all
    
    
##### custom hamming distance for tcrdist3
def trimmed_hamming(
    s1,
    s2,
    ntrim=2,
    ctrim=3,
    unequal_length_distance=9999,
    **kwargs,
):
    """
    Hamming distance after trimming fixed numbers of residues
    from the N- and C-termini.

    Distances:
        0 = identical trimmed sequences
        1 = one AA mismatch
        2 = two AA mismatches
        ...

    Sequences with different lengths after trimming are assigned
    a large distance because strict Hamming distance is only defined
    for equal-length sequences.
    """

    if not isinstance(s1, str) or not isinstance(s2, str):
        return unequal_length_distance

    end1 = -ctrim if ctrim > 0 else None
    end2 = -ctrim if ctrim > 0 else None

    s1_trim = s1[ntrim:end1]
    s2_trim = s2[ntrim:end2]

    if len(s1_trim) != len(s2_trim):
        return unequal_length_distance

    return sum(a != b for a, b in zip(s1_trim, s2_trim))
    
    
def run_tcrdist_pipeline(
    df_clonotypes_all: pd.DataFrame,
    df_original: pd.DataFrame,
    metrics_a,
    metrics_b,
    weights_a,
    weights_b,
    kargs_a,
    kargs_b,
    db_file: str = "alphabeta_gammadelta_db.tsv",
    olga_beta_folder: str = "mouse_T_beta",
    olga_alpha_folder: str = "mouse_T_alpha",
    cutoff_beta: int = 40,
    cutoff_alpha: int = 40,
    cutoff_ab: int = 120,
    cpus: int = 12,
    calc_pgen = True,
) -> list:
    """
    Run the full TCRdist‐based pipeline, returning a list of DataFrames instead of writing to disk.

    Parameters
    ----------
    df_clonotypes_all : pd.DataFrame
        Pre-formatted clonotype DataFrame (grouped by tcr_id, with CDR3 sequences, V/J calls, etc.).
    df_original : pd.DataFrame
        The original “per-cell” DataFrame containing TCR info. Used only at the final merge step.
    metrics_a, metrics_b, weights_a, weights_b, kargs_a, kargs_b
        Pre-computed metric/weight/kwargs objects for alpha and beta chains, exactly as used to configure TCRrep.
    db_file : str (default="alphabeta_gammadelta_db.tsv")
        Path to the TCRdist database file.
    olga_beta_folder, olga_alpha_folder : str
        Folders containing the OLGA models for β and α chains, respectively.
    cutoff_beta, cutoff_alpha, cutoff_ab : int
        Distance cutoffs for clustering on the β chain, α chain, and full αβ chain.
    cpus : int (default=12)
        Number of CPU cores to pass to TCRrep.compute_distances() and to parmap for Pgen calculation.

    Returns
    -------
    list of pd.DataFrame
        [ 
          df_tcrdist_alpha,       # pairwise‐distance matrix for alpha chain
          df_tcrdist_beta,        # pairwise‐distance matrix for beta chain
          df_dist_alpha_beta,     # “full‐chain” (α+β) distance matrix
          clone_df,               # clonotype DataFrame after TCRdist clustering and Pgen annotation
          merged_df               # original df_original merged with Pgen & cluster columns
        ]
    """
    print("Generate new TR file")
    # 1) Instantiate TCRrep and configure it
    tr = TCRrep(
        cell_df = df_clonotypes_all,
        organism = "mouse",
        chains = ["alpha", "beta"],
        db_file = db_file,
        deduplicate = True,
        cpus = cpus,
        compute_distances = False
    )

    # Assign the pre-computed metrics, weights, and kwargs
    tr.metrics_a = metrics_a
    tr.metrics_b = metrics_b
    tr.weights_a = weights_a
    tr.weights_b = weights_b
    tr.kargs_a   = kargs_a
    tr.kargs_b   = kargs_b
    tr.cpus      = cpus

    # 2) Compute pairwise distances (fills tr.pw_alpha and tr.pw_beta)
    print("calculating TCR distances")
    tr.compute_distances()

    # 3) Build DataFrames for α and β distance matrices
    df_tcrdist_alpha = pd.DataFrame(
        tr.pw_alpha,
        columns = tr.clone_df["cdr3_a_aa"],
        index   = tr.clone_df["cdr3_a_aa"]
    )
    df_tcrdist_beta = pd.DataFrame(
        tr.pw_beta,
        columns = tr.clone_df["cdr3_b_aa"],
        index   = tr.clone_df["cdr3_b_aa"]
    )

    # 4) Build the “full‐chain” (α+β) distance matrix
    dist_total = tr.pw_alpha + tr.pw_beta
    columns = tr.clone_df["clonotype_origin"]
    df_dist_alpha_beta = pd.DataFrame(
        dist_total,
        columns = columns,
        index   = columns
    )

    # 5) Initialize a container to hold all output DataFrames
    output_dfs = [
        df_tcrdist_alpha,
        df_tcrdist_beta,
        df_dist_alpha_beta
    ]
    print("calculating TCR clusters")
    # 6) Perform TCRdist clustering on β chain
    clone_df = define_tcrdist_clusters_any_v(
        clone_df  = tr.clone_df,
        df_dist   = df_tcrdist_beta,
        chain     = "b",
        seq_col   = "cdr3_b_aa",
        cutoff    = cutoff_beta,
        sort_by_size = True
    )

    # 7) Then cluster on α chain (updating the same clone_df)
    clone_df = define_tcrdist_clusters_any_v(
        clone_df  = clone_df,
        df_dist   = df_tcrdist_alpha,
        chain     = "a",
        seq_col   = "cdr3_a_aa",
        cutoff    = cutoff_alpha,
        sort_by_size = True
    )

    # 8) Finally, cluster on the combined αβ distances
    clone_df = define_tcrdist_clusters_any_v(
        clone_df  = clone_df,
        df_dist   = df_dist_alpha_beta,
        chain     = "_ab",
        seq_col   = "clonotype_origin",
        cutoff    = cutoff_ab,
        sort_by_size = True
    )
    if calc_pgen == True:
        print("calculating TCR pgens")
        # 9) Load OLGA models for Pgen computation
        olga_beta_model  = OlgaModel(chain_folder = olga_beta_folder,  recomb_type = "VDJ")
        olga_alpha_model = OlgaModel(chain_folder = olga_alpha_folder, recomb_type = "VJ")

        # 10) Compute Pgen for each CDR3 sequence (using parmap for parallelism)
        clone_df["pgen_cdr3_b_aa"] = parmap.map(
            olga_beta_model.compute_aa_cdr3_pgen,
            tr.clone_df["cdr3_b_aa"],
            pm_pbar = True,
            pm_processes = cpus // 2  # for example, half of available cores
        )
        clone_df["pgen_cdr3_a_aa"] = parmap.map(
            olga_alpha_model.compute_aa_cdr3_pgen,
            tr.clone_df["cdr3_a_aa"],
            pm_pbar = True,
            pm_processes = cpus // 2
        )
        # 11) Append the clustered + Pgen‐augmented clonotype DataFrame
        output_dfs.append(clone_df)

        # 12) Merge the Pgen & cluster assignments back into the original “per‐cell” DataFrame
        to_merge = clone_df[[
            "clonotype_origin",
            "pgen_cdr3_b_aa",
            "pgen_cdr3_a_aa",
            "cc_aa_cdr3b",
            "cc_aa_cdr3a",
            "cc_aa_cdr3_ab"
        ]]
    else:
        print("skip pgen")
        # 11) Append the clustered + Pgen‐augmented clonotype DataFrame
        output_dfs.append(clone_df)

        # 12) Merge the Pgen & cluster assignments back into the original “per‐cell” DataFrame
        to_merge = clone_df[[
            "clonotype_origin",
            "cc_aa_cdr3b",
            "cc_aa_cdr3a",
            "cc_aa_cdr3_ab"
        ]]
    print("Merging data")
    

    merged_df = df_original.merge(
        to_merge,
        left_on  = "tcr_id",
        right_on = "clonotype_origin",
        how      = "left"
    )
    # If you don’t want to keep the extra 'clonotype_origin' column:
    #merged_df.drop(columns="clonotype_origin", inplace=True)

    # 13) Append the final merged DataFrame
    output_dfs.append(merged_df)

    return output_dfs
    print("Repertoire processed")
    
    
def run_tcrdist_pgen_only(
    df_clonotypes_all: pd.DataFrame,
    df_original: pd.DataFrame,
    db_file: str = "alphabeta_gammadelta_db.tsv",
    olga_beta_folder: str = "mouse_T_beta",
    olga_alpha_folder: str = "mouse_T_alpha",
    cpus: int = 12,
) -> list:
    """
    Run the full TCRdist‐based pipeline, returning a list of DataFrames instead of writing to disk.

    Parameters
    ----------
    df_clonotypes_all : pd.DataFrame
        Pre-formatted clonotype DataFrame (grouped by tcr_id, with CDR3 sequences, V/J calls, etc.).
    df_original : pd.DataFrame
        The original “per-cell” DataFrame containing TCR info. Used only at the final merge step.
    metrics_a, metrics_b, weights_a, weights_b, kargs_a, kargs_b
        Pre-computed metric/weight/kwargs objects for alpha and beta chains, exactly as used to configure TCRrep.
    db_file : str (default="alphabeta_gammadelta_db.tsv")
        Path to the TCRdist database file.
    olga_beta_folder, olga_alpha_folder : str
        Folders containing the OLGA models for β and α chains, respectively.
    cutoff_beta, cutoff_alpha, cutoff_ab : int
        Distance cutoffs for clustering on the β chain, α chain, and full αβ chain.
    cpus : int (default=12)
        Number of CPU cores to pass to TCRrep.compute_distances() and to parmap for Pgen calculation.

    Returns
    -------
    list of pd.DataFrame
        [ 
          df_tcrdist_alpha,       # pairwise‐distance matrix for alpha chain
          df_tcrdist_beta,        # pairwise‐distance matrix for beta chain
          df_dist_alpha_beta,     # “full‐chain” (α+β) distance matrix
          clone_df,               # clonotype DataFrame after TCRdist clustering and Pgen annotation
          merged_df               # original df_original merged with Pgen & cluster columns
        ]
    """
    print("Generate new TR file")
    # 1) Instantiate TCRrep and configure it
    tr = TCRrep(
        cell_df = df_clonotypes_all,
        organism = "mouse",
        chains = ["alpha", "beta"],
        db_file = db_file,
        deduplicate = True,
        cpus = cpus,
        compute_distances = False
    )
    #stored data
    output_dfs = []
    clone_df = tr.clone_df
    
    print("calculating TCR pgens")
    # 9) Load OLGA models for Pgen computation
    olga_beta_model  = OlgaModel(chain_folder = olga_beta_folder,  recomb_type = "VDJ")
    olga_alpha_model = OlgaModel(chain_folder = olga_alpha_folder, recomb_type = "VJ")

    # 10) Compute Pgen for each CDR3 sequence (using parmap for parallelism)
    clone_df["pgen_cdr3_b_aa"] = parmap.map(
        olga_beta_model.compute_aa_cdr3_pgen,
        tr.clone_df["cdr3_b_aa"],
        pm_pbar = True,
        pm_processes = cpus // 2  # for example, half of available cores
    )
    clone_df["pgen_cdr3_a_aa"] = parmap.map(
        olga_alpha_model.compute_aa_cdr3_pgen,
        tr.clone_df["cdr3_a_aa"],
        pm_pbar = True,
        pm_processes = cpus // 2
    )
    print("Merging data")
    # 11) Append the clustered + Pgen‐augmented clonotype DataFrame
    output_dfs.append(clone_df)

    # 12) Merge the Pgen & cluster assignments back into the original “per‐cell” DataFrame
    to_merge = clone_df[[
        "clonotype_origin",
        "pgen_cdr3_b_aa",
        "pgen_cdr3_a_aa",
    ]]

    merged_df = df_original.merge(
        to_merge,
        left_on  = "tcr_id",
        right_on = "clonotype_origin",
        how      = "left"
    )

    # 13) Append the final merged DataFrame
    output_dfs.append(merged_df)

    return output_dfs
    print("Pgen processed")
    
    
def export_tcrdist_graph_with_labels(
    df_dist:     pd.DataFrame,
    clone_df:    pd.DataFrame,
    seq_col:     str,
    v_a_col:     str,
    v_b_col:     str,
    cluster_col: str,
    epitope_col: str,
    cutoff:      float,
    min_deg:     int     = None,
    min_comp:    int     = None,
    out_file:    str     = "tcrdist_graph_labeled.gexf"
):
    """
    Build & export a TCRdist graph with node attributes:
      - cluster ID            (from clone_df[cluster_col])
      - v_a_gene              (from clone_df[v_a_col])
      - v_b_gene              (from clone_df[v_b_col])
      - epitope               (from clone_df[epitope_col])
    and set each node's displayed label = the cluster ID.

    Parameters
    ----------
    df_dist : pd.DataFrame
        square distance matrix indexed/columned by sequence strings
        (must match the values in clone_df[seq_col]).
    clone_df : pd.DataFrame
        must contain columns [seq_col, v_a_col, v_b_col, cluster_col, epitope_col]
    seq_col : str
        name of the CDR3 column in clone_df (and index of df_dist).
    v_a_col : str
        name of the α‐chain V‐gene column in clone_df.
    v_b_col : str
        name of the β‐chain V‐gene column in clone_df.
    cluster_col : str
        column in clone_df holding the cluster label (will become int→str).
    epitope_col : str
        column in clone_df holding the epitope annotation.
    cutoff : float
        distance threshold to draw edges.
    min_deg : int, optional
        if set, drop nodes with degree < min_deg.
    min_comp : int, optional
        if set, drop connected components smaller than min_comp.
    out_file : str
        file path to write the resulting GEXF.

    Returns
    -------
    G : networkx.Graph
        the filtered, annotated graph.
    """
    # 1) Build graph
    G = nx.Graph()
    nodes = df_dist.index.tolist()
    G.add_nodes_from(nodes)
    mat = df_dist.values
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            if mat[i, j] <= cutoff:
                G.add_edge(nodes[i], nodes[j])

    # 2) Filter by node degree
    if min_deg is not None:
        keep = [n for n, d in G.degree() if d >= min_deg]
        G = G.subgraph(keep).copy()

    # 3) Filter by component size
    if min_comp is not None:
        large_comps = [
            comp for comp in nx.connected_components(G)
            if len(comp) >= min_comp
        ]
        keep = set().union(*large_comps)
        G = G.subgraph(keep).copy()

    # 4) Attach metadata
    lookup = clone_df.set_index(seq_col)
    # cluster as string (so it appears under Partition in Gephi)
    cluster_map = lookup[cluster_col].astype(str).to_dict()
    nx.set_node_attributes(G, cluster_map, 'cluster')
    # also set as the node's visible label in Gephi
    nx.set_node_attributes(G, cluster_map, 'label')

    # V‐gene α & β
    nx.set_node_attributes(G, lookup[v_a_col].to_dict(), 'v_a_gene')
    nx.set_node_attributes(G, lookup[v_b_col].to_dict(), 'v_b_gene')
    # epitope
    nx.set_node_attributes(G, lookup[epitope_col].to_dict(), 'epitope')

    # 5) Export
    nx.write_gexf(G, out_file)
    print(f"Exported graph with {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges → {out_file}")
    return G
    
    
# Simple codon table (most common codons for each amino acid)
codon_table = {
    'A': 'GCT', 'R': 'CGT', 'N': 'AAT', 'D': 'GAT', 'C': 'TGT',
    'Q': 'CAA', 'E': 'GAA', 'G': 'GGT', 'H': 'CAT', 'I': 'ATT',
    'L': 'CTG', 'K': 'AAA', 'M': 'ATG', 'F': 'TTT', 'P': 'CCT',
    'S': 'AGC', 'T': 'ACC', 'W': 'TGG', 'Y': 'TAT', 'V': 'GTG',
    '*': 'TAA'  # stop codon
}


### generate artificial nt cdr3s from amino acid sequence
def reverse_translate_aa_to_nt(aa_seq: str) -> str:
    """
    Reverse-translate an amino acid sequence into a plausible nucleotide sequence.
    Uses a representative codon for each amino acid.
    Non-standard amino acids or missing residues are ignored.
    """
    if not isinstance(aa_seq, str):
        return ""
    aa_seq = aa_seq.strip().upper()
    nt_seq = ""
    for aa in aa_seq:
        nt_seq += codon_table.get(aa, "NNN")  # "NNN" for unknown amino acids
    return nt_seq
    
    
from typing import Literal

## stitch together amino acid TCRs from cellranger
def stitch_aa_from_cellranger(
    df: pd.DataFrame,
    v_info: pd.DataFrame,
    j_info: pd.DataFrame,
    v_gene_col: str,
    j_gene_col: str,
    cdr3_col: str,
    c_col: str,
    chain: Literal["alpha", "beta"],
) -> pd.DataFrame:
    """
    Stitch TCR sequences using amino acid sequences only.
    Leader and constant regions are provided as nt sequences and will be translated.
    """

    if chain not in ("alpha", "beta"):
        raise ValueError(f"Invalid chain {chain!r}; must be 'alpha' or 'beta'")

    df_out = df.copy()

    # Drop overlapping columns before merging v_info
    v_overlap = [
        "v_gene", "leader", "fwr1", "cdr1", "fwr2", "cdr2", "fwr3"
    ]
    df_out.drop(columns=[col for col in v_overlap if col in df_out.columns], inplace=True)

    # Merge v_info and rename AA columns
    df_out = df_out.merge(v_info, left_on=v_gene_col, right_on="v_gene", how="left")
    aa_cols = ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3"]
    df_out.rename(columns={col: f"{col}_{chain}" for col in aa_cols}, inplace=True)

    # Leader: translate nucleotide to AA
    df_out["leader_aa"] = df_out["leader"].fillna("").apply(translate_nt_to_aa)

    # Drop overlapping columns before merging j_info
    j_overlap = ["j_gene", "fwr4"]
    df_out.drop(columns=[col for col in j_overlap if col in df_out.columns], inplace=True)

    # Merge j_info and rename
    df_out = df_out.merge(j_info, left_on=j_gene_col, right_on="j_gene", how="left")
    df_out.rename(columns={"fwr4": f"fwr4_{chain}"}, inplace=True)

    # Fill missing AA columns
    needed_cols = ["leader_aa"] + [f"{col}_{chain}" for col in aa_cols] + [cdr3_col, f"fwr4_{chain}"]
    for col in needed_cols:
        if col not in df_out.columns:
            df_out[col] = ""
        df_out[col] = df_out[col].fillna("")

    # Stitch full TCR AA sequence
    concat_cols = ["leader_aa"] + [f"{col}_{chain}" for col in aa_cols] + [cdr3_col, f"fwr4_{chain}"]
    df_out[f"cellranger_{chain}_aa"] = df_out[concat_cols].agg("".join, axis=1)

    # Add constant region: translate nt to AA
    mTRAC = 'ACATCCAGAACCCAGAACCTGCTGTGTACCAATTGAAAGATCCTCGGTCTCAGGACAGCACCCTCTGCCTGTTCACCGACTTTGACTCCCAAATCAATGTGCCGAAAACAATGGAATCTGGAACGTTCATCACTGACAAAACTGTGCTGGACATGAAAGCTATGGATTCCAAGAGCAATGGAGCTATTGCCTGGAGCAACCAGACAAGCTTCACCTGCCAAGATATCTTCAAAGAGACCAACGCCACCTACCCCAGTTCAGACGTTCCCTGTGATGCCACGTTGACTGAGAAAAGCTTTGAAACAGATATGAACCTAAACTTTCAAAACCTGTCAGTTATGGGACTCCGAATCCTCCTGCTGAAAGTAGCCGGATTTAACCTGCTCATGACGCTGAGGCTGTGGTCCAGT'
    mTRBC1 = 'AGGATCTGAGAAATGTGACTCCACCCAAGGTCTCCTTGTTTGAGCCATCAAAAGCAGAGATTGCAAACAAACAAAAGGCTACCCTCGTGTGCTTGGCCAGGGGCTTCTTCCCTGACCACGTGGAGCTGAGCTGGTGGGTGAATGGCAAGGAAGTCCACAGTGGGGTCAGCACGGACCCTCAGGCCTACAAGGAGAGCAATTATAGCTACTGCCTGAGCAGCCGCCTGAGGGTCTCTGCTACCTTCTGGCACAATCCTCGCAACCACTTCCGCTGCCAAGTGCAGTTCCACGGGCTTTCAGAGGAGGACAAGTGGCCAGAGGGCTCACCCAAACCTGTCACACAGAACATCAGTGCAGAGGCCTGGGGCCGAGCAGACTGTGGGATTACCTCAGCATCCTATCAACAAGGGGTCTTGTCTGCCACCATCCTCTATGAGATCCTGCTAGGGAAAGCCACCCTGTATGCTGTGCTTGTCAGTACACTGGTGGTGATGGCTATGGTCAAAAGAAAGAACTCA'
    mTRBC2 = 'AGGATCTGAGAAATGTGACTCCACCCAAGGTCTCCTTGTTTGAGCCATCAAAAGCAGAGATTGCAAACAAACAAAAGGCTACCCTCGTGTGCTTGGCCAGGGGCTTCTTCCCTGACCACGTGGAGCTGAGCTGGTGGGTGAATGGCAAGGAAGTCCACAGTGGGGTCAGCACGGACCCTCAGGCCTACAAGGAGAGCAATTATAGCTACTGCCTGAGCAGCCGCCTGAGGGTCTCTGCTACCTTCTGGCACAATCCTCGAAACCACTTCCGCTGCCAAGTGCAGTTCCACGGGCTTTCAGAGGAGGACAAGTGGCCAGAGGGCTCACCCAAACCTGTCACACAGAACATCAGTGCAGAGGCCTGGGGCCGAGCAGACTGTGGAATCACTTCAGCATCCTATCATCAGGGGGTTCTGTCTGCAACCATCCTCTATGAGATCCTACTGGGGAAGGCCACCCTATATGCTGTGCTGGTCAGTACCCTGGTGCTGATGGCTATGGTCAAGAAAAAAAATTCC'

    if chain == 'alpha':
        df_out[f"cellranger_{chain}_const_aa"] = df_out[f"cellranger_{chain}_aa"].apply(lambda x: x + translate_nt_to_aa(mTRAC))
    else:  # beta
        df_out[f"cellranger_{chain}_const_aa"] = ""
        mask1 = df[c_col] == "TRBC1"
        df_out.loc[mask1, f"cellranger_{chain}_const_aa"] = (
            df_out.loc[mask1, f"cellranger_{chain}_aa"] + translate_nt_to_aa(mTRBC1)
        )
        mask2 = df[c_col] == "TRBC2"
        df_out.loc[mask2, f"cellranger_{chain}_const_aa"] = (
            df_out.loc[mask2, f"cellranger_{chain}_aa"] + translate_nt_to_aa(mTRBC2)
        )

    return df_out