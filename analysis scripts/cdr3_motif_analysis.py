from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


AA20 = set("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class CDR3FlexConfig:
    # Column names
    epi_cdr3_col: str
    bg_cdr3_col: str

    epi_vj_col: str
    bg_vj_col: str

    ann_vj_col: str
    ann_status_col: str

    # Optional contextual variables, for example epitope.
    # Corresponding entries must represent the same variable.
    epi_context_cols: tuple[str, ...] = ()
    ann_context_cols: tuple[str, ...] = ()

    # Remove conserved CDR3 anchor residues.
    # For sequences such as CASS...EQYF, trim_left=1 and trim_right=1
    # remove the initial C and terminal F.
    trim_left: int = 1
    trim_right: int = 1

    # Minimum number of unique sequences in each VJ × length stratum
    min_epi_n: int = 8
    min_bg_n: int = 20

    # Entropy estimator
    entropy_method: Literal["mle", "miller_madow"] = "miller_madow"

    # Core motif definition
    core_mode: Literal["top_k", "threshold"] = "top_k"
    core_top_k: int = 3

    # A core position must show at least this entropy reduction:
    # H_background - H_epitope
    core_min_constraint: float = 0.05

    # Dominant residue frequency required for a core position
    core_min_dominant_freq: float = 0.35

    # Threshold used to define the breadth of CDR3 restriction
    breadth_constraint_threshold: float = 0.10

    # Optional equal-size subsampling within each VJ × length stratum.
    # Both epitope and background sequences are sampled to the same size.
    equal_size_subsampling: bool = False
    subsample_n: int | None = None
    n_subsamples: int = 200

    random_state: int = 1
	
# ---------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------
	
def _clean_sequence_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def _valid_aa_sequence(sequence: object) -> bool:
    return (
        isinstance(sequence, str)
        and len(sequence) > 0
        and set(sequence).issubset(AA20)
    )


def _trim_sequence(sequence: str, left: int, right: int) -> str:
    end = len(sequence) - right if right > 0 else len(sequence)
    return sequence[left:end]


def prepare_inputs(
    df_epi: pd.DataFrame,
    df_bg: pd.DataFrame,
    df_ann: pd.DataFrame,
    cfg: CDR3FlexConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize the three input dataframes.

    Returns
    -------
    epi:
        Epitope-specific sequences with VJ annotations.
    bg:
        Background sequences.
    """

    if len(cfg.epi_context_cols) != len(cfg.ann_context_cols):
        raise ValueError(
            "epi_context_cols and ann_context_cols must have equal length."
        )

    required_epi = {
        cfg.epi_cdr3_col,
        cfg.epi_vj_col,
        *cfg.epi_context_cols,
    }

    required_bg = {
        cfg.bg_cdr3_col,
        cfg.bg_vj_col,
    }

    required_ann = {
        cfg.ann_vj_col,
        cfg.ann_status_col,
        *cfg.ann_context_cols,
    }

    missing_epi = required_epi.difference(df_epi.columns)
    missing_bg = required_bg.difference(df_bg.columns)
    missing_ann = required_ann.difference(df_ann.columns)

    if missing_epi:
        raise KeyError(
            f"Missing epitope-specific columns: {sorted(missing_epi)}"
        )

    if missing_bg:
        raise KeyError(
            f"Missing background columns: {sorted(missing_bg)}"
        )

    if missing_ann:
        raise KeyError(
            f"Missing annotation columns: {sorted(missing_ann)}"
        )

    epi = pd.DataFrame(
        {
            "_cdr3_full": _clean_sequence_series(
                df_epi[cfg.epi_cdr3_col]
            ),
            "_vj": df_epi[cfg.epi_vj_col].astype("string"),
        }
    )

    for column in cfg.epi_context_cols:
        epi[column] = df_epi[column].values

    bg = pd.DataFrame(
        {
            "_cdr3_full": _clean_sequence_series(
                df_bg[cfg.bg_cdr3_col]
            ),
            "_vj": df_bg[cfg.bg_vj_col].astype("string"),
        }
    )

    ann = pd.DataFrame(
        {
            "_vj": df_ann[cfg.ann_vj_col].astype("string"),
            "_status": df_ann[cfg.ann_status_col].values,
        }
    )

    for epi_column, ann_column in zip(
        cfg.epi_context_cols,
        cfg.ann_context_cols,
    ):
        ann[epi_column] = df_ann[ann_column].values

    annotation_keys = [*cfg.epi_context_cols, "_vj"]

    # Detect contradictory enriched/non-enriched annotations.
    status_counts = (
        ann.groupby(annotation_keys, dropna=False)["_status"]
        .nunique(dropna=False)
    )

    if (status_counts > 1).any():
        raise ValueError(
            "At least one VJ annotation key maps to multiple statuses."
        )

    ann = ann.drop_duplicates(annotation_keys)

    epi = epi.merge(
        ann,
        on=annotation_keys,
        how="left",
        validate="many_to_one",
    )

    n_unannotated = int(epi["_status"].isna().sum())

    if n_unannotated > 0:
        raise ValueError(
            f"{n_unannotated} epitope-specific rows lack a VJ annotation."
        )

    # Keep only standard amino-acid sequences.
    epi = epi[
        epi["_cdr3_full"].map(_valid_aa_sequence)
    ].copy()

    bg = bg[
        bg["_cdr3_full"].map(_valid_aa_sequence)
    ].copy()

    minimum_length = cfg.trim_left + cfg.trim_right + 1

    epi = epi[
        epi["_cdr3_full"].str.len() >= minimum_length
    ].copy()

    bg = bg[
        bg["_cdr3_full"].str.len() >= minimum_length
    ].copy()

    # Remove conserved terminal residues.
    epi["_cdr3"] = epi["_cdr3_full"].map(
        lambda sequence: _trim_sequence(
            sequence,
            cfg.trim_left,
            cfg.trim_right,
        )
    )

    bg["_cdr3"] = bg["_cdr3_full"].map(
        lambda sequence: _trim_sequence(
            sequence,
            cfg.trim_left,
            cfg.trim_right,
        )
    )

    epi["_length"] = epi["_cdr3"].str.len().astype(int)
    bg["_length"] = bg["_cdr3"].str.len().astype(int)

    # Each full CDR3 sequence contributes once.
    epi = epi.drop_duplicates(
        [
            *cfg.epi_context_cols,
            "_vj",
            "_cdr3_full",
        ]
    )

    bg = bg.drop_duplicates(
        [
            "_vj",
            "_cdr3_full",
        ]
    )

    return (
        epi.reset_index(drop=True),
        bg.reset_index(drop=True),
    )
	
# ---------------------------------------------------------------------
# Position-specific entropy and sequence variability
# ---------------------------------------------------------------------

def positional_profile(
    sequences: Sequence[str],
    entropy_method: Literal[
        "mle",
        "miller_madow",
    ] = "miller_madow",
) -> pd.DataFrame:
    """
    Calculate position-specific CDR3β variability.

    All sequences must have identical length.

    entropy_norm:
        Shannon entropy divided by log(20).
        0 = one amino acid.
        1 = maximal 20-amino-acid entropy.

    effective_aa:
        Effective number of amino acids at that position.

    pairwise_disagreement:
        Exact mean pairwise Hamming disagreement at that position.
    """

    sequences = np.asarray(
        list(sequences),
        dtype=object,
    )

    n_sequences = len(sequences)

    if n_sequences == 0:
        raise ValueError("At least one sequence is required.")

    lengths = {len(sequence) for sequence in sequences}

    if len(lengths) != 1:
        raise ValueError(
            "All sequences in a positional profile must have equal length."
        )

    cdr3_length = lengths.pop()

    rows = []

    for position in range(cdr3_length):
        residues = pd.Series(
            [sequence[position] for sequence in sequences],
            dtype="string",
        )

        counts = residues.value_counts()
        probabilities = counts.to_numpy(dtype=float) / n_sequences

        entropy_mle = -np.sum(
            probabilities * np.log(probabilities)
        )

        entropy = entropy_mle

        if entropy_method == "miller_madow":
            observed_states = len(counts)

            entropy += (
                observed_states - 1
            ) / (2.0 * n_sequences)

        elif entropy_method != "mle":
            raise ValueError(
                "entropy_method must be 'mle' or 'miller_madow'."
            )

        entropy = min(
            float(entropy),
            float(np.log(20.0)),
        )

        normalized_entropy = entropy / np.log(20.0)
        effective_aa = float(np.exp(entropy))

        dominant_aa = str(counts.index[0])
        dominant_frequency = float(
            counts.iloc[0] / n_sequences
        )

        # Exact disagreement across all sequence pairs.
        if n_sequences > 1:
            count_values = counts.to_numpy(dtype=float)

            matching_pairs = np.sum(
                count_values * (count_values - 1.0)
            )

            total_ordered_pairs = (
                n_sequences * (n_sequences - 1.0)
            )

            pairwise_disagreement = (
                1.0
                - matching_pairs / total_ordered_pairs
            )
        else:
            pairwise_disagreement = np.nan

        rows.append(
            {
                "position": position + 1,
                "n": n_sequences,
                "entropy_raw": entropy,
                "entropy_norm": normalized_entropy,
                "effective_aa": effective_aa,
                "dominant_aa": dominant_aa,
                "dominant_freq": dominant_frequency,
                "pairwise_disagreement": pairwise_disagreement,
            }
        )

    return pd.DataFrame(rows)
	
# ---------------------------------------------------------------------
# Optional equal-size subsampling
# ---------------------------------------------------------------------

def _mean_resampled_profile(
    sequences: np.ndarray,
    n_target: int,
    n_subsamples: int,
    entropy_method: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Repeatedly subsample sequences without replacement and average the
    resulting positional profiles.
    """

    profiles = []

    for repeat in range(n_subsamples):
        sampled_sequences = rng.choice(
            sequences,
            size=n_target,
            replace=False,
        )

        profile = positional_profile(
            sampled_sequences,
            entropy_method=entropy_method,
        )

        profile["_repeat"] = repeat
        profiles.append(profile)

    combined = pd.concat(
        profiles,
        ignore_index=True,
    )

    numeric_columns = [
        "n",
        "entropy_raw",
        "entropy_norm",
        "effective_aa",
        "dominant_freq",
        "pairwise_disagreement",
    ]

    averaged = (
        combined.groupby(
            "position",
            as_index=False,
        )[numeric_columns]
        .mean()
    )

    # Preserve the dominant amino acid from the complete sequence set.
    complete_profile = positional_profile(
        sequences,
        entropy_method=entropy_method,
    )

    averaged["dominant_aa"] = (
        complete_profile["dominant_aa"].values
    )

    return averaged
	
# ---------------------------------------------------------------------
# Analysis of one VJ × CDR3-length stratum
# ---------------------------------------------------------------------

def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or denominator <= 0
    ):
        return np.nan

    return float(numerator / denominator)


def analyze_stratum(
    epi_sequences: Sequence[str],
    bg_sequences: Sequence[str],
    cfg: CDR3FlexConfig,
    rng: np.random.Generator,
) -> tuple[dict[str, object], pd.DataFrame]:
    """
    Compare epitope-specific sequences with background sequences for one
    exact VJ × CDR3-length stratum.
    """

    epi_sequences = np.asarray(
        list(epi_sequences),
        dtype=object,
    )

    bg_sequences = np.asarray(
        list(bg_sequences),
        dtype=object,
    )

    n_epi = len(epi_sequences)
    n_bg = len(bg_sequences)

    if cfg.equal_size_subsampling:
        n_target = (
            cfg.subsample_n
            if cfg.subsample_n is not None
            else min(n_epi, n_bg)
        )

        if n_target > min(n_epi, n_bg):
            raise ValueError(
                "subsample_n exceeds the smaller stratum size."
            )

        if n_target < 2:
            raise ValueError(
                "subsample_n must be at least 2."
            )

        epi_profile = _mean_resampled_profile(
            sequences=epi_sequences,
            n_target=n_target,
            n_subsamples=cfg.n_subsamples,
            entropy_method=cfg.entropy_method,
            rng=rng,
        )

        bg_profile = _mean_resampled_profile(
            sequences=bg_sequences,
            n_target=n_target,
            n_subsamples=cfg.n_subsamples,
            entropy_method=cfg.entropy_method,
            rng=rng,
        )

    else:
        n_target = None

        epi_profile = positional_profile(
            epi_sequences,
            entropy_method=cfg.entropy_method,
        )

        bg_profile = positional_profile(
            bg_sequences,
            entropy_method=cfg.entropy_method,
        )

    profile = epi_profile.merge(
        bg_profile,
        on="position",
        suffixes=("_epi", "_bg"),
        validate="one_to_one",
    )

    # Positive constraint means lower entropy in the epitope-specific
    # repertoire than in the matched background.
    profile["constraint"] = (
        profile["entropy_norm_bg"]
        - profile["entropy_norm_epi"]
    )

    profile["pairwise_constraint"] = (
        profile["pairwise_disagreement_bg"]
        - profile["pairwise_disagreement_epi"]
    )

    eligible_core = (
        profile["constraint"].ge(
            cfg.core_min_constraint
        )
        & profile["dominant_freq_epi"].ge(
            cfg.core_min_dominant_freq
        )
    )

    if cfg.core_mode == "top_k":
        core_indices = (
            profile.loc[eligible_core]
            .nlargest(
                cfg.core_top_k,
                "constraint",
            )
            .index
        )

    elif cfg.core_mode == "threshold":
        core_indices = profile.index[eligible_core]

    else:
        raise ValueError(
            "core_mode must be 'top_k' or 'threshold'."
        )

    profile["is_core"] = False
    profile.loc[core_indices, "is_core"] = True

    core = profile[profile["is_core"]]
    residual = profile[~profile["is_core"]]

    core_strength = (
        float(core["constraint"].mean())
        if not core.empty
        else np.nan
    )

    residual_constraint = (
        float(residual["constraint"].mean())
        if not residual.empty
        else np.nan
    )

    core_motif = "|".join(
        f"{int(row.position)}:{row.dominant_aa_epi}"
        for row in core.sort_values(
            "position"
        ).itertuples()
    )

    residual_entropy_epi = (
        float(residual["entropy_norm_epi"].mean())
        if not residual.empty
        else np.nan
    )

    residual_entropy_bg = (
        float(residual["entropy_norm_bg"].mean())
        if not residual.empty
        else np.nan
    )

    residual_hamming_epi = (
        float(
            residual[
                "pairwise_disagreement_epi"
            ].mean()
        )
        if not residual.empty
        else np.nan
    )

    residual_hamming_bg = (
        float(
            residual[
                "pairwise_disagreement_bg"
            ].mean()
        )
        if not residual.empty
        else np.nan
    )

    metrics = {
        "n_epi": n_epi,
        "n_bg": n_bg,
        "subsample_n": n_target,
        "cdr3_length": len(epi_sequences[0]),

        "n_core_positions": int(
            profile["is_core"].sum()
        ),

        "core_positions": ",".join(
            profile.loc[
                profile["is_core"],
                "position",
            ]
            .astype(int)
            .astype(str)
            .tolist()
        ),

        "core_motif": core_motif,
        "core_strength": core_strength,

        "constraint_breadth": float(
            profile["constraint"]
            .ge(
                cfg.breadth_constraint_threshold
            )
            .mean()
        ),

        "overall_entropy_epi": float(
            profile["entropy_norm_epi"].mean()
        ),

        "overall_entropy_bg": float(
            profile["entropy_norm_bg"].mean()
        ),

        "overall_flexibility_ratio": _safe_ratio(
            profile["entropy_norm_epi"].mean(),
            profile["entropy_norm_bg"].mean(),
        ),

        "residual_position_count": len(residual),

        "residual_entropy_epi": residual_entropy_epi,
        "residual_entropy_bg": residual_entropy_bg,

        "residual_flexibility_ratio": _safe_ratio(
            residual_entropy_epi,
            residual_entropy_bg,
        ),

        "residual_constraint": residual_constraint,

        "residual_pairwise_hamming_epi": (
            residual_hamming_epi
        ),

        "residual_pairwise_hamming_bg": (
            residual_hamming_bg
        ),

        "residual_hamming_ratio": _safe_ratio(
            residual_hamming_epi,
            residual_hamming_bg,
        ),

        # Larger values indicate that restriction is concentrated
        # in the core motif rather than distributed across the CDR3.
        "constraint_localization": (
            core_strength - residual_constraint
            if pd.notna(core_strength)
            and pd.notna(residual_constraint)
            else np.nan
        ),
    }

    return metrics, profile
	
# ---------------------------------------------------------------------
# Complete analysis pipeline
# ---------------------------------------------------------------------

def run_cdr3b_flexibility_analysis(
    df_epi: pd.DataFrame,
    df_bg: pd.DataFrame,
    df_ann: pd.DataFrame,
    cfg: CDR3FlexConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Analyze each:

        context × VJ × exact CDR3β length

    Returns
    -------
    metrics_df:
        One row per analyzable VJ × length stratum.

    profiles_df:
        One row per CDR3 position per stratum.

    skipped_df:
        Strata excluded because of insufficient sequence numbers.
    """

    epi, bg = prepare_inputs(
        df_epi=df_epi,
        df_bg=df_bg,
        df_ann=df_ann,
        cfg=cfg,
    )

    rng = np.random.default_rng(
        cfg.random_state
    )

    grouping_columns = [
        *cfg.epi_context_cols,
        "_vj",
        "_status",
        "_length",
    ]

    metric_rows = []
    profile_frames = []
    skipped_rows = []

    for key, epi_group in epi.groupby(
        grouping_columns,
        dropna=False,
        sort=False,
    ):
        if not isinstance(key, tuple):
            key = (key,)

        metadata = dict(
            zip(grouping_columns, key)
        )

        bg_group = bg[
            bg["_vj"].eq(metadata["_vj"])
            & bg["_length"].eq(
                metadata["_length"]
            )
        ]

        n_epi = len(epi_group)
        n_bg = len(bg_group)

        if (
            n_epi < cfg.min_epi_n
            or n_bg < cfg.min_bg_n
        ):
            skipped_rows.append(
                {
                    **metadata,
                    "n_epi": n_epi,
                    "n_bg": n_bg,
                    "reason": "insufficient_size",
                }
            )
            continue

        metrics, profile = analyze_stratum(
            epi_sequences=epi_group[
                "_cdr3"
            ].tolist(),
            bg_sequences=bg_group[
                "_cdr3"
            ].tolist(),
            cfg=cfg,
            rng=rng,
        )

        metrics.update(metadata)
        metric_rows.append(metrics)

        for column, value in metadata.items():
            profile[column] = value

        profile["n_epi_total"] = n_epi
        profile["n_bg_total"] = n_bg

        profile_frames.append(profile)

    metrics_df = pd.DataFrame(metric_rows)

    profiles_df = (
        pd.concat(
            profile_frames,
            ignore_index=True,
        )
        if profile_frames
        else pd.DataFrame()
    )

    skipped_df = pd.DataFrame(skipped_rows)

    return metrics_df, profiles_df, skipped_df

# -----------------------------------------------------	
# Aggregate exact-length results to one result per VJ
# -----------------------------------------------------	

DEFAULT_SUMMARY_METRICS = (
    "core_strength",
    "constraint_breadth",
    "overall_flexibility_ratio",
    "residual_flexibility_ratio",
    "residual_hamming_ratio",
    "constraint_localization",
)


def aggregate_metrics_to_vj(
    metrics_df: pd.DataFrame,
    context_cols: Sequence[str] = (),
    metrics: Sequence[str] = DEFAULT_SUMMARY_METRICS,
    weight_by_sequence_count: bool = False,
) -> pd.DataFrame:
    """
    Aggregate length-specific results to one row per VJ.

    By default, each analyzable CDR3 length receives equal weight.

    Set weight_by_sequence_count=True to weight each length by its number
    of epitope-specific sequences.
    """

    group_columns = [
        *context_cols,
        "_vj",
        "_status",
    ]

    rows = []

    for key, group in metrics_df.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        if not isinstance(key, tuple):
            key = (key,)

        row = dict(
            zip(group_columns, key)
        )

        weights = (
            group["n_epi"].to_numpy(
                dtype=float
            )
            if weight_by_sequence_count
            else np.ones(
                len(group),
                dtype=float,
            )
        )

        for metric in metrics:
            values = group[metric].to_numpy(
                dtype=float
            )

            valid = (
                np.isfinite(values)
                & np.isfinite(weights)
            )

            row[metric] = (
                float(
                    np.average(
                        values[valid],
                        weights=weights[valid],
                    )
                )
                if valid.any()
                else np.nan
            )

        row["n_epi"] = int(
            group["n_epi"].sum()
        )

        row["n_bg"] = int(
            group["n_bg"].sum()
        )

        row["n_lengths"] = int(
            group["cdr3_length"].nunique()
        )

        rows.append(row)

    return pd.DataFrame(rows)
	
# ------------------------------------
# Statistical comparison of enriched and non-enriched VJs
# ------------------------------------
	
def compare_status_groups(
    vj_summary: pd.DataFrame,
    enriched_label: object,
    non_enriched_label: object,
    context_cols: Sequence[str] = (),
    metrics: Sequence[str] = DEFAULT_SUMMARY_METRICS,
) -> pd.DataFrame:
    """
    Mann–Whitney comparison using one VJ combination as one unit.

    Positive rank_biserial values indicate larger values in enriched VJs.
    """

    if context_cols:
        grouped = vj_summary.groupby(
            list(context_cols),
            dropna=False,
            sort=False,
        )
    else:
        grouped = [((), vj_summary)]

    rows = []

    for context_key, group in grouped:
        if not isinstance(context_key, tuple):
            context_key = (context_key,)

        context = dict(
            zip(context_cols, context_key)
        )

        for metric in metrics:
            enriched = (
                group.loc[
                    group["_status"].eq(
                        enriched_label
                    ),
                    metric,
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            non_enriched = (
                group.loc[
                    group["_status"].eq(
                        non_enriched_label
                    ),
                    metric,
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            if (
                len(enriched) == 0
                or len(non_enriched) == 0
            ):
                continue

            test = mannwhitneyu(
                enriched,
                non_enriched,
                alternative="two-sided",
                method="auto",
            )

            rank_biserial = (
                2.0 * test.statistic
                / (
                    len(enriched)
                    * len(non_enriched)
                )
                - 1.0
            )

            rows.append(
                {
                    **context,
                    "metric": metric,
                    "n_enriched_vj": len(enriched),
                    "n_non_enriched_vj": len(
                        non_enriched
                    ),
                    "median_enriched": float(
                        np.median(enriched)
                    ),
                    "median_non_enriched": float(
                        np.median(non_enriched)
                    ),
                    "median_difference": float(
                        np.median(enriched)
                        - np.median(non_enriched)
                    ),
                    "mannwhitney_u": float(
                        test.statistic
                    ),
                    "p_value": float(
                        test.pvalue
                    ),
                    "rank_biserial": float(
                        rank_biserial
                    ),
                }
            )

    return pd.DataFrame(rows)
	
# ------------------------------------
# Visualization functions
# ------------------------------------

def _filter_context(
    dataframe: pd.DataFrame,
    context_filter: dict[str, object] | None,
) -> pd.DataFrame:
    result = dataframe.copy()

    if context_filter:
        for column, value in context_filter.items():
            result = result[
                result[column].eq(value)
            ]

    return result


def plot_metric_by_status(
    dataframe: pd.DataFrame,
    metric: str,
    status_order: Sequence[object] | None = None,
    context_filter: dict[str, object] | None = None,
    reference_line: float | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    random_state: int = 1,
):
    """
    Use vj_summary to show one point per VJ.
    Use metrics_df to show one point per VJ × length stratum.
    """

    data = _filter_context(
        dataframe,
        context_filter,
    )

    data = data.dropna(
        subset=["_status", metric]
    )

    if status_order is None:
        status_order = list(
            pd.unique(data["_status"])
        )

    values = [
        data.loc[
            data["_status"].eq(status),
            metric,
        ].to_numpy(dtype=float)
        for status in status_order
    ]

    fig, ax = plt.subplots(
        figsize=(
            2.2 * len(status_order) + 1.5,
            4,
        )
    )

    ax.boxplot(
        values,
        labels=[
            str(status)
            for status in status_order
        ],
        showfliers=False,
    )

    rng = np.random.default_rng(
        random_state
    )

    for position, group_values in enumerate(
        values,
        start=1,
    ):
        jitter = rng.normal(
            loc=position,
            scale=0.045,
            size=len(group_values),
        )

        ax.scatter(
            jitter,
            group_values,
            s=22,
            alpha=0.75,
        )

    if reference_line is not None:
        ax.axhline(
            reference_line,
            linestyle="--",
            linewidth=1,
        )

    ax.set_xlabel("VJ annotation")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or metric)

    fig.tight_layout()

    return fig, ax
	
# ------------------------------------
# Position-specific conservation for one CDR3β length
# ------------------------------------

def plot_constraint_profile(
    profiles_df: pd.DataFrame,
    cdr3_length: int,
    status_order: Sequence[object] | None = None,
    context_filter: dict[str, object] | None = None,
    weight_by_sequence_count: bool = False,
):
    """
    Plot H_background - H_epitope across CDR3 positions.

    Positive values indicate epitope-associated sequence restriction.
    """

    data = _filter_context(
        profiles_df,
        context_filter,
    )

    data = data[
        data["_length"].eq(cdr3_length)
    ].copy()

    if data.empty:
        raise ValueError(
            f"No data for CDR3β length {cdr3_length}."
        )

    if status_order is None:
        status_order = list(
            pd.unique(data["_status"])
        )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    for status in status_order:
        subset = data[
            data["_status"].eq(status)
        ]

        rows = []

        for position, group in subset.groupby(
            "position",
            sort=True,
        ):
            values = group[
                "constraint"
            ].to_numpy(dtype=float)

            weights = (
                group["n_epi_total"].to_numpy(
                    dtype=float
                )
                if weight_by_sequence_count
                else np.ones(
                    len(group),
                    dtype=float,
                )
            )

            valid = (
                np.isfinite(values)
                & np.isfinite(weights)
            )

            if valid.any():
                rows.append(
                    (
                        position,
                        np.average(
                            values[valid],
                            weights=weights[valid],
                        ),
                    )
                )

        if rows:
            positions, constraints = zip(
                *rows
            )

            ax.plot(
                positions,
                constraints,
                marker="o",
                label=str(status),
            )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Position in trimmed CDR3β"
    )

    ax.set_ylabel(
        "Constraint: H(background) − H(epitope)"
    )

    ax.set_title(
        f"CDR3β length {cdr3_length}"
    )

    ax.legend(frameon=False)
    fig.tight_layout()

    return fig, ax

# --------------------------------------	
# Constraint heatmap across CDR3β lengths
# --------------------------------------	

def plot_constraint_heatmap(
    profiles_df: pd.DataFrame,
    status: object,
    context_filter: dict[str, object] | None = None,
    weight_by_sequence_count: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
):
    data = _filter_context(
        profiles_df,
        context_filter,
    )

    data = data[
        data["_status"].eq(status)
    ].copy()

    if data.empty:
        raise ValueError(
            f"No profile data for status {status!r}."
        )

    rows = []

    for (
        cdr3_length,
        position,
    ), group in data.groupby(
        ["_length", "position"],
        sort=True,
    ):
        values = group[
            "constraint"
        ].to_numpy(dtype=float)

        weights = (
            group["n_epi_total"].to_numpy(
                dtype=float
            )
            if weight_by_sequence_count
            else np.ones(
                len(group),
                dtype=float,
            )
        )

        valid = (
            np.isfinite(values)
            & np.isfinite(weights)
        )

        if valid.any():
            rows.append(
                {
                    "_length": int(
                        cdr3_length
                    ),
                    "position": int(
                        position
                    ),
                    "constraint": float(
                        np.average(
                            values[valid],
                            weights=weights[valid],
                        )
                    ),
                }
            )

    aggregated = pd.DataFrame(rows)

    matrix = (
        aggregated.pivot(
            index="_length",
            columns="position",
            values="constraint",
        )
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(
            max(
                6,
                matrix.shape[1] * 0.45,
            ),
            max(
                3,
                matrix.shape[0] * 0.45,
            ),
        )
    )

    image = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(
        np.arange(matrix.shape[1])
    )

    ax.set_xticklabels(
        matrix.columns.astype(int)
    )

    ax.set_yticks(
        np.arange(matrix.shape[0])
    )

    ax.set_yticklabels(
        matrix.index.astype(int)
    )

    ax.set_xlabel(
        "Position in trimmed CDR3β"
    )

    ax.set_ylabel(
        "Trimmed CDR3β length"
    )

    ax.set_title(
        f"Positional constraint: {status}"
    )

    fig.colorbar(
        image,
        ax=ax,
        label=(
            "H(background) − H(epitope)"
        ),
    )

    fig.tight_layout()

    return fig, ax

# --------------------------------------	
# Core conservation versus residual flexibility
# --------------------------------------	
def plot_core_vs_residual(
    vj_summary: pd.DataFrame,
    status_order: Sequence[object] | None = None,
    context_filter: dict[str, object] | None = None,
):
    data = _filter_context(
        vj_summary,
        context_filter,
    )

    data = data.dropna(
        subset=[
            "_status",
            "core_strength",
            "residual_flexibility_ratio",
        ]
    )

    if status_order is None:
        status_order = list(
            pd.unique(data["_status"])
        )

    fig, ax = plt.subplots(
        figsize=(5, 4.5)
    )

    for status in status_order:
        subset = data[
            data["_status"].eq(status)
        ]

        ax.scatter(
            subset["core_strength"],
            subset[
                "residual_flexibility_ratio"
            ],
            label=str(status),
            s=35,
            alpha=0.8,
        )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Core motif strength"
    )

    ax.set_ylabel(
        "Residual flexibility ratio"
    )

    ax.set_title(
        "Core conservation versus non-core flexibility"
    )

    ax.legend(frameon=False)
    fig.tight_layout()

    return fig, ax
    
    
# --------
# position wise entropy plot
# --------

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def _p_to_label(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"

def plot_metric_by_epitope_with_stats(
    dataframe: pd.DataFrame,
    metric: str,
    epitope_col: str = "annotated_specificity",
    epitope_order: Sequence[object] | None = None,
    status_order: Sequence[object] = (
        "Non-enriched",
        "Enriched",
    ),
    ncols: int = 3,
    subplot_size: tuple[float, float] = (2.6, 3.8),
    ylim: tuple[float, float] | None = None,
    whis: tuple[float, float] = (5, 95),
    reference_line: float | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    sharey: bool = True,
    random_state: int = 1,
    p_adjust: str | None = "holm",
    stat_label: str = "stars",
    hide_top_right_spines: bool = True,
):
    """
    Plot one boxplot per epitope and compare the two status groups using
    a two-sided Mann–Whitney U test.

    Each row of `dataframe` is treated as one independent observation.
    When `dataframe` is `vj_summary`, each point represents one VJ
    combination.

    Parameters
    ----------
    p_adjust:
        Multiple-testing correction passed to statsmodels.multipletests.
        Common options are "holm", "fdr_bh", or None.

    stat_label:
        "stars" for significance symbols,
        "p" for adjusted p-values,
        "both" for symbols and adjusted p-values.
    """

    if len(status_order) != 2:
        raise ValueError(
            "Exactly two status groups are required for the statistical test."
        )

    required_columns = {
        epitope_col,
        "_status",
        metric,
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    data = dataframe.dropna(
        subset=[epitope_col, "_status", metric]
    ).copy()

    data = data[
        data["_status"].isin(status_order)
    ].copy()

    if epitope_order is None:
        epitope_order = list(pd.unique(data[epitope_col]))
    else:
        observed_epitopes = set(data[epitope_col])

        epitope_order = [
            epitope
            for epitope in epitope_order
            if epitope in observed_epitopes
        ]

    if len(epitope_order) == 0:
        raise ValueError("No epitopes remain after filtering.")

    # ---------------------------------------------------------------
    # Statistical tests
    # ---------------------------------------------------------------

    stats_rows = []

    status_1, status_2 = status_order

    for epitope in epitope_order:
        epitope_data = data[
            data[epitope_col].eq(epitope)
        ]

        values_1 = (
            epitope_data.loc[
                epitope_data["_status"].eq(status_1),
                metric,
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        values_2 = (
            epitope_data.loc[
                epitope_data["_status"].eq(status_2),
                metric,
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(values_1) > 0 and len(values_2) > 0:
            test = mannwhitneyu(
                values_1,
                values_2,
                alternative="two-sided",
                method="auto",
            )

            p_value = float(test.pvalue)
            u_statistic = float(test.statistic)

            # Positive values indicate larger values in status_1.
            rank_biserial = (
                2.0 * u_statistic
                / (len(values_1) * len(values_2))
                - 1.0
            )
        else:
            p_value = np.nan
            u_statistic = np.nan
            rank_biserial = np.nan

        stats_rows.append(
            {
                epitope_col: epitope,
                "group_1": status_1,
                "group_2": status_2,
                "n_group_1": len(values_1),
                "n_group_2": len(values_2),
                "median_group_1": (
                    float(np.median(values_1))
                    if len(values_1) > 0
                    else np.nan
                ),
                "median_group_2": (
                    float(np.median(values_2))
                    if len(values_2) > 0
                    else np.nan
                ),
                "median_difference": (
                    float(np.median(values_2) - np.median(values_1))
                    if len(values_1) > 0 and len(values_2) > 0
                    else np.nan
                ),
                "mannwhitney_u": u_statistic,
                "p_value": p_value,
                "rank_biserial_group_1_vs_group_2": rank_biserial,
            }
        )

    stats_df = pd.DataFrame(stats_rows)

    stats_df["p_adjusted"] = np.nan

    valid_p = stats_df["p_value"].notna()

    if valid_p.any():
        if p_adjust is None:
            stats_df.loc[valid_p, "p_adjusted"] = (
                stats_df.loc[valid_p, "p_value"]
            )
        else:
            stats_df.loc[valid_p, "p_adjusted"] = multipletests(
                stats_df.loc[valid_p, "p_value"],
                method=p_adjust,
            )[1]

    stats_df["significance"] = (
        stats_df["p_adjusted"].map(_p_to_label)
    )

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------

    n_epitopes = len(epitope_order)
    ncols = min(ncols, n_epitopes)
    nrows = int(np.ceil(n_epitopes / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            subplot_size[0] * ncols,
            subplot_size[1] * nrows,
        ),
        sharey=sharey,
        squeeze=False,
    )

    axes_flat = axes.ravel()
    rng = np.random.default_rng(random_state)

    for ax, epitope in zip(axes_flat, epitope_order):
        epitope_data = data[
            data[epitope_col].eq(epitope)
        ]

        box_values = []

        for status in status_order:
            values = (
                epitope_data.loc[
                    epitope_data["_status"].eq(status),
                    metric,
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            box_values.append(values)

        valid_positions = [
            position
            for position, values in enumerate(box_values, start=1)
            if len(values) > 0
        ]

        valid_values = [
            values
            for values in box_values
            if len(values) > 0
        ]

        if valid_values:
            ax.boxplot(
                valid_values,
                positions=valid_positions,
                widths=0.55,
                whis=(5, 95),
                showfliers=False,
            )

        for position, values in enumerate(box_values, start=1):
            if len(values) == 0:
                continue

            jitter = rng.normal(
                loc=position,
                scale=0.045,
                size=len(values),
            )

            ax.scatter(
                jitter,
                values,
                s=10,
                alpha=0.75,
                zorder=3,
            )

        if reference_line is not None:
            ax.axhline(
                reference_line,
                linestyle="--",
                linewidth=1,
                zorder=1,
            )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            status_order,
            rotation=30,
            ha="right",
        )

        ax.set_title(
            str(epitope),
            y=1.1,
            pad=4,
        )
        ax.set_xlabel("")
        
        if hide_top_right_spines:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Statistical annotation
        stat_row = stats_df.loc[
            stats_df[epitope_col].eq(epitope)
        ].iloc[0]

        plotted_values = np.concatenate(
            [
                values
                for values in box_values
                if len(values) > 0
            ]
        )

        if len(plotted_values) > 0:
            if ylim is not None:
                y_min, y_max = ylim

                if y_max <= y_min:
                    raise ValueError(
                        "The upper y-axis limit must be larger "
                        "than the lower y-axis limit."
                    )

                y_range = y_max - y_min

                # Position the annotation inside the fixed axis limits.
                bracket_y = y_min + 0.88 * y_range
                bracket_height = 0.02 * y_range
                text_y = y_min + 0.92 * y_range

            else:
                data_min = np.nanmin(plotted_values)
                data_max = np.nanmax(plotted_values)
                y_range = data_max - data_min

                if y_range == 0:
                    y_range = max(
                        abs(data_max) * 0.1,
                        0.05,
                    )

                bracket_y = data_max + 0.10 * y_range
                bracket_height = 0.035 * y_range
                text_y = bracket_y + 0.04 * y_range

            ax.plot(
                [1, 1, 2, 2],
                [
                    bracket_y,
                    bracket_y + bracket_height,
                    bracket_y + bracket_height,
                    bracket_y,
                ],
                linewidth=1,
                clip_on=False,
            )

            adjusted_p = stat_row["p_adjusted"]
            significance = stat_row["significance"]

            if stat_label == "stars":
                annotation = significance

            elif stat_label == "p":
                annotation = (
                    f"$p_{{adj}}$ = {adjusted_p:.3g}"
                    if pd.notna(adjusted_p)
                    else "NA"
                )

            elif stat_label == "both":
                annotation = (
                    f"{significance}\n"
                    f"$p_{{adj}}$ = {adjusted_p:.3g}"
                    if pd.notna(adjusted_p)
                    else "NA"
                )

            else:
                raise ValueError(
                    "stat_label must be 'stars', 'p', or 'both'."
                )

            ax.text(
                1.5,
                text_y,
                annotation,
                ha="center",
                va="bottom",
            )

            if ylim is None:
                ax.set_ylim(
                    top=text_y + 0.12 * y_range
                )
                
    if ylim is not None:
        ax.set_ylim(ylim)

    # Remove unused axes
    for ax in axes_flat[n_epitopes:]:
        ax.remove()

    # Show y-tick labels on every displayed axis
    for ax in axes_flat[:n_epitopes]:
        ax.tick_params(
            axis="y",
            which="both",
            labelleft=True,
        )

    for row_index in range(nrows):
        first_axis = axes[row_index, 0]

        if first_axis in fig.axes:
            first_axis.set_ylabel(ylabel or metric)

    if title is not None:
        fig.suptitle(title, y=1.1)

    fig.tight_layout()

    return fig, axes, stats_df