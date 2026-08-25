from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_STATUS_ORDER = ("Non-enriched", "Enriched")
DEFAULT_NON_ENRICHED_COLOR = "darkgray"


# =====================================================================
# Configuration
# =====================================================================


@dataclass(frozen=True)
class CDR3FlexConfig:
    """Configuration for the overall CDR3 flexibility-ratio analysis.

    The analysis is performed independently within each

        context × VJ combination × exact trimmed CDR3 length

    stratum. Only the overall positional-entropy flexibility ratio is
    calculated; no core/residual motif metrics are computed.

    Parameters
    ----------
    entropy_normalization:
        ``"maximum"`` divides every positional entropy by ln(20), the
        theoretical maximum for the 20 canonical amino acids.

        ``"observed"`` divides both epitope and background entropy at a
        position by ln(K_union), where K_union is the number of amino-acid
        states observed in the pooled epitope + matched-background sequences
        at that position. The same denominator is used for both repertoires.
    """

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

    # Remove conserved terminal CDR3 anchor residues.
    trim_left: int = 1
    trim_right: int = 1

    # Minimum numbers of unique sequences in each VJ × length stratum.
    min_epi_n: int = 8
    min_bg_n: int = 20

    # Entropy estimator and normalization.
    entropy_method: Literal["mle", "miller_madow"] = "miller_madow"
    entropy_normalization: Literal["maximum", "observed"] = "maximum"

    # Optional equal-size subsampling within each VJ × length stratum.
    equal_size_subsampling: bool = False
    subsample_n: int | None = None
    n_subsamples: int = 200

    random_state: int = 1


# =====================================================================
# Validation and input preparation
# =====================================================================


def _require_columns(
    dataframe: pd.DataFrame,
    required: Sequence[str],
    dataframe_name: str,
) -> None:
    missing = sorted(set(required).difference(dataframe.columns))
    if missing:
        raise KeyError(
            f"{dataframe_name} is missing required columns: {missing}"
        )


def _validate_config(cfg: CDR3FlexConfig) -> None:
    if len(cfg.epi_context_cols) != len(cfg.ann_context_cols):
        raise ValueError(
            "epi_context_cols and ann_context_cols must have equal length."
        )

    if cfg.trim_left < 0 or cfg.trim_right < 0:
        raise ValueError("trim_left and trim_right must be non-negative.")

    if cfg.min_epi_n < 1 or cfg.min_bg_n < 1:
        raise ValueError("min_epi_n and min_bg_n must be at least 1.")

    if cfg.entropy_method not in {"mle", "miller_madow"}:
        raise ValueError(
            "entropy_method must be 'mle' or 'miller_madow'."
        )

    if cfg.entropy_normalization not in {"maximum", "observed"}:
        raise ValueError(
            "entropy_normalization must be 'maximum' or 'observed'."
        )

    if cfg.equal_size_subsampling:
        if cfg.subsample_n is not None and cfg.subsample_n < 2:
            raise ValueError("subsample_n must be at least 2.")
        if cfg.n_subsamples < 1:
            raise ValueError("n_subsamples must be at least 1.")


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
    """Standardize, annotate, filter, trim, and deduplicate the inputs."""
    _validate_config(cfg)

    _require_columns(
        df_epi,
        [cfg.epi_cdr3_col, cfg.epi_vj_col, *cfg.epi_context_cols],
        "df_epi",
    )
    _require_columns(
        df_bg,
        [cfg.bg_cdr3_col, cfg.bg_vj_col],
        "df_bg",
    )
    _require_columns(
        df_ann,
        [cfg.ann_vj_col, cfg.ann_status_col, *cfg.ann_context_cols],
        "df_ann",
    )

    epi = pd.DataFrame(
        {
            "_cdr3_full": _clean_sequence_series(df_epi[cfg.epi_cdr3_col]),
            "_vj": df_epi[cfg.epi_vj_col].astype("string"),
        }
    )
    for column in cfg.epi_context_cols:
        epi[column] = df_epi[column].values

    bg = pd.DataFrame(
        {
            "_cdr3_full": _clean_sequence_series(df_bg[cfg.bg_cdr3_col]),
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

    status_counts = (
        ann.groupby(annotation_keys, dropna=False)["_status"]
        .nunique(dropna=False)
    )
    if (status_counts > 1).any():
        raise ValueError(
            "At least one context × VJ annotation maps to multiple statuses."
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

    epi = epi.loc[
        epi["_cdr3_full"].map(_valid_aa_sequence)
        & epi["_vj"].notna()
    ].copy()
    bg = bg.loc[
        bg["_cdr3_full"].map(_valid_aa_sequence)
        & bg["_vj"].notna()
    ].copy()

    minimum_length = cfg.trim_left + cfg.trim_right + 1
    epi = epi.loc[epi["_cdr3_full"].str.len() >= minimum_length].copy()
    bg = bg.loc[bg["_cdr3_full"].str.len() >= minimum_length].copy()

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

    # Each full sequence contributes once per context and VJ combination.
    epi = epi.drop_duplicates(
        [*cfg.epi_context_cols, "_vj", "_cdr3_full"]
    )
    bg = bg.drop_duplicates(["_vj", "_cdr3_full"])

    return epi.reset_index(drop=True), bg.reset_index(drop=True)


# =====================================================================
# Positional entropy and overall flexibility ratio
# =====================================================================


def _entropy_from_counts(
    counts: np.ndarray,
    n_observations: int,
    method: Literal["mle", "miller_madow"],
) -> float:
    if n_observations <= 0 or len(counts) == 0:
        return np.nan

    probabilities = counts.astype(float) / float(n_observations)
    entropy = float(-np.sum(probabilities * np.log(probabilities)))

    if method == "miller_madow":
        entropy += (len(counts) - 1) / (2.0 * n_observations)
    elif method != "mle":
        raise ValueError("method must be 'mle' or 'miller_madow'.")

    return entropy


def _normalization_denominator(
    pooled_residues: Sequence[str],
    mode: Literal["maximum", "observed"],
) -> tuple[float, int]:
    """Return the entropy denominator and its amino-acid state count."""
    if mode == "maximum":
        return float(np.log(20.0)), 20

    observed_states = len(set(pooled_residues))

    # A position with only one pooled amino-acid state has zero possible
    # entropy. It contributes normalized entropy 0 to both repertoires.
    if observed_states <= 1:
        return 0.0, observed_states

    return float(np.log(observed_states)), observed_states


def paired_positional_entropy_profile(
    epi_sequences: Sequence[str],
    bg_sequences: Sequence[str],
    entropy_method: Literal["mle", "miller_madow"] = "miller_madow",
    entropy_normalization: Literal["maximum", "observed"] = "maximum",
) -> pd.DataFrame:
    """Calculate matched epitope/background positional entropy.

    All sequences must have the same trimmed CDR3 length. Under observed-state
    normalization, both epitope and background entropy use the same pooled
    positional denominator ln(K_union), preventing group-specific scaling.
    """
    epi = np.asarray(list(epi_sequences), dtype=object)
    bg = np.asarray(list(bg_sequences), dtype=object)

    if len(epi) == 0 or len(bg) == 0:
        raise ValueError(
            "Both epitope and background sequence collections are required."
        )

    lengths = {len(sequence) for sequence in np.concatenate([epi, bg])}
    if len(lengths) != 1:
        raise ValueError(
            "All epitope and background sequences must have equal length."
        )

    cdr3_length = lengths.pop()
    rows: list[dict[str, object]] = []

    for position in range(cdr3_length):
        epi_residues = [sequence[position] for sequence in epi]
        bg_residues = [sequence[position] for sequence in bg]

        epi_counts = (
            pd.Series(epi_residues, dtype="string")
            .value_counts()
            .to_numpy(dtype=float)
        )
        bg_counts = (
            pd.Series(bg_residues, dtype="string")
            .value_counts()
            .to_numpy(dtype=float)
        )

        entropy_epi_raw = _entropy_from_counts(
            epi_counts,
            len(epi),
            entropy_method,
        )
        entropy_bg_raw = _entropy_from_counts(
            bg_counts,
            len(bg),
            entropy_method,
        )

        denominator, normalization_states = _normalization_denominator(
            [*epi_residues, *bg_residues],
            entropy_normalization,
        )

        if denominator > 0:
            # Miller-Madow correction can exceed the theoretical entropy for
            # the selected support slightly, so cap before normalization.
            entropy_epi_norm = min(entropy_epi_raw, denominator) / denominator
            entropy_bg_norm = min(entropy_bg_raw, denominator) / denominator
        else:
            entropy_epi_norm = 0.0
            entropy_bg_norm = 0.0

        rows.append(
            {
                "position": position + 1,
                "n_epi": len(epi),
                "n_bg": len(bg),
                "normalization": entropy_normalization,
                "normalization_states": normalization_states,
                "normalization_entropy": denominator,
                "entropy_epi_raw": entropy_epi_raw,
                "entropy_bg_raw": entropy_bg_raw,
                "entropy_epi_norm": float(entropy_epi_norm),
                "entropy_bg_norm": float(entropy_bg_norm),
            }
        )

    return pd.DataFrame(rows)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or denominator <= 0
    ):
        return np.nan
    return float(numerator / denominator)


def _summarize_profile(profile: pd.DataFrame) -> dict[str, float]:
    overall_entropy_epi = float(profile["entropy_epi_norm"].mean())
    overall_entropy_bg = float(profile["entropy_bg_norm"].mean())

    return {
        "overall_entropy_epi": overall_entropy_epi,
        "overall_entropy_bg": overall_entropy_bg,
        "overall_flexibility_ratio": _safe_ratio(
            overall_entropy_epi,
            overall_entropy_bg,
        ),
    }


def _analyze_stratum_once(
    epi_sequences: np.ndarray,
    bg_sequences: np.ndarray,
    cfg: CDR3FlexConfig,
) -> dict[str, float]:
    profile = paired_positional_entropy_profile(
        epi_sequences=epi_sequences,
        bg_sequences=bg_sequences,
        entropy_method=cfg.entropy_method,
        entropy_normalization=cfg.entropy_normalization,
    )
    return _summarize_profile(profile)


def analyze_stratum(
    epi_sequences: Sequence[str],
    bg_sequences: Sequence[str],
    cfg: CDR3FlexConfig,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Calculate the overall flexibility ratio for one matched stratum."""
    epi = np.asarray(list(epi_sequences), dtype=object)
    bg = np.asarray(list(bg_sequences), dtype=object)

    n_epi = len(epi)
    n_bg = len(bg)

    if n_epi == 0 or n_bg == 0:
        raise ValueError("Both strata must contain at least one sequence.")

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
            raise ValueError("subsample_n must be at least 2.")

        repeat_rows = []

        for repeat in range(cfg.n_subsamples):
            epi_sample = rng.choice(epi, size=n_target, replace=False)
            bg_sample = rng.choice(bg, size=n_target, replace=False)

            result = _analyze_stratum_once(
                epi_sequences=epi_sample,
                bg_sequences=bg_sample,
                cfg=cfg,
            )
            result["_repeat"] = repeat
            repeat_rows.append(result)

        repeat_df = pd.DataFrame(repeat_rows)

        overall_entropy_epi = float(
            repeat_df["overall_entropy_epi"].mean()
        )
        overall_entropy_bg = float(
            repeat_df["overall_entropy_bg"].mean()
        )

        # Ratio of the averaged entropies, matching the non-subsampled
        # definition and avoiding an average-of-ratios bias.
        overall_flexibility_ratio = _safe_ratio(
            overall_entropy_epi,
            overall_entropy_bg,
        )

    else:
        n_target = None
        result = _analyze_stratum_once(
            epi_sequences=epi,
            bg_sequences=bg,
            cfg=cfg,
        )
        overall_entropy_epi = result["overall_entropy_epi"]
        overall_entropy_bg = result["overall_entropy_bg"]
        overall_flexibility_ratio = result[
            "overall_flexibility_ratio"
        ]

    return {
        "n_epi": n_epi,
        "n_bg": n_bg,
        "subsample_n": n_target,
        "cdr3_length": len(epi[0]),
        "entropy_method": cfg.entropy_method,
        "entropy_normalization": cfg.entropy_normalization,
        "overall_entropy_epi": overall_entropy_epi,
        "overall_entropy_bg": overall_entropy_bg,
        "overall_flexibility_ratio": overall_flexibility_ratio,
    }


# =====================================================================
# Complete analysis pipeline
# =====================================================================


def run_cdr3_flexibility_analysis(
    df_epi: pd.DataFrame,
    df_bg: pd.DataFrame,
    df_ann: pd.DataFrame,
    cfg: CDR3FlexConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the overall flexibility-ratio analysis.

    Returns
    -------
    metrics_df:
        One row per analyzable context × VJ × exact CDR3-length stratum.

    skipped_df:
        Strata excluded because of insufficient epitope or background size.
    """
    epi, bg = prepare_inputs(
        df_epi=df_epi,
        df_bg=df_bg,
        df_ann=df_ann,
        cfg=cfg,
    )

    rng = np.random.default_rng(cfg.random_state)

    grouping_columns = [
        *cfg.epi_context_cols,
        "_vj",
        "_status",
        "_length",
    ]

    metric_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    for key, epi_group in epi.groupby(
        grouping_columns,
        dropna=False,
        sort=False,
    ):
        if not isinstance(key, tuple):
            key = (key,)

        metadata = dict(zip(grouping_columns, key))

        bg_group = bg.loc[
            bg["_vj"].eq(metadata["_vj"])
            & bg["_length"].eq(metadata["_length"])
        ]

        n_epi = len(epi_group)
        n_bg = len(bg_group)

        if n_epi < cfg.min_epi_n or n_bg < cfg.min_bg_n:
            skipped_rows.append(
                {
                    **metadata,
                    "n_epi": n_epi,
                    "n_bg": n_bg,
                    "reason": "insufficient_size",
                }
            )
            continue

        metrics = analyze_stratum(
            epi_sequences=epi_group["_cdr3"].tolist(),
            bg_sequences=bg_group["_cdr3"].tolist(),
            cfg=cfg,
            rng=rng,
        )
        metrics.update(metadata)
        metric_rows.append(metrics)

    return pd.DataFrame(metric_rows), pd.DataFrame(skipped_rows)


# Backward-compatible alias for the original function name.
run_cdr3b_flexibility_analysis = run_cdr3_flexibility_analysis


# =====================================================================
# Aggregate exact-length results to one result per VJ
# =====================================================================


def aggregate_metrics_to_vj(
    metrics_df: pd.DataFrame,
    context_cols: Sequence[str] = (),
    weight_by_sequence_count: bool = False,
) -> pd.DataFrame:
    """Aggregate length-specific ratios to one row per context × VJ.

    By default, each analyzable exact CDR3 length receives equal weight.
    With ``weight_by_sequence_count=True``, each length is weighted by its
    number of epitope-specific sequences.
    """
    required = {
        *context_cols,
        "_vj",
        "_status",
        "n_epi",
        "n_bg",
        "cdr3_length",
        "overall_flexibility_ratio",
    }
    _require_columns(metrics_df, required, "metrics_df")

    group_columns = [*context_cols, "_vj", "_status"]
    rows: list[dict[str, object]] = []

    for key, group in metrics_df.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        if not isinstance(key, tuple):
            key = (key,)

        row = dict(zip(group_columns, key))

        values = group["overall_flexibility_ratio"].to_numpy(dtype=float)
        weights = (
            group["n_epi"].to_numpy(dtype=float)
            if weight_by_sequence_count
            else np.ones(len(group), dtype=float)
        )

        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

        row["overall_flexibility_ratio"] = (
            float(np.average(values[valid], weights=weights[valid]))
            if valid.any()
            else np.nan
        )
        row["n_epi"] = int(group["n_epi"].sum())
        row["n_bg"] = int(group["n_bg"].sum())
        row["n_lengths"] = int(group["cdr3_length"].nunique())
        row["entropy_method"] = group["entropy_method"].iloc[0]
        row["entropy_normalization"] = group[
            "entropy_normalization"
        ].iloc[0]

        rows.append(row)

    return pd.DataFrame(rows)


# =====================================================================
# Statistical comparison
# =====================================================================


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


def compare_status_groups(
    vj_summary: pd.DataFrame,
    enriched_label: object,
    non_enriched_label: object,
    context_cols: Sequence[str] = (),
    p_adjust: str | None = None,
) -> pd.DataFrame:
    """Compare overall flexibility ratios using one VJ as one observation.

    A two-sided Mann-Whitney U test is calculated within each context.
    Positive rank-biserial values indicate larger ratios in enriched VJs.
    """
    required = {
        *context_cols,
        "_status",
        "overall_flexibility_ratio",
    }
    _require_columns(vj_summary, required, "vj_summary")

    grouped = (
        vj_summary.groupby(
            list(context_cols),
            dropna=False,
            sort=False,
        )
        if context_cols
        else [((), vj_summary)]
    )

    rows: list[dict[str, object]] = []

    for context_key, group in grouped:
        if not isinstance(context_key, tuple):
            context_key = (context_key,)
        context = dict(zip(context_cols, context_key))

        enriched = (
            pd.to_numeric(
                group.loc[
                    group["_status"].eq(enriched_label),
                    "overall_flexibility_ratio",
                ],
                errors="coerce",
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
        )
        non_enriched = (
            pd.to_numeric(
                group.loc[
                    group["_status"].eq(non_enriched_label),
                    "overall_flexibility_ratio",
                ],
                errors="coerce",
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(enriched) > 0 and len(non_enriched) > 0:
            test = mannwhitneyu(
                enriched,
                non_enriched,
                alternative="two-sided",
                method="auto",
            )
            u_statistic = float(test.statistic)
            p_value = float(test.pvalue)
            rank_biserial = (
                2.0
                * u_statistic
                / (len(enriched) * len(non_enriched))
                - 1.0
            )
        else:
            u_statistic = np.nan
            p_value = np.nan
            rank_biserial = np.nan

        rows.append(
            {
                **context,
                "metric": "overall_flexibility_ratio",
                "n_enriched_vj": len(enriched),
                "n_non_enriched_vj": len(non_enriched),
                "median_enriched": (
                    float(np.median(enriched))
                    if len(enriched) > 0
                    else np.nan
                ),
                "median_non_enriched": (
                    float(np.median(non_enriched))
                    if len(non_enriched) > 0
                    else np.nan
                ),
                "median_difference": (
                    float(np.median(enriched) - np.median(non_enriched))
                    if len(enriched) > 0 and len(non_enriched) > 0
                    else np.nan
                ),
                "mannwhitney_u": u_statistic,
                "p_value": p_value,
                "rank_biserial": rank_biserial,
            }
        )

    stats_df = pd.DataFrame(rows)
    stats_df["p_adjusted"] = np.nan

    valid = stats_df["p_value"].notna()
    if valid.any():
        raw_p = stats_df.loc[valid, "p_value"].to_numpy(dtype=float)
        adjusted = (
            raw_p
            if p_adjust is None
            else multipletests(raw_p, method=p_adjust)[1]
        )
        stats_df.loc[valid, "p_adjusted"] = adjusted

    stats_df["significance"] = stats_df["p_adjusted"].map(_p_to_label)
    return stats_df


# =====================================================================
# Matplotlib-only plotting
# =====================================================================


def _build_context_colors(
    contexts: Sequence[object],
    colors: Mapping[object, str] | None,
) -> dict[object, str]:
    resolved = dict(colors or {})
    defaults = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    for index, context in enumerate(contexts):
        resolved.setdefault(context, defaults[index % len(defaults)])
    return resolved


def plot_flexibility_ratio_by_context(
    dataframe: pd.DataFrame,
    context_col: str,
    context_order: Sequence[object] | None = None,
    status_order: Sequence[object] = DEFAULT_STATUS_ORDER,
    context_colors: Mapping[object, str] | None = None,
    non_enriched_color: str = DEFAULT_NON_ENRICHED_COLOR,
    ncols: int = 3,
    subplot_size: tuple[float, float] = (2.6, 3.8),
    ylim: tuple[float, float] | None = None,
    reference_line: float | None = 1.0,
    whis: tuple[float, float] = (5, 95),
    boxplot_width: float = 0.55,
    box_linewidth: float = 2.0,
    median_linewidth: float = 2.0,
    point_color: str = "black",
    point_size: float = 18,
    point_alpha: float = 0.75,
    point_jitter: float = 0.045,
    random_state: int = 1,
    p_adjust: str | None = "holm-sidak",
    stat_label: Literal["stars", "p", "both"] = "stars",
    save_pdf: bool = False,
    pdf_path: str | Path | None = None,
    show: bool = True,
) -> dict[str, object]:
    """Plot overall flexibility ratio by status for each context."""
    if len(status_order) != 2:
        raise ValueError("status_order must contain exactly two groups.")

    required = {context_col, "_status", "overall_flexibility_ratio"}
    _require_columns(dataframe, required, "dataframe")

    data = dataframe.copy()
    data["overall_flexibility_ratio"] = pd.to_numeric(
        data["overall_flexibility_ratio"],
        errors="coerce",
    )
    data = data.loc[
        data[context_col].notna()
        & data["_status"].isin(status_order)
        & data["overall_flexibility_ratio"].notna()
        & np.isfinite(data["overall_flexibility_ratio"])
    ].copy()

    if context_order is None:
        context_order = list(pd.unique(data[context_col]))
    else:
        observed = set(data[context_col])
        context_order = [value for value in context_order if value in observed]

    if not context_order:
        raise ValueError("No contexts remain after filtering.")

    resolved_colors = _build_context_colors(context_order, context_colors)

    stats_rows = []
    group_1, group_2 = status_order

    for context in context_order:
        subset = data.loc[data[context_col].eq(context)]
        values_1 = subset.loc[
            subset["_status"].eq(group_1),
            "overall_flexibility_ratio",
        ].to_numpy(dtype=float)
        values_2 = subset.loc[
            subset["_status"].eq(group_2),
            "overall_flexibility_ratio",
        ].to_numpy(dtype=float)

        if len(values_1) > 0 and len(values_2) > 0:
            test = mannwhitneyu(
                values_1,
                values_2,
                alternative="two-sided",
                method="auto",
            )
            u = float(test.statistic)
            p = float(test.pvalue)
            rank_biserial = 2.0 * u / (len(values_1) * len(values_2)) - 1.0
        else:
            u = np.nan
            p = np.nan
            rank_biserial = np.nan

        stats_rows.append(
            {
                context_col: context,
                "group_1": group_1,
                "group_2": group_2,
                "n_group_1": len(values_1),
                "n_group_2": len(values_2),
                "median_group_1": (
                    float(np.median(values_1)) if len(values_1) else np.nan
                ),
                "median_group_2": (
                    float(np.median(values_2)) if len(values_2) else np.nan
                ),
                "median_difference": (
                    float(np.median(values_2) - np.median(values_1))
                    if len(values_1) and len(values_2)
                    else np.nan
                ),
                "mannwhitney_u": u,
                "p_value": p,
                "rank_biserial_group_1_vs_group_2": rank_biserial,
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    stats_df["p_adjusted"] = np.nan
    valid = stats_df["p_value"].notna()
    if valid.any():
        raw = stats_df.loc[valid, "p_value"].to_numpy(dtype=float)
        adjusted = (
            raw
            if p_adjust is None
            else multipletests(raw, method=p_adjust)[1]
        )
        stats_df.loc[valid, "p_adjusted"] = adjusted
    stats_df["significance"] = stats_df["p_adjusted"].map(_p_to_label)

    n_contexts = len(context_order)
    ncols = min(ncols, n_contexts)
    nrows = int(np.ceil(n_contexts / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(subplot_size[0] * ncols, subplot_size[1] * nrows),
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    rng = np.random.default_rng(random_state)

    for ax, context in zip(axes_flat, context_order):
        subset = data.loc[data[context_col].eq(context)]
        context_color = resolved_colors[context]
        group_colors = {
            group_1: non_enriched_color,
            group_2: context_color,
        }

        box_values = [
            subset.loc[
                subset["_status"].eq(status),
                "overall_flexibility_ratio",
            ].to_numpy(dtype=float)
            for status in status_order
        ]

        valid_entries = [
            (position, status, values)
            for position, (status, values) in enumerate(
                zip(status_order, box_values),
                start=1,
            )
            if len(values) > 0
        ]

        if valid_entries:
            box = ax.boxplot(
                [entry[2] for entry in valid_entries],
                positions=[entry[0] for entry in valid_entries],
                widths=boxplot_width,
                whis=whis,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": "none", "linewidth": box_linewidth},
                whiskerprops={"linewidth": box_linewidth},
                capprops={"linewidth": box_linewidth},
                medianprops={
                    "color": context_color,
                    "linewidth": median_linewidth,
                },
            )

            for index, (_, status, _) in enumerate(valid_entries):
                color = group_colors[status]
                box["boxes"][index].set_edgecolor(color)
                for whisker in box["whiskers"][2 * index : 2 * index + 2]:
                    whisker.set_color(color)
                for cap in box["caps"][2 * index : 2 * index + 2]:
                    cap.set_color(color)

        for position, values in enumerate(box_values, start=1):
            if len(values) == 0:
                continue
            jitter = rng.normal(position, point_jitter, len(values))
            ax.scatter(
                jitter,
                values,
                s=point_size,
                color=point_color,
                alpha=point_alpha,
                linewidths=0,
                zorder=3,
            )

        if reference_line is not None:
            ax.axhline(
                reference_line,
                color="black",
                linestyle="--",
                linewidth=1,
                zorder=0,
            )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(status_order, rotation=30, ha="right")
        ax.set_xlim(0.5, 2.5)
        if ylim is not None:
            ax.set_ylim(ylim)

        stat_row = stats_df.loc[stats_df[context_col].eq(context)].iloc[0]
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
                f"{significance}\n$p_{{adj}}$ = {adjusted_p:.3g}"
                if pd.notna(adjusted_p)
                else "NA"
            )
        else:
            raise ValueError("stat_label must be 'stars', 'p', or 'both'.")

        ax.plot(
            [1, 1, 2, 2],
            [0.82, 0.84, 0.84, 0.82],
            transform=ax.get_xaxis_transform(),
            color="black",
            linewidth=1,
            clip_on=False,
        )
        ax.text(
            1.5,
            0.86,
            annotation,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
        )

        ax.set_title(str(context), y=1.06, pad=4)
        ax.set_xlabel("")
        ax.set_ylabel("Overall flexibility ratio")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", which="both", labelleft=True)

    for ax in axes_flat[n_contexts:]:
        ax.remove()

    fig.tight_layout()

    if save_pdf:
        output_path = Path(
            pdf_path or "overall_cdr3_flexibility_ratio.pdf"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="pdf", bbox_inches="tight")

    if show:
        plt.show()

    return {
        "fig": fig,
        "axes": axes,
        "stats": stats_df,
        "plot_data": data,
    }


__all__ = [
    "AA20",
    "CDR3FlexConfig",
    "DEFAULT_NON_ENRICHED_COLOR",
    "DEFAULT_STATUS_ORDER",
    "aggregate_metrics_to_vj",
    "analyze_stratum",
    "compare_status_groups",
    "paired_positional_entropy_profile",
    "plot_flexibility_ratio_by_context",
    "prepare_inputs",
    "run_cdr3_flexibility_analysis",
    "run_cdr3b_flexibility_analysis",
]
