
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


DEFAULT_GROUP_ORDER = ("Non-enriched", "Enriched")
DEFAULT_NON_ENRICHED_COLOR = "darkgray"
DEFAULT_ENRICHED_COLOR = "#1F77B4"


# helper functions


def _as_tuple(values: Sequence[str] | str) -> tuple[str, ...]:
    """Convert a single string or sequence of strings to a tuple."""
    if isinstance(values, str):
        return (values,)
    return tuple(values)


def _require_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    dataframe_name: str,
) -> None:
    missing = set(columns).difference(dataframe.columns)
    if missing:
        raise KeyError(
            f"{dataframe_name} is missing required columns: {sorted(missing)}"
        )


def _safe_numeric_array(values: Sequence[Any]) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def _resolve_epitope_palette(
    epitope: object,
    group_order: Sequence[object],
    palette: Mapping[object, str] | None,
    epitope_colors: Mapping[object, str] | None,
    non_enriched_color: str,
    enriched_color: str,
) -> dict[object, str]:
    """Return colors for the two enrichment groups for one epitope."""
    if len(group_order) != 2:
        raise ValueError("group_order must contain exactly two group labels.")

    group_1, group_2 = group_order

    if palette is not None:
        resolved = {
            group_1: palette.get(group_1, non_enriched_color),
            group_2: palette.get(group_2, enriched_color),
        }
    else:
        resolved = {
            group_1: non_enriched_color,
            group_2: enriched_color,
        }

    if epitope_colors is not None and epitope in epitope_colors:
        resolved[group_2] = epitope_colors[epitope]

    return resolved


def _darken_color(color: str, factor: float = 0.65) -> tuple[float, float, float]:
    """Darken a Matplotlib-compatible color without seaborn."""
    from matplotlib.colors import to_rgb

    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple(np.clip(rgb * factor, 0.0, 1.0))


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


# shannon calculation


def category_counts(values: Sequence[Any]) -> np.ndarray:
    """Return occurrence counts of non-missing categorical values."""
    series = pd.Series(values).dropna()
    if series.empty:
        return np.asarray([], dtype=float)
    return series.astype(str).value_counts().to_numpy(dtype=float)


def shannon_entropy_from_counts(counts: Sequence[float]) -> float:
    """
    Empirical Shannon entropy in nats.

    H = -sum(p_i * ln(p_i))
    """
    counts_array = np.asarray(counts, dtype=float)
    counts_array = counts_array[np.isfinite(counts_array) & (counts_array > 0)]

    if counts_array.size == 0:
        return np.nan

    probabilities = counts_array / counts_array.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def corrected_shannon_entropy_from_counts(counts: Sequence[float]) -> float:
    """
    Miller-Madow-corrected Shannon entropy in nats.

    H_MM = H_MLE + (K - 1) / (2N)

    K is the number of observed categories and N is the total number of
    observations. The correction is zero for a single observed category.
    """
    counts_array = np.asarray(counts, dtype=float)
    counts_array = counts_array[np.isfinite(counts_array) & (counts_array > 0)]

    if counts_array.size == 0:
        return np.nan

    entropy_mle = shannon_entropy_from_counts(counts_array)
    n_observations = float(counts_array.sum())
    n_observed_categories = int(counts_array.size)

    correction = (n_observed_categories - 1) / (2.0 * n_observations)
    return float(entropy_mle + correction)


def shannon_entropy_from_values(values: Sequence[Any]) -> float:
    return shannon_entropy_from_counts(category_counts(values))


def corrected_shannon_entropy_from_values(values: Sequence[Any]) -> float:
    return corrected_shannon_entropy_from_counts(category_counts(values))




#k-mer extraction


def trim_sequence(sequence: Any, start: int = 0, end: int = 0) -> str:
    """Trim amino acids from the left and right ends of a sequence."""
    if start < 0 or end < 0:
        raise ValueError("start and end must be non-negative integers.")

    sequence_string = str(sequence)

    if end == 0:
        return sequence_string[start:]

    if start + end >= len(sequence_string):
        return ""

    return sequence_string[start:-end]


def extract_kmers_from_sequence(
    sequence: Any,
    k: int = 3,
    start: int = 0,
    end: int = 0,
) -> list[str]:
    """Extract overlapping k-mers after optional terminal trimming."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    core = trim_sequence(sequence, start=start, end=end)

    if len(core) < k:
        return []

    return [core[index : index + k] for index in range(len(core) - k + 1)]


def extract_kmers_from_sequences(
    sequences: Sequence[Any],
    k: int = 3,
    start: int = 0,
    end: int = 0,
) -> list[str]:
    """Extract and pool overlapping k-mers from all non-missing sequences."""
    kmers: list[str] = []

    for sequence in pd.Series(sequences).dropna().astype(str):
        kmers.extend(
            extract_kmers_from_sequence(
                sequence=sequence,
                k=k,
                start=start,
                end=end,
            )
        )

    return kmers


def kmer_shannon_entropy(
    sequences: Sequence[Any],
    k: int = 3,
    start: int = 0,
    end: int = 0,
) -> float:
    kmers = extract_kmers_from_sequences(sequences, k=k, start=start, end=end)
    return shannon_entropy_from_values(kmers)


def kmer_corrected_shannon_entropy(
    sequences: Sequence[Any],
    k: int = 3,
    start: int = 0,
    end: int = 0,
) -> float:
    kmers = extract_kmers_from_sequences(sequences, k=k, start=start, end=end)
    return corrected_shannon_entropy_from_values(kmers)




# V-J grouping


def compute_kmer_and_cdr3_shannon_per_framework(
    df_tcr_subset: pd.DataFrame,
    df_enrich: pd.DataFrame,
    enrich_col: str = "trb_vj",
    paired_col: str = "tra_vj",
    cdr3_col: str = "cdr3b_aa",
    enrich_col_enrich: str = "gene_recombination",
    group_col: str = "enrichment_group",
    k: int = 3,
    trim_start: int = 0,
    trim_end: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate full-sequence, paired-chain, full-k-mer and trimmed-k-mer
    diversity for each enriched-chain VJ combination.

    Full k-mer metrics always use start=0 and end=0. Trimmed k-mer metrics use
    trim_start and trim_end. When both are zero, full and trimmed metrics are
    identical by definition; retaining both columns keeps the output schema
    stable across analyses.
    """
    _require_columns(
        df_tcr_subset,
        [enrich_col, paired_col, cdr3_col],
        "df_tcr_subset",
    )
    _require_columns(
        df_enrich,
        [enrich_col_enrich, group_col],
        "df_enrich",
    )

    if k < 1:
        raise ValueError("k must be at least 1.")
    if trim_start < 0 or trim_end < 0:
        raise ValueError("trim_start and trim_end must be non-negative.")

    group_map = (
        df_enrich[[enrich_col_enrich, group_col]]
        .drop_duplicates()
        .rename(columns={enrich_col_enrich: enrich_col})
    )

    contradictory = group_map.groupby(enrich_col, dropna=False)[group_col].nunique()
    if (contradictory > 1).any():
        bad_values = contradictory[contradictory > 1].index.tolist()
        raise ValueError(
            "Some enriched-chain VJ combinations map to more than one "
            f"enrichment group: {bad_values}"
        )

    group_map = group_map.drop_duplicates(subset=[enrich_col])

    df_work = (
        df_tcr_subset.copy()
        .merge(
            group_map,
            on=enrich_col,
            how="inner",
            validate="many_to_one",
        )
        .dropna(subset=[enrich_col, paired_col, cdr3_col, group_col])
        .copy()
    )

    rows: list[dict[str, Any]] = []

    for (framework, enrichment_group), group in df_work.groupby(
        [enrich_col, group_col],
        dropna=False,
        sort=False,
    ):
        cdr3_values = group[cdr3_col]
        paired_values = group[paired_col]

        full_kmers = extract_kmers_from_sequences(
            cdr3_values,
            k=k,
            start=0,
            end=0,
        )
        trimmed_kmers = extract_kmers_from_sequences(
            cdr3_values,
            k=k,
            start=trim_start,
            end=trim_end,
        )

        rows.append(
            {
                enrich_col: framework,
                group_col: enrichment_group,
                "n_tcr": int(len(group)),
                "n_unique_cdr3": int(cdr3_values.astype(str).nunique()),
                "n_unique_paired": int(paired_values.astype(str).nunique()),
                "n_full_kmers": int(len(full_kmers)),
                "n_unique_full_kmers": int(pd.Series(full_kmers).nunique()),
                "n_trimmed_kmers": int(len(trimmed_kmers)),
                "n_unique_trimmed_kmers": int(pd.Series(trimmed_kmers).nunique()),
                # Complete CDR3 sequence categories
                "cdr3_shannon_entropy": shannon_entropy_from_values(cdr3_values),
                "cdr3_shannon_entropy_corrected": (
                    corrected_shannon_entropy_from_values(cdr3_values)
                ),
                # Paired-chain VJ categories
                "paired_shannon_entropy": shannon_entropy_from_values(paired_values),
                "paired_shannon_entropy_corrected": (
                    corrected_shannon_entropy_from_values(paired_values)
                ),
                # Full, untrimmed k-mer categories
                "kmer_full_shannon_entropy": shannon_entropy_from_values(full_kmers),
                "kmer_full_shannon_entropy_corrected": (
                    corrected_shannon_entropy_from_values(full_kmers)
                ),
                # Trimmed k-mer categories
                "kmer_trimmed_shannon_entropy": shannon_entropy_from_values(
                    trimmed_kmers
                ),
                "kmer_trimmed_shannon_entropy_corrected": (
                    corrected_shannon_entropy_from_values(trimmed_kmers)
                ),
            }
        )

    diversity = pd.DataFrame(rows)

    if not diversity.empty:
        diversity["k"] = int(k)
        diversity["trim_start"] = int(trim_start)
        diversity["trim_end"] = int(trim_end)
        diversity["enrich_col"] = enrich_col
        diversity["paired_col"] = paired_col
        diversity["cdr3_col"] = cdr3_col

    return diversity, df_work


def run_kmer_and_cdr3_shannon_for_epitopes(
    df_tcr: pd.DataFrame,
    df_enrich_all: pd.DataFrame,
    epitopes: Sequence[object],
    enrich_col: str = "trb_vj",
    paired_col: str = "tra_vj",
    cdr3_col: str = "cdr3b_aa",
    enrich_col_enrich: str = "gene_recombination",
    epitope_col_tcr: str = "annotated_specificity",
    epitope_col_enrich: str = "epitope",
    group_col: str = "enrichment_group",
    k: int = 3,
    trim_start: int = 0,
    trim_end: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the framework-level analysis independently for each epitope."""
    _require_columns(df_tcr, [epitope_col_tcr], "df_tcr")
    _require_columns(df_enrich_all, [epitope_col_enrich], "df_enrich_all")

    diversity_frames: list[pd.DataFrame] = []
    work_frames: list[pd.DataFrame] = []

    for epitope in epitopes:
        diversity, df_work = compute_kmer_and_cdr3_shannon_per_framework(
            df_tcr_subset=df_tcr.loc[df_tcr[epitope_col_tcr].eq(epitope)],
            df_enrich=df_enrich_all.loc[df_enrich_all[epitope_col_enrich].eq(epitope)],
            enrich_col=enrich_col,
            paired_col=paired_col,
            cdr3_col=cdr3_col,
            enrich_col_enrich=enrich_col_enrich,
            group_col=group_col,
            k=k,
            trim_start=trim_start,
            trim_end=trim_end,
        )

        diversity["epitope"] = epitope
        df_work = df_work.copy()
        df_work["epitope"] = epitope

        diversity_frames.append(diversity)
        work_frames.append(df_work)

    diversity_all = (
        pd.concat(diversity_frames, ignore_index=True)
        if diversity_frames
        else pd.DataFrame()
    )
    df_work_all = (
        pd.concat(work_frames, ignore_index=True)
        if work_frames
        else pd.DataFrame()
    )

    return diversity_all, df_work_all


# statistical testing


def mannwhitney_metric_by_epitope(
    diversity_all: pd.DataFrame,
    metrics: Sequence[str],
    epitope_col: str = "epitope",
    group_col: str = "enrichment_group",
    group_order: Sequence[object] = DEFAULT_GROUP_ORDER,
    p_adjust: str | None = "holm",
) -> pd.DataFrame:
    """
    Compare enrichment groups for each epitope and metric.

    Multiple-testing correction is applied across epitopes separately for each
    metric. Positive median_difference means group_order[1] is larger than
    group_order[0]. Positive rank_biserial_group_1_vs_group_2 means values tend
    to be larger in group_order[0].
    """
    if len(group_order) != 2:
        raise ValueError("group_order must contain exactly two groups.")

    metrics = _as_tuple(metrics)
    _require_columns(
        diversity_all,
        [epitope_col, group_col, *metrics],
        "diversity_all",
    )

    group_1, group_2 = group_order
    rows: list[dict[str, Any]] = []

    epitope_order = list(pd.unique(diversity_all[epitope_col].dropna()))

    for metric in metrics:
        for epitope in epitope_order:
            subset = diversity_all.loc[diversity_all[epitope_col].eq(epitope)]

            values_1 = _safe_numeric_array(
                subset.loc[subset[group_col].eq(group_1), metric]
            )
            values_2 = _safe_numeric_array(
                subset.loc[subset[group_col].eq(group_2), metric]
            )

            if values_1.size > 0 and values_2.size > 0:
                test = mannwhitneyu(
                    values_1,
                    values_2,
                    alternative="two-sided",
                    method="auto",
                )
                u_statistic = float(test.statistic)
                p_value = float(test.pvalue)
                rank_biserial = float(
                    2.0 * u_statistic / (values_1.size * values_2.size) - 1.0
                )
            else:
                u_statistic = np.nan
                p_value = np.nan
                rank_biserial = np.nan

            median_1 = float(np.median(values_1)) if values_1.size else np.nan
            median_2 = float(np.median(values_2)) if values_2.size else np.nan

            rows.append(
                {
                    "epitope": epitope,
                    "metric": metric,
                    "group_1": group_1,
                    "group_2": group_2,
                    "n_group_1": int(values_1.size),
                    "n_group_2": int(values_2.size),
                    "median_group_1": median_1,
                    "median_group_2": median_2,
                    "median_difference": (
                        median_2 - median_1
                        if np.isfinite(median_1) and np.isfinite(median_2)
                        else np.nan
                    ),
                    "mannwhitney_u": u_statistic,
                    "p_value": p_value,
                    "rank_biserial_group_1_vs_group_2": rank_biserial,
                }
            )

    stats_df = pd.DataFrame(rows)
    stats_df["p_adjusted"] = np.nan

    if not stats_df.empty:
        for metric, metric_index in stats_df.groupby("metric", sort=False).groups.items():
            metric_index = list(metric_index)
            valid_index = [
                index
                for index in metric_index
                if pd.notna(stats_df.at[index, "p_value"])
            ]

            if not valid_index:
                continue

            raw_p_values = stats_df.loc[valid_index, "p_value"].to_numpy(dtype=float)

            if p_adjust is None:
                adjusted = raw_p_values
            else:
                adjusted = multipletests(raw_p_values, method=p_adjust)[1]

            stats_df.loc[valid_index, "p_adjusted"] = adjusted

    stats_df["significance"] = stats_df["p_adjusted"].map(_p_to_label)
    stats_df["p_adjust_method"] = p_adjust if p_adjust is not None else "none"

    return stats_df


# plotting functions


def _format_stat_annotation(
    stat_row: pd.Series,
    stat_label: str,
) -> str:
    adjusted_p = stat_row.get("p_adjusted", np.nan)
    significance = stat_row.get("significance", "NA")

    if stat_label == "stars":
        return str(significance)

    if stat_label == "p":
        return f"$p_{{adj}}$ = {adjusted_p:.3g}" if pd.notna(adjusted_p) else "NA"

    if stat_label == "both":
        if pd.isna(adjusted_p):
            return "NA"
        return f"{significance}\n$p_{{adj}}$ = {adjusted_p:.3g}"

    raise ValueError("stat_label must be 'stars', 'p', or 'both'.")


def plot_metric_boxplot(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    epitope: object,
    stat_row: pd.Series | None,
    group_col: str = "enrichment_group",
    group_order: Sequence[object] = DEFAULT_GROUP_ORDER,
    palette: Mapping[object, str] | None = None,
    whis: tuple[float, float] = (5, 95),
    boxplot_width: float = 0.55,
    box_linewidth: float = 2.0,
    median_color: str | None = None,
    median_linewidth: float = 2.0,
    point_color: str = "black",
    point_size: float = 24,
    point_alpha: float = 0.75,
    point_jitter: float = 0.045,
    xlim: tuple[float, float] = (0.5, 2.5),
    ylim: tuple[float, float] | None = None,
    stat_label: str = "stars",
    stat_bracket_y: float = 0.80,
    stat_text_y: float = 0.84,
    title_y: float = 1.06,
    random_state: int = 1,
    xlabel_rotation: float = 30,
) -> None:
    """Draw one flexibility-style boxplot with all observations shown.

    If median_color is None, both median lines use the epitope-specific
    enriched-group color. Passing a color explicitly overrides this default.
    """
    if len(group_order) != 2:
        raise ValueError("group_order must contain exactly two groups.")

    if palette is None:
        palette = {
            group_order[0]: DEFAULT_NON_ENRICHED_COLOR,
            group_order[1]: DEFAULT_ENRICHED_COLOR,
        }

    resolved_median_color = (
        palette[group_order[1]]
        if median_color is None
        else median_color
    )

    plot_data = data.copy()
    plot_data[metric] = pd.to_numeric(plot_data[metric], errors="coerce")
    plot_data = plot_data.loc[
        plot_data[metric].notna()
        & np.isfinite(plot_data[metric])
        & plot_data[group_col].isin(group_order)
    ].copy()

    values_by_group = [
        plot_data.loc[plot_data[group_col].eq(group), metric].to_numpy(dtype=float)
        for group in group_order
    ]

    valid_entries = [
        {
            "position": position,
            "values": values,
            "color": palette[group],
        }
        for position, (group, values) in enumerate(
            zip(group_order, values_by_group),
            start=1,
        )
        if values.size > 0
    ]

    valid_positions = [
        entry["position"]
        for entry in valid_entries
    ]
    valid_values = [
        entry["values"]
        for entry in valid_entries
    ]
    valid_colors = [
        entry["color"]
        for entry in valid_entries
    ]

    if valid_values:
        boxplot = ax.boxplot(
            valid_values,
            positions=valid_positions,
            widths=boxplot_width,
            whis=whis,
            showfliers=False,
            patch_artist=True,
            boxprops={
                "facecolor": "none",
                "linewidth": box_linewidth,
            },
            whiskerprops={
                "linewidth": box_linewidth,
            },
            capprops={
                "linewidth": box_linewidth,
            },
            medianprops={
                "color": resolved_median_color,
                "linewidth": median_linewidth,
            },
        )

        for index, color in enumerate(valid_colors):
            boxplot["boxes"][index].set_edgecolor(color)

            for whisker in boxplot["whiskers"][
                2 * index : 2 * index + 2
            ]:
                whisker.set_color(color)
                whisker.set_linewidth(box_linewidth)

            for cap in boxplot["caps"][
                2 * index : 2 * index + 2
            ]:
                cap.set_color(color)
                cap.set_linewidth(box_linewidth)

            boxplot["medians"][index].set_color(resolved_median_color)
            boxplot["medians"][index].set_linewidth(
                median_linewidth
            )

    panel_seed = (
        int(random_state)
        + sum(ord(character) for character in f"{epitope}|{metric}")
    ) % (2**32 - 1)
    rng = np.random.default_rng(panel_seed)

    for position, values in enumerate(values_by_group, start=1):
        if values.size == 0:
            continue

        jitter = rng.normal(
            loc=position,
            scale=point_jitter,
            size=values.size,
        )
        ax.scatter(
            jitter,
            values,
            s=point_size,
            alpha=point_alpha,
            color=point_color,
            linewidths=0,
            zorder=3,
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(group_order, rotation=xlabel_rotation, ha="right")
    ax.set_xlim(xlim)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{epitope}: {ylabel}", y=title_y, pad=4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", which="both", labelleft=True, labelsize = 14)

    non_empty_values = [values for values in values_by_group if values.size > 0]

    if non_empty_values and stat_row is not None:
        plotted_values = np.concatenate(non_empty_values)

        if ylim is not None:
            y_min, y_max = ylim
            if y_max <= y_min:
                raise ValueError("The upper y-axis limit must exceed the lower limit.")
            y_range = y_max - y_min
            bracket_y = y_min + stat_bracket_y * y_range
            bracket_height = 0.02 * y_range
            text_y = y_min + stat_text_y * y_range
        else:
            data_min = float(np.nanmin(plotted_values))
            data_max = float(np.nanmax(plotted_values))
            y_range = data_max - data_min
            if y_range <= 0:
                y_range = max(abs(data_max) * 0.1, 0.05)
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
            color="black",
            linewidth=1,
            clip_on=False,
        )
        ax.text(
            1.5,
            text_y,
            _format_stat_annotation(stat_row, stat_label),
            ha="center",
            va="bottom",
        )

        if ylim is None:
            ax.set_ylim(top=text_y + 0.12 * y_range)

    if ylim is not None:
        ax.set_ylim(ylim)


# running


def default_metric_specs(
    k: int,
    trim_start: int,
    trim_end: int,
    include_cdr3: bool = True,
    include_paired: bool = False,
) -> list[tuple[str, str]]:
    """Return default plot metrics, including full and trimmed k-mer results."""
    trimmed_region = (
        "Full CDR3"
        if trim_start == 0 and trim_end == 0
        else f"CDR3[+{trim_start} : -{trim_end}]"
    )

    metrics: list[tuple[str, str]] = [
        ("kmer_full_shannon_entropy", f"{k}-mer Shannon\nFull CDR3"),
        (
            "kmer_full_shannon_entropy_corrected",
            f"{k}-mer corrected Shannon\nFull CDR3",
        ),
        ("kmer_trimmed_shannon_entropy", f"{k}-mer Shannon\n{trimmed_region}"),
        (
            "kmer_trimmed_shannon_entropy_corrected",
            f"{k}-mer corrected Shannon\n{trimmed_region}",
        ),
    ]

    if include_cdr3:
        metrics.extend(
            [
                ("cdr3_shannon_entropy", "CDR3 Shannon"),
                ("cdr3_shannon_entropy_corrected", "CDR3 corrected Shannon"),
            ]
        )

    if include_paired:
        metrics.extend(
            [
                ("paired_shannon_entropy", "Paired-chain Shannon"),
                (
                    "paired_shannon_entropy_corrected",
                    "Paired-chain corrected Shannon",
                ),
            ]
        )

    return metrics


def summarize_diversity(
    diversity_all: pd.DataFrame,
    metrics: Sequence[str],
    enrich_col: str,
    group_col: str,
) -> pd.DataFrame:
    """Return group-level medians for all plotted metrics."""
    metrics = _as_tuple(metrics)

    aggregations: dict[str, tuple[str, str]] = {
        "n_frameworks": (enrich_col, "size"),
        "median_tcr_per_framework": ("n_tcr", "median"),
    }

    for metric in metrics:
        aggregations[f"median_{metric}"] = (metric, "median")

    return (
        diversity_all.groupby(["epitope", group_col], dropna=False)
        .agg(**aggregations)
        .reset_index()
    )


def plot_kmer_and_cdr3_shannon_all_epitopes(
    df_tcr: pd.DataFrame,
    df_enrich_all: pd.DataFrame,
    epitopes: Sequence[object],
    enrich_col: str = "trb_vj",
    paired_col: str = "tra_vj",
    cdr3_col: str = "cdr3b_aa",
    enrich_col_enrich: str = "gene_recombination",
    epitope_col_tcr: str = "annotated_specificity",
    epitope_col_enrich: str = "epitope",
    group_col: str = "enrichment_group",
    group_order: Sequence[object] = DEFAULT_GROUP_ORDER,
    palette: Mapping[object, str] | None = None,
    epitope_colors: Mapping[object, str] | None = None,
    non_enriched_color: str = DEFAULT_NON_ENRICHED_COLOR,
    enriched_color: str = DEFAULT_ENRICHED_COLOR,
    k: int = 3,
    trim_start: int = 3,
    trim_end: int = 3,
    metrics: Sequence[tuple[str, str]] | None = None,
    include_cdr3: bool = True,
    include_paired: bool = False,
    box_width: float = 2.4,
    subplot_height: float = 3.2,
    whis: tuple[float, float] = (5, 95),
    boxplot_width: float = 0.55,
    box_linewidth: float = 2.0,
    median_color: str | None = None,
    median_linewidth: float = 2.0,
    point_color: str = "black",
    point_size: float = 24,
    point_alpha: float = 0.75,
    point_jitter: float = 0.045,
    xlim: tuple[float, float] = (0.5, 2.5),
    ylim_dict: Mapping[str, tuple[float, float] | None] | None = None,
    p_adjust: str | None = "holm",
    stat_label: str = "stars",
    stat_bracket_y: float = 0.80,
    stat_text_y: float = 0.84,
    title_y: float = 1.06,
    random_state: int = 1,
    title: str | None = None,
    save_pdf: bool = False,
    pdf_path: str | Path | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """
    Run the complete analysis and plot one epitope per row.

    Default columns include both untrimmed and trimmed k-mer Shannon and
    Miller-Madow-corrected Shannon metrics. Complete-CDR3 metrics are
    included by default. Paired-chain metrics are calculated but plotted only
    when include_paired=True or explicitly supplied through metrics.
    """
    epitopes = tuple(epitopes)
    group_order = tuple(group_order)

    if len(epitopes) == 0:
        raise ValueError("At least one epitope is required.")
    if len(group_order) != 2:
        raise ValueError("group_order must contain exactly two groups.")

    if metrics is None:
        metric_specs = default_metric_specs(
            k=k,
            trim_start=trim_start,
            trim_end=trim_end,
            include_cdr3=include_cdr3,
            include_paired=include_paired,
        )
    else:
        metric_specs = list(metrics)

    metric_names = [metric for metric, _ in metric_specs]

    diversity_all, df_work_all = run_kmer_and_cdr3_shannon_for_epitopes(
        df_tcr=df_tcr,
        df_enrich_all=df_enrich_all,
        epitopes=epitopes,
        enrich_col=enrich_col,
        paired_col=paired_col,
        cdr3_col=cdr3_col,
        enrich_col_enrich=enrich_col_enrich,
        epitope_col_tcr=epitope_col_tcr,
        epitope_col_enrich=epitope_col_enrich,
        group_col=group_col,
        k=k,
        trim_start=trim_start,
        trim_end=trim_end,
    )

    if diversity_all.empty:
        raise ValueError("No analyzable framework-level data were generated.")

    _require_columns(diversity_all, metric_names, "diversity_all")

    stats_df = mannwhitney_metric_by_epitope(
        diversity_all=diversity_all,
        metrics=metric_names,
        epitope_col="epitope",
        group_col=group_col,
        group_order=group_order,
        p_adjust=p_adjust,
    )

    n_rows = len(epitopes)
    n_cols = len(metric_specs)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(box_width * n_cols, subplot_height * n_rows),
        squeeze=False,
        sharey=False,
    )

    for row_index, epitope in enumerate(epitopes):
        epitope_data = diversity_all.loc[diversity_all["epitope"].eq(epitope)].copy()

        epitope_palette = _resolve_epitope_palette(
            epitope=epitope,
            group_order=group_order,
            palette=palette,
            epitope_colors=epitope_colors,
            non_enriched_color=non_enriched_color,
            enriched_color=enriched_color,
        )

        for column_index, (metric, ylabel) in enumerate(metric_specs):
            axis = axes[row_index, column_index]

            stat_match = stats_df.loc[
                stats_df["epitope"].eq(epitope)
                & stats_df["metric"].eq(metric)
            ]
            stat_row = stat_match.iloc[0] if not stat_match.empty else None

            metric_ylim = ylim_dict.get(metric) if ylim_dict is not None else None

            plot_metric_boxplot(
                ax=axis,
                data=epitope_data,
                metric=metric,
                ylabel=ylabel,
                epitope=epitope,
                stat_row=stat_row,
                group_col=group_col,
                group_order=group_order,
                palette=epitope_palette,
                whis=whis,
                boxplot_width=boxplot_width,
                box_linewidth=box_linewidth,
                median_color=median_color,
                median_linewidth=median_linewidth,
                point_color=point_color,
                point_size=point_size,
                point_alpha=point_alpha,
                point_jitter=point_jitter,
                xlim=xlim,
                ylim=metric_ylim,
                stat_label=stat_label,
                stat_bracket_y=stat_bracket_y,
                stat_text_y=stat_text_y,
                title_y=title_y,
                random_state=random_state,
            )

    if title is not None:
        fig.suptitle(title, fontsize=14, y=1.01)

    fig.tight_layout()

    if save_pdf:
        if pdf_path is None:
            pdf_path = (
                f"{enrich_col}_{cdr3_col}_{k}mer_shannon_"
                f"trim_{trim_start}_{trim_end}.pdf"
            )

        pdf_path = Path(pdf_path)
        if pdf_path.suffix.lower() != ".pdf":
            pdf_path = pdf_path.with_suffix(".pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            pdf_path,
            format="pdf",
            bbox_inches="tight",
        )

    if show:
        plt.show()

    summary = summarize_diversity(
        diversity_all=diversity_all,
        metrics=metric_names,
        enrich_col=enrich_col,
        group_col=group_col,
    )

    return {
        "fig": fig,
        "axes": axes,
        "div_all": diversity_all,
        "df_work_all": df_work_all,
        "summary": summary,
        "stats": stats_df,
        "pdf_path": str(pdf_path) if save_pdf and pdf_path is not None else None,
    }


__all__ = [
    "category_counts",
    "shannon_entropy_from_counts",
    "corrected_shannon_entropy_from_counts",
    "shannon_entropy_from_values",
    "corrected_shannon_entropy_from_values",
    "trim_sequence",
    "extract_kmers_from_sequence",
    "extract_kmers_from_sequences",
    "kmer_shannon_entropy",
    "kmer_corrected_shannon_entropy",
    "compute_kmer_and_cdr3_shannon_per_framework",
    "run_kmer_and_cdr3_shannon_for_epitopes",
    "mannwhitney_metric_by_epitope",
    "plot_metric_boxplot",
    "default_metric_specs",
    "summarize_diversity",
    "plot_kmer_and_cdr3_shannon_all_epitopes",
]