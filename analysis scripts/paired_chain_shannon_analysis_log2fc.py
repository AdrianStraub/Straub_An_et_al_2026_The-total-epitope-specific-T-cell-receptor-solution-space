from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from matplotlib.ticker import MultipleLocator


DEFAULT_GROUP_ORDER = ("Non-enriched", "Enriched")
DEFAULT_NON_ENRICHED_COLOR = "darkgray"


# ============================================================
# General utilities
# ============================================================


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


def get_axis_limit(
    axis_limits: Mapping[str, Mapping[str, Any]] | None,
    axis_key: str,
    limit_name: str,
):
    """Return an axis limit from a nested dictionary, or None."""
    if axis_limits is None:
        return None

    specification = axis_limits.get(axis_key)
    if specification is None:
        return None

    return specification.get(limit_name)


def apply_axis_limits(
    ax: plt.Axes,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def style_axis(ax: plt.Axes) -> None:
    """Remove the top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _build_epitope_colors(
    epitopes: Sequence[object],
    epitope_colors: Mapping[object, str] | None,
) -> dict[object, str]:
    """Use supplied epitope colors or fill missing entries from Matplotlib."""
    resolved = dict(epitope_colors or {})
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color",
        ["C0"],
    )

    for index, epitope in enumerate(epitopes):
        resolved.setdefault(
            epitope,
            default_colors[index % len(default_colors)],
        )

    return resolved


# ============================================================
# Diversity metrics
# ============================================================


def shannon_entropy_from_values(values) -> float:
    """
    Empirical Shannon entropy of a categorical distribution in nats.

    The supplied values are converted to category frequencies. Missing
    values are excluded.
    """
    series = pd.Series(values).dropna()
    n = len(series)

    if n == 0:
        return np.nan

    counts = series.value_counts().to_numpy(dtype=float)
    probabilities = counts / counts.sum()

    return float(
        -np.sum(probabilities * np.log(probabilities))
    )


def corrected_shannon_entropy_from_values(values) -> float:
    """
    Miller-Madow-corrected Shannon entropy in nats.

    H_MM = H_MLE + (K - 1) / (2n)

    K is the number of observed categories and n is the number of
    observations after removing missing values.
    """
    series = pd.Series(values).dropna()
    n = len(series)

    if n == 0:
        return np.nan

    counts = series.value_counts().to_numpy(dtype=float)
    probabilities = counts / counts.sum()
    n_observed_categories = len(counts)

    entropy_mle = -np.sum(
        probabilities * np.log(probabilities)
    )

    entropy_mm = (
        entropy_mle
        + (n_observed_categories - 1) / (2.0 * n)
    )

    return float(entropy_mm)



# ============================================================
# Analysis functions
# ============================================================


def compute_enrichment_for_epitope(
    df_tcr: pd.DataFrame,
    df_naive_tcr: pd.DataFrame,
    epitope,
    enrich_col: str = "trb_vj",
    specificity_col: str = "annotated_specificity",
    clone_size_col: str = "norm_TCR_expansion",
    pgen_col: str = "pgen_cdr3_ab",
    fc_threshold: float = 2,
    eps: float = 1e-10,
    naive_missing_count: float = 1,
) -> pd.DataFrame:
    """Calculate reactive-versus-naive enrichment for each gene combination."""
    _require_columns(
        df_tcr,
        [
            specificity_col,
            enrich_col,
            clone_size_col,
            pgen_col,
        ],
        "df_tcr",
    )
    _require_columns(
        df_naive_tcr,
        [enrich_col],
        "df_naive_tcr",
    )

    if fc_threshold <= 0:
        raise ValueError("fc_threshold must be larger than zero.")
    if eps < 0:
        raise ValueError("eps must be non-negative.")
    if naive_missing_count < 0:
        raise ValueError("naive_missing_count must be non-negative.")

    tcr_agg = (
        df_tcr.loc[
            df_tcr[specificity_col].eq(epitope)
        ]
        .copy()
        .assign(
            gene_recombination=lambda frame: frame[enrich_col]
        )
        .dropna(subset=["gene_recombination"])
        .groupby("gene_recombination", observed=True)
        .agg(
            react_count=("gene_recombination", "size"),
            mean_clone_size=(clone_size_col, "mean"),
            sum_clone_size=(clone_size_col, "sum"),
            max_clone_size=(clone_size_col, "max"),
            mean_pgen=(pgen_col, "mean"),
        )
        .reset_index()
    )

    if tcr_agg.empty:
        raise ValueError(
            f"No rows were found for epitope {epitope!r}."
        )

    naive_agg = (
        df_naive_tcr
        .copy()
        .assign(
            gene_recombination=lambda frame: frame[enrich_col]
        )
        .dropna(subset=["gene_recombination"])
        .groupby("gene_recombination", observed=True)
        .size()
        .reset_index(name="naive_count")
    )

    result = tcr_agg.merge(
        naive_agg,
        on="gene_recombination",
        how="left",
        validate="one_to_one",
    )

    result["naive_count"] = (
        result["naive_count"]
        .fillna(naive_missing_count)
        .astype(float)
    )

    react_total = result["react_count"].sum()
    naive_total = result["naive_count"].sum()

    if react_total <= 0 or naive_total <= 0:
        raise ValueError(
            "Reactive and naive totals must both be larger than zero."
        )

    result["f_react"] = result["react_count"] / react_total
    result["f_naive"] = result["naive_count"] / naive_total

    fold_change_ratio = (
        (result["f_react"] + eps)
        / (result["f_naive"] + eps)
    )

    # Retain natural-log enrichment for backwards compatibility.
    result["enrichment"] = np.log(fold_change_ratio)

    # Explicit log2 fold change for plotting and interpretation.
    result["log2fc"] = np.log2(fold_change_ratio)

    result["enrichment_group"] = np.where(
        result["log2fc"] > np.log2(fc_threshold),
        "Enriched",
        "Non-enriched",
    )

    result["epitope"] = epitope
    result["enrich_col"] = enrich_col

    return result


def compute_paired_chain_diversity_for_epitope(
    df_tcr: pd.DataFrame,
    df_enrich: pd.DataFrame,
    epitope,
    enrich_col: str = "trb_vj",
    paired_col: str = "tra_vj",
    paired_cdr3_len_col: str = "cdr3a_aa_len",
    specificity_col: str = "annotated_specificity",
) -> pd.DataFrame:
    """
    Calculate paired-chain Shannon entropy per enriched-chain VJ.

    Each row in the returned dataframe represents one value of enrich_col.
    """
    _require_columns(
        df_tcr,
        [
            specificity_col,
            enrich_col,
            paired_col,
            paired_cdr3_len_col,
        ],
        "df_tcr",
    )
    _require_columns(
        df_enrich,
        ["gene_recombination", "enrichment_group"],
        "df_enrich",
    )

    group_map = (
        df_enrich[["gene_recombination", "enrichment_group"]]
        .drop_duplicates()
        .rename(columns={"gene_recombination": enrich_col})
    )

    conflicting = (
        group_map.groupby(enrich_col, observed=True)[
            "enrichment_group"
        ]
        .nunique(dropna=False)
    )
    if (conflicting > 1).any():
        raise ValueError(
            "At least one gene combination maps to multiple "
            "enrichment groups."
        )

    group_map = group_map.drop_duplicates(subset=[enrich_col])

    df_pair = (
        df_tcr.loc[
            df_tcr[specificity_col].eq(epitope)
        ]
        .copy()
        .merge(
            group_map,
            on=enrich_col,
            how="inner",
            validate="many_to_one",
        )
        .dropna(
            subset=[
                enrich_col,
                paired_col,
                paired_cdr3_len_col,
                "enrichment_group",
            ]
        )
    )

    if df_pair.empty:
        return pd.DataFrame(
            columns=[
                enrich_col,
                "enrichment_group",
                "n_tcr",
                "paired_shannon_entropy",
                "paired_shannon_entropy_corrected",
                "paired_cdr3_len_shannon_entropy",
                "paired_cdr3_len_shannon_entropy_corrected",
                "epitope",
                "enrich_col",
                "paired_col",
                "paired_cdr3_len_col",
            ]
        )

    diversity = (
        df_pair
        .groupby(
            [enrich_col, "enrichment_group"],
            observed=True,
        )
        .agg(
            n_tcr=(paired_col, "size"),
            paired_shannon_entropy=(
                paired_col,
                shannon_entropy_from_values,
            ),
            paired_shannon_entropy_corrected=(
                paired_col,
                corrected_shannon_entropy_from_values,
            ),
            paired_cdr3_len_shannon_entropy=(
                paired_cdr3_len_col,
                shannon_entropy_from_values,
            ),
            paired_cdr3_len_shannon_entropy_corrected=(
                paired_cdr3_len_col,
                corrected_shannon_entropy_from_values,
            ),
        )
        .reset_index()
    )

    diversity["epitope"] = epitope
    diversity["enrich_col"] = enrich_col
    diversity["paired_col"] = paired_col
    diversity["paired_cdr3_len_col"] = paired_cdr3_len_col

    return diversity


# ============================================================
# Statistical analysis
# ============================================================


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


def _mannwhitney_two_groups(
    data: pd.DataFrame,
    value_col: str,
    group_col: str = "enrichment_group",
    group_order: Sequence[object] = DEFAULT_GROUP_ORDER,
) -> dict[str, object]:
    """Two-sided Mann-Whitney U test and rank-biserial effect size."""
    if len(group_order) != 2:
        raise ValueError(
            "group_order must contain exactly two groups."
        )

    _require_columns(
        data,
        [group_col, value_col],
        "data",
    )

    group_1, group_2 = group_order

    values_1 = (
        pd.to_numeric(
            data.loc[data[group_col].eq(group_1), value_col],
            errors="coerce",
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )

    values_2 = (
        pd.to_numeric(
            data.loc[data[group_col].eq(group_2), value_col],
            errors="coerce",
        )
        .replace([np.inf, -np.inf], np.nan)
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

        u_statistic = float(test.statistic)
        p_value = float(test.pvalue)

        # Positive values indicate larger values in group_1.
        rank_biserial = (
            2.0 * u_statistic
            / (len(values_1) * len(values_2))
            - 1.0
        )
    else:
        u_statistic = np.nan
        p_value = np.nan
        rank_biserial = np.nan

    return {
        "group_1": group_1,
        "group_2": group_2,
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
        # With the default order: Enriched - Non-enriched.
        "median_difference": (
            float(np.median(values_2) - np.median(values_1))
            if len(values_1) > 0 and len(values_2) > 0
            else np.nan
        ),
        "mannwhitney_u": u_statistic,
        "p_value": p_value,
        "rank_biserial_group_1_vs_group_2": rank_biserial,
    }


def compute_statistics_across_epitopes(
    df_enrich_all: pd.DataFrame,
    paired_div_all: pd.DataFrame,
    epitopes: Sequence[object],
    group_order: Sequence[object] = DEFAULT_GROUP_ORDER,
    p_adjust: str | None = "holm-sidak",
) -> pd.DataFrame:
    """
    Compare enrichment groups within each epitope for every plotted metric.
    """
    metric_specs = [
        ("enrich", "mean_clone_size", "enrichment"),
        ("enrich", "mean_pgen", "enrichment"),
        (
            "paired",
            "paired_shannon_entropy",
            "paired-chain diversity",
        ),
        (
            "paired",
            "paired_shannon_entropy_corrected",
            "paired-chain diversity",
        ),
        (
            "paired",
            "paired_cdr3_len_shannon_entropy",
            "paired-CDR3-length diversity",
        ),
        (
            "paired",
            "paired_cdr3_len_shannon_entropy_corrected",
            "paired-CDR3-length diversity",
        ),
    ]

    rows: list[dict[str, object]] = []

    for source, metric, analysis in metric_specs:
        source_df = (
            df_enrich_all
            if source == "enrich"
            else paired_div_all
        )

        for epitope in epitopes:
            subset = source_df.loc[
                source_df["epitope"].eq(epitope)
            ]

            result = _mannwhitney_two_groups(
                data=subset,
                value_col=metric,
                group_col="enrichment_group",
                group_order=group_order,
            )

            result.update(
                {
                    "epitope": epitope,
                    "metric": metric,
                    "analysis": analysis,
                }
            )
            rows.append(result)

    stats_df = pd.DataFrame(rows)
    stats_df["p_adjusted"] = np.nan

    for _, index in stats_df.groupby(
        "metric",
        sort=False,
    ).groups.items():
        metric_index = list(index)
        valid_index = [
            i
            for i in metric_index
            if pd.notna(stats_df.at[i, "p_value"])
        ]

        if not valid_index:
            continue

        raw_p = stats_df.loc[
            valid_index,
            "p_value",
        ].to_numpy(dtype=float)

        if p_adjust is None:
            adjusted_p = raw_p
        else:
            adjusted_p = multipletests(
                raw_p,
                method=p_adjust,
            )[1]

        stats_df.loc[
            valid_index,
            "p_adjusted",
        ] = adjusted_p

    stats_df["significance"] = (
        stats_df["p_adjusted"].map(_p_to_label)
    )

    return stats_df


def run_epitope_analysis_generalized(
    df_tcr: pd.DataFrame,
    df_naive_tcr: pd.DataFrame,
    epitopes: Sequence[object],
    enrich_col: str = "trb_vj",
    paired_col: str = "tra_vj",
    paired_cdr3_len_col: str = "cdr3a_aa_len",
    specificity_col: str = "annotated_specificity",
    clone_size_col: str = "norm_TCR_expansion",
    pgen_col: str = "pgen_cdr3_ab",
    fc_threshold: float = 2,
    eps: float = 1e-10,
    naive_missing_count: float = 1,
    group_order: Sequence[object] = DEFAULT_GROUP_ORDER,
    p_adjust: str | None = "holm-sidak",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run enrichment, paired diversity, and statistics for all epitopes."""
    if isinstance(epitopes, str):
        epitopes = (epitopes,)

    enrich_list: list[pd.DataFrame] = []
    diversity_list: list[pd.DataFrame] = []

    for epitope in epitopes:
        df_enrich = compute_enrichment_for_epitope(
            df_tcr=df_tcr,
            df_naive_tcr=df_naive_tcr,
            epitope=epitope,
            enrich_col=enrich_col,
            specificity_col=specificity_col,
            clone_size_col=clone_size_col,
            pgen_col=pgen_col,
            fc_threshold=fc_threshold,
            eps=eps,
            naive_missing_count=naive_missing_count,
        )

        df_diversity = compute_paired_chain_diversity_for_epitope(
            df_tcr=df_tcr,
            df_enrich=df_enrich,
            epitope=epitope,
            enrich_col=enrich_col,
            paired_col=paired_col,
            paired_cdr3_len_col=paired_cdr3_len_col,
            specificity_col=specificity_col,
        )

        enrich_list.append(df_enrich)
        diversity_list.append(df_diversity)

    df_enrich_all = pd.concat(
        enrich_list,
        ignore_index=True,
    )
    paired_div_all = pd.concat(
        diversity_list,
        ignore_index=True,
    )

    stats_all = compute_statistics_across_epitopes(
        df_enrich_all=df_enrich_all,
        paired_div_all=paired_div_all,
        epitopes=epitopes,
        group_order=group_order,
        p_adjust=p_adjust,
    )

    return df_enrich_all, paired_div_all, stats_all


def recompute_stats_after_relabeling(
    df_enrich_all: pd.DataFrame,
    paired_div_all: pd.DataFrame,
    epitopes: Sequence[object],
    group_order: Sequence[object] = DEFAULT_GROUP_ORDER,
    p_adjust: str | None = "holm",
) -> pd.DataFrame:
    """Recalculate all tests after enrichment labels were changed."""
    return compute_statistics_across_epitopes(
        df_enrich_all=df_enrich_all,
        paired_div_all=paired_div_all,
        epitopes=epitopes,
        group_order=group_order,
        p_adjust=p_adjust,
    )


# ============================================================
# Plotting functions
# ============================================================


def plot_enrichment_scatter(
    ax: plt.Axes,
    data: pd.DataFrame,
    epitope,
    y_col: str,
    ylabel: str,
    logfc_threshold: float,
    epitope_colors: Mapping[object, str],
    x_col: str = "enrichment",
    xlabel: str = "ln(freactive / fnaive)",
    non_enriched_color: str = DEFAULT_NON_ENRICHED_COLOR,
    group_order: Sequence[object] = DEFAULT_GROUP_ORDER,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    Plot the enrichment scatter using the original seaborn design.

    This deliberately preserves the supplied scatter-plot implementation.
    """
    epitope_color = epitope_colors[epitope]

    palette = {
        "Non-enriched": non_enriched_color,
        "Enriched": epitope_color,
    }

    sns.scatterplot(
        ax=ax,
        data=data,
        x=x_col,
        y=y_col,
        hue="enrichment_group",
        hue_order=group_order,
        palette=palette,
        s=20,
        alpha=0.75,
        linewidth=0,
        legend=False,
    )

    ax.axvline(
        logfc_threshold,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax.set_yscale("log")

    apply_axis_limits(ax, xlim=xlim, ylim=ylim)
    ax.xaxis.set_major_locator(MultipleLocator(1))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{epitope}: {ylabel}")

    style_axis(ax)
    
    ax.tick_params(
        axis="y",
        which="both",
        labelsize=14,
        labelleft=True,
    )
    
    ax.tick_params(
        axis="x",
        labelsize=14,
    )


def _axis_fraction_to_data(
    ax: plt.Axes,
    y_fraction: float,
) -> float:
    """Convert an axes-relative y coordinate to data coordinates."""
    display_xy = ax.transAxes.transform((0.0, y_fraction))
    return float(
        ax.transData.inverted().transform(display_xy)[1]
    )


def _add_vertical_headroom(
    ax: plt.Axes,
    fraction: float = 0.25,
) -> None:
    """Increase the upper y-limit while respecting linear/log scaling."""
    y_min, y_max = ax.get_ylim()

    if ax.get_yscale() == "log":
        if y_min <= 0 or y_max <= 0:
            return

        log_min = np.log10(y_min)
        log_max = np.log10(y_max)
        log_range = log_max - log_min
        if log_range <= 0:
            log_range = 1.0

        ax.set_ylim(
            10**log_min,
            10 ** (log_max + fraction * log_range),
        )
    else:
        y_range = y_max - y_min
        if y_range <= 0:
            y_range = max(abs(y_max), 1.0)

        ax.set_ylim(
            y_min,
            y_max + fraction * y_range,
        )


def _format_stat_annotation(
    stat_row: pd.Series | Mapping[str, object] | None,
    stat_label: str,
) -> str:
    if stat_row is None:
        return "NA"

    adjusted_p = stat_row.get("p_adjusted", np.nan)
    significance = stat_row.get("significance", "NA")

    if stat_label == "stars":
        return str(significance)

    if stat_label == "p":
        return (
            f"$p_{{adj}}$ = {adjusted_p:.3g}"
            if pd.notna(adjusted_p)
            else "NA"
        )

    if stat_label == "both":
        return (
            f"{significance}\n$p_{{adj}}$ = {adjusted_p:.3g}"
            if pd.notna(adjusted_p)
            else "NA"
        )

    raise ValueError(
        "stat_label must be 'stars', 'p', or 'both'."
    )


def plot_group_metric(
    ax: plt.Axes,
    data: pd.DataFrame,
    epitope,
    value_col: str,
    ylabel: str,
    stat_row: pd.Series | Mapping[str, object] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    log_scale: bool = False,
    group_order: Sequence[object] = DEFAULT_GROUP_ORDER,
    epitope_colors: Mapping[object, str] | None = None,
    non_enriched_color: str = DEFAULT_NON_ENRICHED_COLOR,
    whis: tuple[float, float] = (5, 95),
    boxplot_width: float = 0.55,
    box_linewidth: float = 2.0,
    median_color: str | None = None,
    median_linewidth: float = 2.0,
    point_color: str = "black",
    point_size: float = 24,
    point_alpha: float = 0.75,
    point_jitter: float = 0.045,
    stat_label: str = "stars",
    stat_bracket_y: float = 0.80,
    stat_text_y: float = 0.84,
    title_y: float = 1.06,
    rng: np.random.Generator | None = None,
) -> None:
    """
    Matplotlib boxplot matching the overall-flexibility-ratio design.

    The box shows the interquartile range, the central line is the median,
    the whiskers use the supplied percentiles, and every observation is
    displayed in the scatter layer. If median_color is None, both median
    lines use the epitope-specific color; an explicit color overrides it.
    """
    if len(group_order) != 2:
        raise ValueError(
            "group_order must contain exactly two groups."
        )

    if rng is None:
        rng = np.random.default_rng(1)

    plot_data = data.copy()
    plot_data[value_col] = pd.to_numeric(
        plot_data[value_col],
        errors="coerce",
    )

    plot_data = plot_data.loc[
        plot_data[value_col].notna()
        & np.isfinite(plot_data[value_col])
        & plot_data["enrichment_group"].isin(group_order)
    ].copy()

    if log_scale:
        plot_data = plot_data.loc[
            plot_data[value_col] > 0
        ].copy()

    box_values = [
        plot_data.loc[
            plot_data["enrichment_group"].eq(group),
            value_col,
        ].to_numpy(dtype=float)
        for group in group_order
    ]

    if epitope_colors is None:
        epitope_colors = {}

    default_plot_color = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color",
        ["C0"],
    )[0]
    epitope_color = epitope_colors.get(
        epitope,
        default_plot_color,
    )

    # By default, both median lines use the epitope-specific color.
    # Passing median_color explicitly overrides this behavior.
    resolved_median_color = (
        epitope_color
        if median_color is None
        else median_color
    )

    group_colors = {
        group_order[0]: non_enriched_color,
        group_order[1]: epitope_color,
    }

    valid_entries = [
        {
            "position": position,
            "values": values,
            "color": group_colors[group],
        }
        for position, (group, values) in enumerate(
            zip(group_order, box_values),
            start=1,
        )
        if len(values) > 0
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
        boxplot_result = ax.boxplot(
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
            boxplot_result["boxes"][index].set_edgecolor(color)

            for whisker in boxplot_result["whiskers"][
                2 * index : 2 * index + 2
            ]:
                whisker.set_color(color)
                whisker.set_linewidth(box_linewidth)

            for cap in boxplot_result["caps"][
                2 * index : 2 * index + 2
            ]:
                cap.set_color(color)
                cap.set_linewidth(box_linewidth)

            boxplot_result["medians"][index].set_color(
                resolved_median_color
            )
            boxplot_result["medians"][index].set_linewidth(
                median_linewidth
            )

    for position, values in enumerate(box_values, start=1):
        if len(values) == 0:
            continue

        jittered_x = rng.normal(
            loc=position,
            scale=point_jitter,
            size=len(values),
        )

        ax.scatter(
            jittered_x,
            values,
            s=point_size,
            color=point_color,
            alpha=point_alpha,
            linewidths=0,
            zorder=3,
        )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        group_order,
        rotation=30,
        ha="right",
    )

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0.5, 2.5)

    if ylim is not None:
        if ylim[1] <= ylim[0]:
            raise ValueError(
                "The upper y-axis limit must exceed the lower limit."
            )
        ax.set_ylim(ylim)
    else:
        ax.relim()
        ax.autoscale_view()
        _add_vertical_headroom(ax, fraction=0.25)

    if all(len(values) > 0 for values in box_values):
        bracket_bottom = _axis_fraction_to_data(
            ax,
            stat_bracket_y,
        )
        bracket_top = _axis_fraction_to_data(
            ax,
            stat_bracket_y + 0.02,
        )
        text_y = _axis_fraction_to_data(
            ax,
            stat_text_y,
        )

        ax.plot(
            [1, 1, 2, 2],
            [
                bracket_bottom,
                bracket_top,
                bracket_top,
                bracket_bottom,
            ],
            linewidth=1,
            clip_on=False,
        )

        ax.text(
            1.5,
            text_y,
            _format_stat_annotation(
                stat_row=stat_row,
                stat_label=stat_label,
            ),
            ha="center",
            va="bottom",
        )

    ax.set_title(
        f"{epitope}: {ylabel}",
        fontsize=7,
        y=title_y,
        pad=4,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)

    style_axis(ax)

    # Explicitly retain y tick labels on every subplot.
    ax.tick_params(
        axis="y",
        which="both",
        labelsize = 14,
        labelleft=True,
    )


def plot_all_epitopes_generalized(
    df_enrich_all: pd.DataFrame,
    paired_div_all: pd.DataFrame,
    stats_all: pd.DataFrame,
    epitopes: Sequence[object],
    enrich_col: str = "trb_vj",
    paired_col: str = "tra_vj",
    paired_cdr3_len_col: str = "cdr3a_aa_len",
    fc_threshold: float = 2,
    scatter_width: float = 3.4,
    box_width: float = 2.4,
    subplot_height: float = 3.5,
    axis_limits: Mapping[str, Mapping[str, Any]] | None = None,
    epitope_colors: Mapping[object, str] | None = None,
    non_enriched_color: str = DEFAULT_NON_ENRICHED_COLOR,
    group_order: Sequence[object] = DEFAULT_GROUP_ORDER,
    whis: tuple[float, float] = (5, 95),
    boxplot_width: float = 0.55,
    box_linewidth: float = 2.0,
    median_color: str | None = None,
    median_linewidth: float = 2.0,
    point_color: str = "black",
    point_size: float = 24,
    point_alpha: float = 0.75,
    point_jitter: float = 0.045,
    stat_label: str = "stars",
    stat_bracket_y: float = 0.80,
    stat_text_y: float = 0.84,
    title_y: float = 1.06,
    random_state: int = 1,
    save_pdf: bool = True,
    pdf_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot all enrichment and paired-diversity results in one figure."""
    if isinstance(epitopes, str):
        epitopes = (epitopes,)

    if len(epitopes) == 0:
        raise ValueError("At least one epitope is required.")

    resolved_colors = _build_epitope_colors(
        epitopes=epitopes,
        epitope_colors=epitope_colors,
    )

    if pdf_path is None:
        pdf_path = (
            f"{enrich_col}_enrichment_"
            f"{paired_col}_shannon.pdf"
        )

    ln_fc_threshold = np.log(fc_threshold)
    log2fc_threshold = np.log2(fc_threshold)
    paired_label = paired_col.upper()
    cdr3_len_label = paired_cdr3_len_col.upper()

    plot_specs = [
        {
            "kind": "scatter",
            "data": "enrich",
            "value_col": "mean_clone_size",
            "ylabel": "Mean clone size",
            "axis_key": "mean_clone_size_scatter",
            "log_scale": True,
            "x_col": "log2fc",
            "xlabel": "log2(freactive / fnaive)",
            "x_threshold": log2fc_threshold,
        },
        {
            "kind": "scatter",
            "data": "enrich",
            "value_col": "mean_pgen",
            "ylabel": "Mean pgen",
            "axis_key": "mean_pgen_scatter",
            "log_scale": True,
            "x_col": "log2fc",
            "xlabel": "log2(freactive / fnaive)",
            "x_threshold": log2fc_threshold,
        },
        {
            "kind": "box",
            "data": "enrich",
            "value_col": "mean_clone_size",
            "ylabel": "Mean clone size",
            "axis_key": "mean_clone_size_box",
            "log_scale": True,
        },
        {
            "kind": "box",
            "data": "enrich",
            "value_col": "mean_pgen",
            "ylabel": "Mean pgen",
            "axis_key": "mean_pgen_box",
            "log_scale": True,
        },
        {
            "kind": "box",
            "data": "paired",
            "value_col": "paired_shannon_entropy",
            "ylabel": f"{paired_label} Shannon entropy",
            "axis_key": "paired_shannon_box",
            "log_scale": False,
        },
        {
            "kind": "box",
            "data": "paired",
            "value_col": "paired_shannon_entropy_corrected",
            "ylabel": f"{paired_label} corrected Shannon entropy",
            "axis_key": "paired_shannon_corrected_box",
            "log_scale": False,
        },
        {
            "kind": "box",
            "data": "paired",
            "value_col": "paired_cdr3_len_shannon_entropy",
            "ylabel": f"{cdr3_len_label} Shannon entropy",
            "axis_key": "paired_cdr3_len_shannon_box",
            "log_scale": False,
        },
        {
            "kind": "box",
            "data": "paired",
            "value_col": "paired_cdr3_len_shannon_entropy_corrected",
            "ylabel": (
                f"{cdr3_len_label} corrected Shannon entropy"
            ),
            "axis_key": "paired_cdr3_len_shannon_corrected_box",
            "log_scale": False,
        },
    ]

    width_ratios = [
        scatter_width if spec["kind"] == "scatter" else box_width
        for spec in plot_specs
    ]

    n_rows = len(epitopes)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=len(plot_specs),
        figsize=(
            sum(width_ratios),
            subplot_height * n_rows,
        ),
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
    )

    rng = np.random.default_rng(random_state)

    for row, epitope in enumerate(epitopes):
        enrich = df_enrich_all.loc[
            df_enrich_all["epitope"].eq(epitope)
        ].copy()
        paired = paired_div_all.loc[
            paired_div_all["epitope"].eq(epitope)
        ].copy()

        for col, spec in enumerate(plot_specs):
            axis = axes[row, col]
            data = enrich if spec["data"] == "enrich" else paired

            if spec["kind"] == "scatter":
                plot_enrichment_scatter(
                    ax=axis,
                    data=data,
                    epitope=epitope,
                    y_col=spec["value_col"],
                    ylabel=spec["ylabel"],
                    logfc_threshold=spec["x_threshold"],
                    epitope_colors=resolved_colors,
                    x_col=spec["x_col"],
                    xlabel=spec["xlabel"],
                    non_enriched_color=non_enriched_color,
                    group_order=group_order,
                    xlim=get_axis_limit(
                        axis_limits,
                        spec["axis_key"],
                        "xlim",
                    ),
                    ylim=get_axis_limit(
                        axis_limits,
                        spec["axis_key"],
                        "ylim",
                    ),
                )
            else:
                stat_match = stats_all.loc[
                    stats_all["epitope"].eq(epitope)
                    & stats_all["metric"].eq(spec["value_col"])
                ]
                stat_row = (
                    stat_match.iloc[0]
                    if not stat_match.empty
                    else None
                )

                plot_group_metric(
                    ax=axis,
                    data=data,
                    epitope=epitope,
                    value_col=spec["value_col"],
                    ylabel=spec["ylabel"],
                    stat_row=stat_row,
                    xlim=get_axis_limit(
                        axis_limits,
                        spec["axis_key"],
                        "xlim",
                    ),
                    ylim=get_axis_limit(
                        axis_limits,
                        spec["axis_key"],
                        "ylim",
                    ),
                    log_scale=spec["log_scale"],
                    group_order=group_order,
                    epitope_colors=resolved_colors,
                    non_enriched_color=non_enriched_color,
                    whis=whis,
                    boxplot_width=boxplot_width,
                    box_linewidth=box_linewidth,
                    median_color=median_color,
                    median_linewidth=median_linewidth,
                    point_color=point_color,
                    point_size=point_size,
                    point_alpha=point_alpha,
                    point_jitter=point_jitter,
                    stat_label=stat_label,
                    stat_bracket_y=stat_bracket_y,
                    stat_text_y=stat_text_y,
                    title_y=title_y,
                    rng=rng,
                )

    fig.tight_layout()

    if save_pdf:
        output_path = Path(pdf_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            output_path,
            format="pdf",
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return fig


__all__ = [
    "DEFAULT_GROUP_ORDER",
    "DEFAULT_NON_ENRICHED_COLOR",
    "apply_axis_limits",
    "compute_enrichment_for_epitope",
    "compute_paired_chain_diversity_for_epitope",
    "compute_statistics_across_epitopes",
    "corrected_shannon_entropy_from_values",
    "get_axis_limit",
    "plot_all_epitopes_generalized",
    "plot_enrichment_scatter",
    "plot_group_metric",
    "recompute_stats_after_relabeling",
    "run_epitope_analysis_generalized",
    "shannon_entropy_from_values",
    "style_axis",
]