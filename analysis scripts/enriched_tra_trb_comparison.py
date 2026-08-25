from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


DEFAULT_COMPARISON_ORDER = (
    "Enriched TRA-VJ",
    "Enriched TRB-VJ",
)

DEFAULT_METRICS = (
    ("paired_shannon_entropy", "Paired-chain Shannon entropy"),
    (
        "paired_shannon_entropy_corrected",
        "Paired-chain corrected Shannon entropy",
    ),
    (
        "paired_cdr3_len_shannon_entropy_corrected",
        "Paired CDR3-length corrected Shannon entropy",
    ),
)


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


def style_axis(ax: plt.Axes) -> None:
    """Remove top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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


def _normalize_metrics(
    metrics: Sequence[str | tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    """Accept metric names or (metric, ylabel) pairs."""
    if metrics is None:
        return list(DEFAULT_METRICS)

    normalized: list[tuple[str, str]] = []

    for item in metrics:
        if isinstance(item, str):
            normalized.append((item, item))
            continue

        if len(item) != 2:
            raise ValueError(
                "Each metric entry must be a metric string or a "
                "(metric, ylabel) pair."
            )

        metric, ylabel = item
        normalized.append((str(metric), str(ylabel)))

    if not normalized:
        raise ValueError("At least one metric is required.")

    return normalized


def build_enriched_framework_comparison(
    paired_div_tra: pd.DataFrame,
    paired_div_trb: pd.DataFrame,
    epitope_col: str = "epitope",
    enrichment_group_col: str = "enrichment_group",
    enriched_label: str = "Enriched",
    comparison_col: str = "enriched_framework_type",
    tra_label: str = "Enriched TRA-VJ",
    trb_label: str = "Enriched TRB-VJ",
) -> pd.DataFrame:
    """Combine enriched TRA-VJ and enriched TRB-VJ diversity rows."""
    _require_columns(
        paired_div_tra,
        [epitope_col, enrichment_group_col],
        "paired_div_tra",
    )
    _require_columns(
        paired_div_trb,
        [epitope_col, enrichment_group_col],
        "paired_div_trb",
    )

    tra = (
        paired_div_tra.loc[
            paired_div_tra[enrichment_group_col].eq(enriched_label)
        ]
        .copy()
        .assign(**{comparison_col: tra_label})
    )

    trb = (
        paired_div_trb.loc[
            paired_div_trb[enrichment_group_col].eq(enriched_label)
        ]
        .copy()
        .assign(**{comparison_col: trb_label})
    )

    return pd.concat([tra, trb], ignore_index=True, sort=False)


def mannwhitney_two_groups_tra_trb(
    data: pd.DataFrame,
    value_col: str,
    group_col: str = "enriched_framework_type",
    group_order: Sequence[object] = DEFAULT_COMPARISON_ORDER,
) -> dict[str, object]:
    """Two-sided Mann-Whitney U test with rank-biserial effect size."""
    if len(group_order) != 2:
        raise ValueError("group_order must contain exactly two groups.")

    _require_columns(data, [group_col, value_col], "data")

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

        # Positive: values tend to be higher in group_1.
        rank_biserial = (
            2.0 * u_statistic
            / (len(values_1) * len(values_2))
            - 1.0
        )
    else:
        u_statistic = np.nan
        p_value = np.nan
        rank_biserial = np.nan

    median_1 = (
        float(np.median(values_1))
        if len(values_1) > 0
        else np.nan
    )
    median_2 = (
        float(np.median(values_2))
        if len(values_2) > 0
        else np.nan
    )

    return {
        "group_1": group_1,
        "group_2": group_2,
        "n_group_1": len(values_1),
        "n_group_2": len(values_2),
        "median_group_1": median_1,
        "median_group_2": median_2,
        # With the default order: Enriched TRB-VJ - Enriched TRA-VJ.
        "median_difference": (
            median_2 - median_1
            if pd.notna(median_1) and pd.notna(median_2)
            else np.nan
        ),
        "mannwhitney_u": u_statistic,
        "p_value": p_value,
        "rank_biserial_group_1_vs_group_2": rank_biserial,
    }


def compute_tra_trb_comparison_statistics(
    compare_df: pd.DataFrame,
    epitopes: Sequence[object],
    metrics: Sequence[str | tuple[str, str]] | None = None,
    epitope_col: str = "epitope",
    group_col: str = "enriched_framework_type",
    group_order: Sequence[object] = DEFAULT_COMPARISON_ORDER,
    p_adjust: str | None = "holm-sidak",
) -> pd.DataFrame:
    """
    Test TRA-VJ versus TRB-VJ enriched groups within each epitope.

    P-values are adjusted across epitopes separately for each metric,
    matching the previous paired-diversity/flexibility-ratio procedure.
    """
    metric_specs = _normalize_metrics(metrics)
    metric_names = [metric for metric, _ in metric_specs]

    _require_columns(
        compare_df,
        [epitope_col, group_col, *metric_names],
        "compare_df",
    )

    rows: list[dict[str, object]] = []

    for metric, ylabel in metric_specs:
        for epitope in epitopes:
            subset = compare_df.loc[
                compare_df[epitope_col].eq(epitope)
            ]

            result = mannwhitney_two_groups_tra_trb(
                data=subset,
                value_col=metric,
                group_col=group_col,
                group_order=group_order,
            )
            result.update(
                {
                    "epitope": epitope,
                    "metric": metric,
                    "ylabel": ylabel,
                    "comparison": (
                        f"{group_order[0]} vs {group_order[1]}"
                    ),
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


def _get_axis_setting(
    axis_settings: Mapping[str, Mapping[str, Any]] | None,
    metric: str,
    setting: str,
    default=None,
):
    if axis_settings is None:
        return default

    metric_settings = axis_settings.get(metric)
    if metric_settings is None:
        return default

    return metric_settings.get(setting, default)


def plot_tra_trb_comparison_metric(
    ax: plt.Axes,
    data: pd.DataFrame,
    epitope,
    value_col: str,
    ylabel: str,
    stat_row: pd.Series | Mapping[str, object] | None = None,
    epitope_colors: Mapping[object, str] | None = None,
    group_col: str = "enriched_framework_type",
    group_order: Sequence[object] = DEFAULT_COMPARISON_ORDER,
    first_group_color: str = "darkgray",
    xlim: tuple[float, float] | None = (0.5, 2.5),
    ylim: tuple[float, float] | None = None,
    yscale: str = "linear",
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
    stat_linewidth: float = 1.0,
    title_y: float = 1.06,
    title_fontsize: float = 7,
    ylabel_fontsize: float | None = None,
    xtick_rotation: float = 30,
    rng: np.random.Generator | None = None,
) -> None:
    """Plot one TRA-VJ versus TRB-VJ comparison panel."""
    if len(group_order) != 2:
        raise ValueError("group_order must contain exactly two groups.")

    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'.")

    if rng is None:
        rng = np.random.default_rng(1)

    _require_columns(data, [group_col, value_col], "data")

    plot_data = data.copy()
    plot_data[value_col] = pd.to_numeric(
        plot_data[value_col],
        errors="coerce",
    )
    plot_data = plot_data.loc[
        plot_data[value_col].notna()
        & np.isfinite(plot_data[value_col])
        & plot_data[group_col].isin(group_order)
    ].copy()

    if yscale == "log":
        plot_data = plot_data.loc[
            plot_data[value_col] > 0
        ].copy()

    box_values = [
        plot_data.loc[
            plot_data[group_col].eq(group),
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

    resolved_median_color = (
        epitope_color
        if median_color is None
        else median_color
    )

    group_colors = {
        group_order[0]: first_group_color,
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

    # Set scale before explicit limits and before annotation conversion.
    ax.set_yscale(yscale)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        group_order,
        rotation=xtick_rotation,
        ha="right",
        rotation_mode="anchor",
    )

    if xlim is not None:
        if xlim[1] <= xlim[0]:
            raise ValueError(
                "The upper x-axis limit must exceed the lower limit."
            )
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0.5, 2.5)

    if ylim is not None:
        if ylim[1] <= ylim[0]:
            raise ValueError(
                "The upper y-axis limit must exceed the lower limit."
            )
        if yscale == "log" and ylim[0] <= 0:
            raise ValueError(
                "The lower y-axis limit must be positive for log scale."
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
            color="black",
            linewidth=stat_linewidth,
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
        fontsize=title_fontsize,
        y=title_y,
        pad=4,
    )
    ax.set_xlabel("")
    ax.set_ylabel(
        ylabel,
        fontsize=ylabel_fontsize,
    )

    style_axis(ax)
    ax.tick_params(
        axis="y",
        which="both",
        labelsize = 14,
        labelleft=True,
    )


def plot_enriched_tra_trb_comparison(
    compare_df: pd.DataFrame,
    epitopes: Sequence[object],
    metrics: Sequence[str | tuple[str, str]] | None = None,
    axis_settings: Mapping[str, Mapping[str, Any]] | None = None,
    epitope_colors: Mapping[object, str] | None = None,
    epitope_col: str = "epitope",
    group_col: str = "enriched_framework_type",
    group_order: Sequence[object] = DEFAULT_COMPARISON_ORDER,
    first_group_color: str = "darkgray",
    p_adjust: str | None = "holm-sidak",
    box_width: float = 2.0,
    subplot_height: float = 3.5,
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
    stat_linewidth: float = 1.0,
    title_y: float = 1.06,
    title_fontsize: float = 7,
    ylabel_fontsize: float | None = None,
    xtick_rotation: float = 30,
    random_state: int = 1,
    figure_title: str | None = None,
    save_pdf: bool = False,
    pdf_path: str | Path | None = None,
    show: bool = True,
) -> dict[str, object]:
    """Calculate statistics and plot all epitope/metric comparisons."""
    if isinstance(epitopes, str):
        epitopes = (epitopes,)

    if len(epitopes) == 0:
        raise ValueError("At least one epitope is required.")

    metric_specs = _normalize_metrics(metrics)
    metric_names = [metric for metric, _ in metric_specs]

    _require_columns(
        compare_df,
        [epitope_col, group_col, *metric_names],
        "compare_df",
    )

    if epitope_colors is None:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key().get(
            "color",
            ["C0"],
        )
        epitope_colors = {
            epitope: default_colors[index % len(default_colors)]
            for index, epitope in enumerate(epitopes)
        }
    else:
        epitope_colors = dict(epitope_colors)
        default_colors = plt.rcParams["axes.prop_cycle"].by_key().get(
            "color",
            ["C0"],
        )
        for index, epitope in enumerate(epitopes):
            epitope_colors.setdefault(
                epitope,
                default_colors[index % len(default_colors)],
            )

    stats_df = compute_tra_trb_comparison_statistics(
        compare_df=compare_df,
        epitopes=epitopes,
        metrics=metric_specs,
        epitope_col=epitope_col,
        group_col=group_col,
        group_order=group_order,
        p_adjust=p_adjust,
    )

    n_rows = len(epitopes)
    n_cols = len(metric_specs)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(
            box_width * n_cols,
            subplot_height * n_rows,
        ),
        squeeze=False,
    )

    rng = np.random.default_rng(random_state)

    for row, epitope in enumerate(epitopes):
        epitope_data = compare_df.loc[
            compare_df[epitope_col].eq(epitope)
        ].copy()

        for col, (metric, ylabel) in enumerate(metric_specs):
            axis = axes[row, col]

            stat_match = stats_df.loc[
                stats_df["epitope"].eq(epitope)
                & stats_df["metric"].eq(metric)
            ]
            stat_row = (
                stat_match.iloc[0]
                if not stat_match.empty
                else None
            )

            xlim = _get_axis_setting(
                axis_settings,
                metric,
                "xlim",
                (0.5, 2.5),
            )
            ylim = _get_axis_setting(
                axis_settings,
                metric,
                "ylim",
                None,
            )
            yscale = _get_axis_setting(
                axis_settings,
                metric,
                "yscale",
                "linear",
            )

            plot_tra_trb_comparison_metric(
                ax=axis,
                data=epitope_data,
                epitope=epitope,
                value_col=metric,
                ylabel=ylabel,
                stat_row=stat_row,
                epitope_colors=epitope_colors,
                group_col=group_col,
                group_order=group_order,
                first_group_color=first_group_color,
                xlim=xlim,
                ylim=ylim,
                yscale=yscale,
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
                stat_linewidth=stat_linewidth,
                title_y=title_y,
                title_fontsize=title_fontsize,
                ylabel_fontsize=ylabel_fontsize,
                xtick_rotation=xtick_rotation,
                rng=rng,
            )

    if figure_title is not None:
        fig.suptitle(
            figure_title,
            y=1.01,
        )

    fig.tight_layout()

    if save_pdf:
        if pdf_path is None:
            pdf_path = "enriched_TRA_VJ_vs_TRB_VJ_diversity.pdf"

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

    return {
        "fig": fig,
        "axes": axes,
        "stats": stats_df,
        "compare_df": compare_df,
    }


__all__ = [
    "DEFAULT_COMPARISON_ORDER",
    "DEFAULT_METRICS",
    "build_enriched_framework_comparison",
    "compute_tra_trb_comparison_statistics",
    "mannwhitney_two_groups_tra_trb",
    "plot_enriched_tra_trb_comparison",
    "plot_tra_trb_comparison_metric",
    "style_axis",
]
