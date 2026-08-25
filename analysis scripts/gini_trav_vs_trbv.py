from collections.abc import Sequence

import numpy as np
import pandas as pd


def gini_from_values(values) -> float:
    """
    Calculate the Gini coefficient of category-frequency counts.

    For gene calls, the input values are converted into gene frequencies.
    Gini = 0 indicates equal usage of all observed genes.
    Larger values indicate increasing dominance by one or a few genes.

    Missing values are excluded.
    """
    series = pd.Series(values).dropna()

    if series.empty:
        return np.nan

    counts = (
        series.value_counts()
        .to_numpy(dtype=float)
    )

    if len(counts) == 0 or counts.sum() <= 0:
        return np.nan

    counts = np.sort(counts)
    n_categories = len(counts)

    category_index = np.arange(
        1,
        n_categories + 1,
        dtype=float,
    )

    gini = (
        2.0
        * np.sum(category_index * counts)
        / (n_categories * counts.sum())
        - (n_categories + 1.0) / n_categories
    )

    return float(np.clip(gini, 0.0, 1.0))


def _calculate_gini_per_group(
    dataframe: pd.DataFrame,
    group_cols: Sequence[str],
    gene_col: str,
    chain: str,
    min_observations: int = 1,
) -> pd.DataFrame:
    """
    Calculate gene-usage Gini for each combination of group columns.
    """
    required_cols = list(group_cols) + [gene_col]

    missing_cols = [
        col
        for col in required_cols
        if col not in dataframe.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing columns in dataframe: {missing_cols}"
        )

    data = (
        dataframe[required_cols]
        .dropna(subset=required_cols)
        .copy()
    )

    result = (
        data
        .groupby(
            list(group_cols),
            observed=True,
        )
        .agg(
            n_observations=(gene_col, "size"),
            n_unique_genes=(gene_col, "nunique"),
            gini=(gene_col, gini_from_values),
        )
        .reset_index()
    )

    result.loc[
        result["n_observations"] < min_observations,
        "gini",
    ] = np.nan

    result["chain"] = chain
    result["gene_col"] = gene_col

    return result


def calculate_epitope_baseline_gini_delta(
    df_baseline: pd.DataFrame,
    df_epitope: pd.DataFrame,
    epitopes,
    donor_col: str,
    epitope_col: str,
    trav_col: str,
    trbv_col: str,
    min_observations: int = 1,
):
    if isinstance(epitopes, str):
        epitopes = [epitopes]
    else:
        epitopes = list(epitopes)

    chain_columns = {
        "TRAV": trav_col,
        "TRBV": trbv_col,
    }


# Naive baseline


    baseline_results = []

    for chain, gene_col in chain_columns.items():
        baseline_chain = _calculate_gini_per_group(
            dataframe=df_baseline,
            group_cols=[donor_col],
            gene_col=gene_col,
            chain=chain,
            min_observations=min_observations,
        )

        baseline_results.append(baseline_chain)

    baseline_per_donor = pd.concat(
        baseline_results,
        ignore_index=True,
    )

    baseline_summary = (
        baseline_per_donor
        .groupby("chain", observed=True)
        .agg(
            baseline_mean_gini=("gini", "mean"),
            baseline_median_gini=("gini", "median"),
            baseline_sd_gini=("gini", "std"),
            n_baseline_donors=("gini", "count"),
        )
        .reset_index()
    )


# Epitope repertoire: calculate separately per epitope and donor


    epitope_data = (
        df_epitope.loc[
            df_epitope[epitope_col].isin(epitopes)
        ]
        .copy()
    )

    epitope_results = []

    for chain, gene_col in chain_columns.items():
        epitope_chain = _calculate_gini_per_group(
            dataframe=epitope_data,
            group_cols=[
                epitope_col,
                donor_col,
            ],
            gene_col=gene_col,
            chain=chain,
            min_observations=min_observations,
        )

        epitope_results.append(epitope_chain)

    epitope_per_donor = pd.concat(
        epitope_results,
        ignore_index=True,
    )


# Compare each epitope-donor value with the average baseline


    result = epitope_per_donor.merge(
        baseline_summary,
        on="chain",
        how="left",
        validate="many_to_one",
    )

    result["delta_gini"] = (
        result["gini"]
        - result["baseline_mean_gini"]
    )

    result = result.rename(
        columns={
            "gini": "epitope_gini",
            "n_observations": "n_epitope_observations",
            "n_unique_genes": "n_unique_epitope_genes",
        }
    )

    baseline_per_donor = baseline_per_donor.rename(
        columns={
            "gini": "baseline_gini",
            "n_observations": "n_baseline_observations",
            "n_unique_genes": "n_unique_baseline_genes",
        }
    )

    return (
        result,
        baseline_per_donor,
        baseline_summary,
    )
	
	
# plotting functions	


from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
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


def _format_stat_annotation(
    stat_row: pd.Series | None,
    stat_label: str,
) -> str:
    if stat_row is None:
        return "NA"

    p_adjusted = stat_row.get("p_adjusted", np.nan)
    significance = stat_row.get("significance", "NA")

    if stat_label == "stars":
        return str(significance)

    if stat_label == "p":
        return (
            f"$p_{{adj}}$ = {p_adjusted:.3g}"
            if pd.notna(p_adjusted)
            else "NA"
        )

    if stat_label == "both":
        return (
            f"{significance}\n$p_{{adj}}$ = {p_adjusted:.3g}"
            if pd.notna(p_adjusted)
            else "NA"
        )

    raise ValueError(
        "stat_label must be 'stars', 'p', or 'both'."
    )

def prepare_paired_delta_gini_data(
    data: pd.DataFrame,
    epitopes: Sequence[object],
    donor_col: str,
    epitope_col: str,
    chain_col: str = "chain",
    value_col: str = "delta_gini",
    chain_order: Sequence[str] = ("TRAV", "TRBV"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retain only donors with both TRAV and TRBV values within an epitope.

    Returns
    -------
    paired_wide:
        One row per epitope and donor, with separate TRAV and TRBV columns.

    paired_long:
        Long-format version used for plotting.
    """
    if isinstance(epitopes, str):
        epitopes = [epitopes]
    else:
        epitopes = list(epitopes)

    if len(chain_order) != 2:
        raise ValueError(
            "chain_order must contain exactly two chain labels."
        )

    required_columns = {
        donor_col,
        epitope_col,
        chain_col,
        value_col,
    }

    missing_columns = sorted(
        required_columns.difference(data.columns)
    )

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    plot_data = data.loc[
        data[epitope_col].isin(epitopes)
        & data[chain_col].isin(chain_order),
        [
            epitope_col,
            donor_col,
            chain_col,
            value_col,
        ],
    ].copy()

    plot_data[value_col] = pd.to_numeric(
        plot_data[value_col],
        errors="coerce",
    )

    plot_data = plot_data.loc[
        plot_data[value_col].notna()
        & np.isfinite(plot_data[value_col])
    ].copy()

    duplicate_counts = (
        plot_data
        .groupby(
            [
                epitope_col,
                donor_col,
                chain_col,
            ],
            observed=True,
        )
        .size()
    )

    duplicated = duplicate_counts.loc[
        duplicate_counts > 1
    ]

    if not duplicated.empty:
        raise ValueError(
            "More than one delta-Gini value was found for at least "
            "one epitope × donor × chain combination. Each combination "
            "must have exactly one value."
        )

    paired_wide = (
        plot_data
        .pivot(
            index=[
                epitope_col,
                donor_col,
            ],
            columns=chain_col,
            values=value_col,
        )
        .reset_index()
    )

    for chain in chain_order:
        if chain not in paired_wide.columns:
            paired_wide[chain] = np.nan

    paired_wide = paired_wide.dropna(
        subset=list(chain_order)
    ).reset_index(drop=True)

    paired_long = paired_wide.melt(
        id_vars=[
            epitope_col,
            donor_col,
        ],
        value_vars=list(chain_order),
        var_name=chain_col,
        value_name=value_col,
    )

    return paired_wide, paired_long

def compute_delta_gini_chain_statistics(
    paired_wide: pd.DataFrame,
    epitopes: Sequence[object],
    epitope_col: str,
    chain_order: Sequence[str] = ("TRAV", "TRBV"),
    p_adjust: str | None = "holm-sidak",
    min_pairs: int = 2,
) -> pd.DataFrame:
    """
    Compare donor-matched TRAV and TRBV delta Gini using a two-sided
    Wilcoxon signed-rank test within each epitope.

    The paired difference is defined as:

        group_2 - group_1 = TRBV - TRAV

    P-values are adjusted across epitopes.
    """
    group_1, group_2 = chain_order

    rows = []

    for epitope in epitopes:
        epitope_data = paired_wide.loc[
            paired_wide[epitope_col].eq(epitope)
        ].copy()

        values_1 = epitope_data[group_1].to_numpy(dtype=float)
        values_2 = epitope_data[group_2].to_numpy(dtype=float)

        finite = (
            np.isfinite(values_1)
            & np.isfinite(values_2)
        )

        values_1 = values_1[finite]
        values_2 = values_2[finite]

        difference = values_2 - values_1
        n_pairs = len(difference)

        if n_pairs < min_pairs:
            statistic = np.nan
            p_value = np.nan
            rank_biserial = np.nan

        elif np.allclose(difference, 0):
            statistic = 0.0
            p_value = 1.0
            rank_biserial = 0.0

        else:
            test = wilcoxon(
                values_1,
                values_2,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

            statistic = float(test.statistic)
            p_value = float(test.pvalue)

            nonzero_difference = difference[
                ~np.isclose(difference, 0)
            ]

            absolute_ranks = rankdata(
                np.abs(nonzero_difference)
            )

            positive_rank_sum = absolute_ranks[
                nonzero_difference > 0
            ].sum()

            negative_rank_sum = absolute_ranks[
                nonzero_difference < 0
            ].sum()

            rank_sum = (
                positive_rank_sum
                + negative_rank_sum
            )

            rank_biserial = (
                (positive_rank_sum - negative_rank_sum)
                / rank_sum
                if rank_sum > 0
                else 0.0
            )

        rows.append(
            {
                "epitope": epitope,
                "group_1": group_1,
                "group_2": group_2,
                "n_pairs": n_pairs,
                "median_group_1": (
                    float(np.median(values_1))
                    if n_pairs > 0
                    else np.nan
                ),
                "median_group_2": (
                    float(np.median(values_2))
                    if n_pairs > 0
                    else np.nan
                ),
                "median_paired_difference": (
                    float(np.median(difference))
                    if n_pairs > 0
                    else np.nan
                ),
                "mean_paired_difference": (
                    float(np.mean(difference))
                    if n_pairs > 0
                    else np.nan
                ),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "matched_rank_biserial": rank_biserial,
            }
        )

    stats_df = pd.DataFrame(rows)
    stats_df["p_adjusted"] = np.nan

    valid = stats_df["p_value"].notna()

    if valid.any():
        raw_p_values = stats_df.loc[
            valid,
            "p_value",
        ].to_numpy(dtype=float)

        if p_adjust is None:
            adjusted_p_values = raw_p_values
        else:
            adjusted_p_values = multipletests(
                raw_p_values,
                method=p_adjust,
            )[1]

        stats_df.loc[
            valid,
            "p_adjusted",
        ] = adjusted_p_values

    stats_df["significance"] = (
        stats_df["p_adjusted"].map(_p_to_label)
    )

    return stats_df

def plot_delta_gini_trav_vs_trbv(
    data: pd.DataFrame,
    epitopes: Sequence[object],
    donor_col: str,
    epitope_col: str,
    epitope_colors: Mapping[object, str],

    chain_col: str = "chain",
    value_col: str = "delta_gini",
    chain_order: Sequence[str] = ("TRAV", "TRBV"),

    trav_color: str = "darkgray",
    trbv_color: str | None = None,

    ncols: int = 4,
    subplot_size: tuple[float, float] = (2.3, 3.4),
    sharey: bool = True,

    xlim: tuple[float, float] = (0.5, 2.5),
    ylim: tuple[float, float] | None = None,
    reference_line: float | None = 0.0,

    whis: tuple[float, float] = (5, 95),
    boxplot_width: float = 0.55,
    box_linewidth: float = 2.0,

    # None: epitope color for both median lines
    median_color: str | None = None,
    median_linewidth: float = 2.0,

    point_color: str = "black",
    point_size: float = 24,
    point_alpha: float = 0.75,
    point_jitter: float = 0.045,

    connect_pairs: bool = False,
    pair_line_color: str = "0.75",
    pair_linewidth: float = 0.8,
    pair_line_alpha: float = 0.6,

    p_adjust: str | None = "holm-sidak",
    min_pairs: int = 2,
    stat_label: str = "both",
    stat_bracket_y: float = 0.80,
    stat_text_y: float = 0.84,
    stat_linewidth: float = 1.0,

    title_y: float = 1.06,
    ylabel: str = (
        "Δ Gini\n"
        "(epitope − mean baseline)"
    ),

    random_state: int = 1,

    save_pdf: bool = False,
    pdf_path: str | Path | None = None,
    show: bool = True,
):
    """
    Plot donor-level delta Gini for TRAV versus TRBV within each epitope.
    """
    if isinstance(epitopes, str):
        epitopes = [epitopes]
    else:
        epitopes = list(epitopes)

    paired_wide, paired_long = (
        prepare_paired_delta_gini_data(
            data=data,
            epitopes=epitopes,
            donor_col=donor_col,
            epitope_col=epitope_col,
            chain_col=chain_col,
            value_col=value_col,
            chain_order=chain_order,
        )
    )

    stats_df = compute_delta_gini_chain_statistics(
        paired_wide=paired_wide,
        epitopes=epitopes,
        epitope_col=epitope_col,
        chain_order=chain_order,
        p_adjust=p_adjust,
        min_pairs=min_pairs,
    )

    n_epitopes = len(epitopes)
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

    group_1, group_2 = chain_order

    for axis, epitope in zip(
        axes_flat,
        epitopes,
    ):
        epitope_color = epitope_colors[epitope]

        resolved_group_2_color = (
            epitope_color
            if trbv_color is None
            else trbv_color
        )

        resolved_median_color = (
            epitope_color
            if median_color is None
            else median_color
        )

        group_colors = {
            group_1: trav_color,
            group_2: resolved_group_2_color,
        }

        epitope_data = paired_long.loc[
            paired_long[epitope_col].eq(epitope)
        ].copy()

        box_values = [
            epitope_data.loc[
                epitope_data[chain_col].eq(chain),
                value_col,
            ].to_numpy(dtype=float)
            for chain in chain_order
        ]

        valid_entries = [
            {
                "position": position,
                "chain": chain,
                "values": values,
                "color": group_colors[chain],
            }
            for position, (chain, values) in enumerate(
                zip(chain_order, box_values),
                start=1,
            )
            if len(values) > 0
        ]

        if valid_entries:
            boxplot_result = axis.boxplot(
                [
                    entry["values"]
                    for entry in valid_entries
                ],
                positions=[
                    entry["position"]
                    for entry in valid_entries
                ],
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

            for index, entry in enumerate(valid_entries):
                color = entry["color"]

                boxplot_result["boxes"][
                    index
                ].set_edgecolor(color)

                for whisker in boxplot_result["whiskers"][
                    2 * index : 2 * index + 2
                ]:
                    whisker.set_color(color)
                    whisker.set_linewidth(
                        box_linewidth
                    )

                for cap in boxplot_result["caps"][
                    2 * index : 2 * index + 2
                ]:
                    cap.set_color(color)
                    cap.set_linewidth(
                        box_linewidth
                    )

                boxplot_result["medians"][
                    index
                ].set_color(
                    resolved_median_color
                )

                boxplot_result["medians"][
                    index
                ].set_linewidth(
                    median_linewidth
                )

        epitope_wide = paired_wide.loc[
            paired_wide[epitope_col].eq(epitope)
        ].copy()

        donor_jitter = rng.normal(
            loc=0.0,
            scale=point_jitter,
            size=len(epitope_wide),
        )

        x_group_1 = 1.0 + donor_jitter
        x_group_2 = 2.0 + donor_jitter

        if connect_pairs:
            for index in range(len(epitope_wide)):
                axis.plot(
                    [
                        x_group_1[index],
                        x_group_2[index],
                    ],
                    [
                        epitope_wide.iloc[index][group_1],
                        epitope_wide.iloc[index][group_2],
                    ],
                    color=pair_line_color,
                    linewidth=pair_linewidth,
                    alpha=pair_line_alpha,
                    zorder=1,
                )

        axis.scatter(
            x_group_1,
            epitope_wide[group_1],
            color=point_color,
            s=point_size,
            alpha=point_alpha,
            linewidths=0,
            zorder=3,
        )

        axis.scatter(
            x_group_2,
            epitope_wide[group_2],
            color=point_color,
            s=point_size,
            alpha=point_alpha,
            linewidths=0,
            zorder=3,
        )

        if reference_line is not None:
            axis.axhline(
                reference_line,
                linestyle="--",
                linewidth=1,
                color="black",
                zorder=0,
            )

        axis.set_xticks([1, 2])
        axis.set_xticklabels(
            chain_order,
            rotation=30,
            ha="right",
        )

        axis.set_xlim(xlim)

        if ylim is not None:
            axis.set_ylim(ylim)

        stat_match = stats_df.loc[
            stats_df["epitope"].eq(epitope)
        ]

        stat_row = (
            stat_match.iloc[0]
            if not stat_match.empty
            else None
        )

        if len(epitope_wide) > 0:
            transform = axis.get_xaxis_transform()

            axis.plot(
                [1, 1, 2, 2],
                [
                    stat_bracket_y,
                    stat_bracket_y + 0.02,
                    stat_bracket_y + 0.02,
                    stat_bracket_y,
                ],
                transform=transform,
                color="black",
                linewidth=stat_linewidth,
                clip_on=False,
            )

            axis.text(
                1.5,
                stat_text_y,
                _format_stat_annotation(
                    stat_row=stat_row,
                    stat_label=stat_label,
                ),
                transform=transform,
                ha="center",
                va="bottom",
            )

        axis.set_title(
            str(epitope),
            y=title_y,
            pad=4,
        )

        axis.set_xlabel("")
        axis.set_ylabel(ylabel)

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        axis.tick_params(
            axis="y",
            which="both",
            labelleft=True,
        )

    for axis in axes_flat[n_epitopes:]:
        axis.remove()

    fig.tight_layout()

    if save_pdf:
        if pdf_path is None:
            pdf_path = "delta_gini_TRAV_vs_TRBV.pdf"

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
        "paired_wide": paired_wide,
        "paired_long": paired_long,
    }