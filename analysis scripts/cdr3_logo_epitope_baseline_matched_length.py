
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as patches
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
import logomaker


AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


def strip_tcr_gene_prefix(x):
    x = str(x)
    return re.sub(r"TR[AB][VJ]", "", x)


def prepare_equal_length_sequences(seqs, length_mode="most_common", length=None):
    """Return sequences of one exact length and the selected length.

    When ``length`` is None, the length is selected from the supplied
    sequences using ``length_mode``. When ``length`` is supplied, the same
    exact length is enforced. This is used by the multi-plot function so the
    epitope repertoire determines the length and its paired baseline uses the
    identical length.
    """
    seqs = pd.Series(seqs).dropna().astype(str)
    seqs = seqs[seqs.str.len() > 0]

    if seqs.empty:
        raise ValueError("No non-empty CDR3 sequences are available.")

    if length is None:
        sequence_lengths = seqs.str.len()

        if length_mode == "most_common":
            length = int(sequence_lengths.mode().iloc[0])
        elif length_mode == "min":
            length = int(sequence_lengths.min())
        elif length_mode == "max":
            length = int(sequence_lengths.max())
        else:
            raise ValueError(
                "length_mode must be 'most_common', 'min', or 'max'."
            )
    else:
        length = int(length)

    seqs = seqs[seqs.str.len() == length]

    if seqs.empty:
        raise ValueError(
            f"No CDR3 sequences of the selected length {length} are available."
        )

    return seqs.tolist(), length


def cdr3_probability_matrix(seqs, alphabet=AA_ALPHABET):
    length = len(seqs[0])

    mat = pd.DataFrame(
        0.0,
        index=np.arange(length),
        columns=alphabet
    )

    for seq in seqs:
        for i, aa in enumerate(seq):
            if aa in alphabet:
                mat.loc[i, aa] += 1

    mat = mat.div(mat.sum(axis=1), axis=0).fillna(0)

    return mat


def probability_to_information_matrix(prob_mat):
    n_symbols = prob_mat.shape[1]

    H = -(
        prob_mat *
        np.log2(prob_mat.replace(0, np.nan))
    ).sum(axis=1).fillna(0)

    R = np.log2(n_symbols) - H

    return prob_mat.mul(R, axis=0)


def draw_scaled_text(
    ax,
    text,
    x,
    y,
    height,
    width=0.6,
    color="black",
    fontweight="bold",
):
    fp = FontProperties(weight=fontweight)

    tp = TextPath(
        (0, 0),
        text,
        size=1,
        prop=fp,
    )

    bbox = tp.get_extents()

    text_height = bbox.height
    text_width = bbox.width

    if text_height == 0 or text_width == 0:
        return

    scale_x = width / text_width
    scale_y = height / text_height

    trans = (
        transforms.Affine2D()
        .scale(scale_x, scale_y)
        .translate(x - width / 2, y)
        + ax.transData
    )

    patch = patches.PathPatch(
        tp,
        transform=trans,
        facecolor=color,
        edgecolor="none",
    )

    ax.add_patch(patch)


def _logo_title(base_title, n_sequences):
    """Add the number of plotted sequences to an individual logo title."""
    return f"{base_title}\n(n={n_sequences:,})"


def plot_gene_usage_logo(
    ax,
    values,
    title,
    max_genes=15,
    color="black",
):
    usage = (
        pd.Series(values)
        .dropna()
        .astype(str)
        .map(strip_tcr_gene_prefix)
        .value_counts(normalize=True)
        .head(max_genes)
        .sort_values()
    )

    y = 0

    for gene, freq in usage.items():
        draw_scaled_text(
            ax=ax,
            text=gene,
            x=0,
            y=y,
            height=freq,
            width=1,
            color=color,
        )
        y += freq

    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.tick_params(
                axis="y",
                labelsize=14,
            )
    ax.tick_params(
                axis="x",
                labelsize=14,
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_cdr3_logo(
    df,
    cdr3_col,
    v_col,
    j_col,
    title=None,
    length_mode="most_common",
    length=None,
    max_v_genes=15,
    max_j_genes=15,
    figsize=(12, 6),
    save_path=None,
):
    plot_df = df[[cdr3_col, v_col, j_col]].dropna().copy()

    seqs, selected_length = prepare_equal_length_sequences(
        plot_df[cdr3_col],
        length_mode=length_mode,
        length=length,
    )

    plot_df = plot_df[
        plot_df[cdr3_col].astype(str).str.len() == selected_length
    ].copy()

    n_sequences_used = len(seqs)
    prob_mat = cdr3_probability_matrix(seqs)
    info_mat = probability_to_information_matrix(prob_mat)

    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[1.2, 5.5, 1.2],
        height_ratios=[1, 1],
        wspace=0.35,
        hspace=0.55,
    )

    ax_v_bits = fig.add_subplot(gs[0, 0])
    ax_cdr3_bits = fig.add_subplot(gs[0, 1])
    ax_j_bits = fig.add_subplot(gs[0, 2])

    ax_v_prob = fig.add_subplot(gs[1, 0])
    ax_cdr3_prob = fig.add_subplot(gs[1, 1])
    ax_j_prob = fig.add_subplot(gs[1, 2])

    plot_gene_usage_logo(
        ax=ax_v_bits,
        values=plot_df[v_col],
        title=_logo_title(f"{v_col} usage", n_sequences_used),
        max_genes=max_v_genes,
    )

    logomaker.Logo(
        info_mat,
        ax=ax_cdr3_bits,
        color_scheme="chemistry",
    )

    ax_cdr3_bits.set_ylim(0, np.log2(len(AA_ALPHABET)))
    ax_cdr3_bits.set_ylabel("Bits")
    ax_cdr3_bits.set_title(
        _logo_title("CDR3 information logo", n_sequences_used)
    )
    ax_cdr3_bits.spines["top"].set_visible(False)
    ax_cdr3_bits.spines["right"].set_visible(False)
    ax_cdr3_bits.tick_params(axis="y", labelsize=14)
    ax_cdr3_bits.tick_params(axis="x", labelsize=14)

    plot_gene_usage_logo(
        ax=ax_j_bits,
        values=plot_df[j_col],
        title=_logo_title(f"{j_col} usage", n_sequences_used),
        max_genes=max_j_genes,
    )

    plot_gene_usage_logo(
        ax=ax_v_prob,
        values=plot_df[v_col],
        title=_logo_title(f"{v_col} usage", n_sequences_used),
        max_genes=max_v_genes,
    )

    logomaker.Logo(
        prob_mat,
        ax=ax_cdr3_prob,
        color_scheme="chemistry",
    )

    ax_cdr3_prob.set_ylim(0, 1)
    ax_cdr3_prob.set_ylabel("Probability")
    ax_cdr3_prob.set_xlabel("")
    ax_cdr3_prob.set_title(
        _logo_title("CDR3 probability logo", n_sequences_used)
    )
    ax_cdr3_prob.spines["top"].set_visible(False)
    ax_cdr3_prob.spines["right"].set_visible(False)
    ax_cdr3_prob.tick_params(axis="y", labelsize=14)
    ax_cdr3_prob.tick_params(axis="x", labelsize=14)

    plot_gene_usage_logo(
        ax=ax_j_prob,
        values=plot_df[j_col],
        title=_logo_title(f"{j_col} usage", n_sequences_used),
        max_genes=max_j_genes,
    )

    if title is not None:
        fig.suptitle(
            f"{title} (CDR3 length={selected_length})",
            fontsize=14,
            y=1.02,
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()

    return fig, {
        "n_sequences_used": n_sequences_used,
        "cdr3_length_used": selected_length,
        "probability_matrix": prob_mat,
        "information_matrix": info_mat,
    }


def plot_cdr3_logo_panel(
    axes,
    df,
    cdr3_col,
    v_col,
    j_col,
    title=None,
    length_mode="most_common",
    length=None,
    max_v_genes=15,
    max_j_genes=15,
    show_probability_logo=False,
):
    plot_df = df[[cdr3_col, v_col, j_col]].dropna().copy()

    seqs, selected_length = prepare_equal_length_sequences(
        plot_df[cdr3_col],
        length_mode=length_mode,
        length=length,
    )

    plot_df = plot_df[
        plot_df[cdr3_col].astype(str).str.len() == selected_length
    ].copy()

    n_sequences_used = len(seqs)
    prob_mat = cdr3_probability_matrix(seqs)
    info_mat = probability_to_information_matrix(prob_mat)

    if show_probability_logo:
        ax_v_bits, ax_cdr3_bits, ax_j_bits = axes[0]
        ax_v_prob, ax_cdr3_prob, ax_j_prob = axes[1]
    else:
        ax_v_bits, ax_cdr3_bits, ax_j_bits = axes

    plot_gene_usage_logo(
        ax=ax_v_bits,
        values=plot_df[v_col],
        title=_logo_title(f"{v_col} usage", n_sequences_used),
        max_genes=max_v_genes,
    )

    logomaker.Logo(
        info_mat,
        ax=ax_cdr3_bits,
        color_scheme="chemistry",
    )

    ax_cdr3_bits.set_ylim(0, np.log2(len(AA_ALPHABET)))
    ax_cdr3_bits.set_ylabel("Bits")
    ax_cdr3_bits.set_title(
        _logo_title("CDR3 information logo", n_sequences_used)
    )
    ax_cdr3_bits.spines["top"].set_visible(False)
    ax_cdr3_bits.spines["right"].set_visible(False)
    ax_cdr3_bits.tick_params(axis="y", labelsize=14)
    ax_cdr3_bits.tick_params(axis="x", labelsize=14)

    plot_gene_usage_logo(
        ax=ax_j_bits,
        values=plot_df[j_col],
        title=_logo_title(f"{j_col} usage", n_sequences_used),
        max_genes=max_j_genes,
    )

    if show_probability_logo:
        plot_gene_usage_logo(
            ax=ax_v_prob,
            values=plot_df[v_col],
            title=_logo_title(f"{v_col} usage", n_sequences_used),
            max_genes=max_v_genes,
        )

        logomaker.Logo(
            prob_mat,
            ax=ax_cdr3_prob,
            color_scheme="chemistry",
        )

        ax_cdr3_prob.set_ylim(0, 1)
        ax_cdr3_prob.set_ylabel("Probability")
        ax_cdr3_prob.set_xlabel("")
        ax_cdr3_prob.set_title(
            _logo_title("CDR3 probability logo", n_sequences_used)
        )
        ax_cdr3_prob.spines["top"].set_visible(False)
        ax_cdr3_prob.spines["right"].set_visible(False)
        ax_cdr3_prob.tick_params(axis="y", labelsize=14)
        ax_cdr3_prob.tick_params(axis="x", labelsize=14)

        plot_gene_usage_logo(
            ax=ax_j_prob,
            values=plot_df[j_col],
            title=_logo_title(f"{j_col} usage", n_sequences_used),
            max_genes=max_j_genes,
        )

    if title is not None:
        ax_cdr3_bits.text(
            0.5,
            1.48,
            f"{title} — CDR3 length {selected_length}",
            transform=ax_cdr3_bits.transAxes,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    return {
        "title": title,
        "n_sequences_used": n_sequences_used,
        "cdr3_length_used": selected_length,
        "probability_matrix": prob_mat,
        "information_matrix": info_mat,
    }


def plot_cdr3_logo_multi(
    df_inputs,
    cdr3_col,
    v_col,
    j_col,
    length_mode="most_common",
    length=None,
    max_v_genes=15,
    max_j_genes=15,
    figsize_per_df=(12, 3),
    show_probability_logo=False,
    save_path=None,
):
    """Plot alternating epitope and baseline logo rows.

    ``df_inputs`` must be ordered as:

        epitope 1, baseline 1, epitope 2, baseline 2, ...

    For each pair, the epitope dataframe selects the CDR3 length using
    ``length_mode``. The paired baseline is then filtered to exactly that same
    length. Supplying ``length`` overrides automatic selection and applies the
    specified length to all epitope-baseline pairs.
    """
    if isinstance(df_inputs, dict):
        df_items = list(df_inputs.items())
    else:
        df_items = []

        for i, item in enumerate(df_inputs):
            if isinstance(item, tuple):
                df_items.append(item)
            else:
                df_items.append((f"Dataset {i + 1}", item))

    n_dfs = len(df_items)

    if n_dfs == 0:
        raise ValueError(
            "df_inputs must contain at least one epitope-baseline pair."
        )

    if n_dfs % 2 != 0:
        raise ValueError(
            "df_inputs must contain an even number of datasets ordered as "
            "epitope, baseline, epitope, baseline, ..."
        )

    pair_lengths = []

    for pair_start in range(0, n_dfs, 2):
        epitope_title, epitope_df = df_items[pair_start]

        epitope_plot_df = epitope_df[
            [cdr3_col, v_col, j_col]
        ].dropna().copy()

        try:
            _, selected_pair_length = prepare_equal_length_sequences(
                epitope_plot_df[cdr3_col],
                length_mode=length_mode,
                length=length,
            )
        except ValueError as error:
            raise ValueError(
                f"Could not select a CDR3 length from epitope dataset "
                f"{epitope_title!r}: {error}"
            ) from error

        pair_lengths.append(selected_pair_length)

    rows_per_df = 2 if show_probability_logo else 1

    fig = plt.figure(
        figsize=(
            figsize_per_df[0],
            figsize_per_df[1] * n_dfs * rows_per_df,
        )
    )

    gs = fig.add_gridspec(
        nrows=rows_per_df * n_dfs,
        ncols=3,
        width_ratios=[1.2, 5.5, 1.2],
        height_ratios=[1] * (rows_per_df * n_dfs),
        wspace=0.35,
        hspace=1.0,
    )

    results = {}

    for i, (title, df) in enumerate(df_items):
        row0 = rows_per_df * i
        pair_index = i // 2
        selected_pair_length = pair_lengths[pair_index]
        dataset_role = "epitope" if i % 2 == 0 else "baseline"

        if show_probability_logo:
            axes = np.array([
                [
                    fig.add_subplot(gs[row0, 0]),
                    fig.add_subplot(gs[row0, 1]),
                    fig.add_subplot(gs[row0, 2]),
                ],
                [
                    fig.add_subplot(gs[row0 + 1, 0]),
                    fig.add_subplot(gs[row0 + 1, 1]),
                    fig.add_subplot(gs[row0 + 1, 2]),
                ],
            ])
        else:
            axes = np.array([
                fig.add_subplot(gs[row0, 0]),
                fig.add_subplot(gs[row0, 1]),
                fig.add_subplot(gs[row0, 2]),
            ])

        try:
            panel_result = plot_cdr3_logo_panel(
                axes=axes,
                df=df,
                cdr3_col=cdr3_col,
                v_col=v_col,
                j_col=j_col,
                title=title,
                length_mode=length_mode,
                length=selected_pair_length,
                max_v_genes=max_v_genes,
                max_j_genes=max_j_genes,
                show_probability_logo=show_probability_logo,
            )
        except ValueError as error:
            raise ValueError(
                f"Could not plot {dataset_role} dataset {title!r} at the "
                f"epitope-selected CDR3 length {selected_pair_length}: {error}"
            ) from error

        panel_result["dataset_role"] = dataset_role
        panel_result["pair_index"] = pair_index
        panel_result["length_selected_from"] = df_items[2 * pair_index][0]
        results[title] = panel_result

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()

    return fig, results


###### Plotting of kmer logos

# ============================================================
# KMER-ANCHORED CDR3 ALIGNMENT
# ============================================================

def _find_kmer_start(seq, kmer):
    """
    Find the k-mer occurrence to use for alignment.

    If the k-mer occurs more than once, use the occurrence
    whose centre is closest to the centre of the CDR3.
    """

    starts = [
        m.start()
        for m in re.finditer(
            f"(?={re.escape(kmer)})",
            seq,
        )
    ]

    if len(starts) == 0:
        return None

    if len(starts) == 1:
        return starts[0]

    seq_center = (len(seq) - 1) / 2
    kmer_half = (len(kmer) - 1) / 2

    return min(
        starts,
        key=lambda x: abs(
            (x + kmer_half) - seq_center
        ),
    )


def prepare_kmer_aligned_sequences(
    df,
    cdr3_col,
    v_col,
    j_col,
    kmer,
    start_anchor="C",
    end_anchor="F",
    max_left=None,
    max_right=None,
):
    """
    Align variable-length CDR3 sequences around a shared k-mer.

    Alignment scheme
    ----------------

    C + N-terminal flank + gaps + KMER + gaps +
        C-terminal flank + F

    This ensures that:

        1. initial C is aligned
        2. the selected k-mer is aligned
        3. terminal F is aligned

    Residues N-terminal to the k-mer are aligned relative to C.
    Residues C-terminal to the k-mer are aligned relative to F.
    Length differences are therefore represented by gaps placed
    immediately around the k-mer.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    cdr3_col : str
        CDR3 amino-acid column.

    v_col : str
        V-gene column.

    j_col : str
        J-gene column.

    kmer : str
        Exact k-mer used as the central alignment anchor.

    start_anchor : str
        N-terminal anchor. Default = "C".

    end_anchor : str
        C-terminal anchor. Default = "F".

    max_left, max_right : int or None
        Optional fixed flank widths. Useful when several datasets
        should use exactly the same alignment coordinates.

    Returns
    -------
    aligned_df : pd.DataFrame
        Filtered dataframe with additional alignment columns.

    metadata : dict
        Alignment information.
    """

    if not isinstance(kmer, str) or len(kmer) == 0:
        raise ValueError("kmer must be a non-empty string.")

    plot_df = (
        df[
            [
                cdr3_col,
                v_col,
                j_col,
            ]
        ]
        .dropna()
        .copy()
    )

    if plot_df.empty:
        raise ValueError(
            "No complete CDR3/V/J observations are available."
        )

    records = []

    n_no_anchor = 0
    n_no_kmer = 0
    n_anchor_overlap = 0
    n_multiple_kmer = 0

    for idx, row in plot_df.iterrows():

        seq = str(row[cdr3_col])

        # ------------------------------------------------
        # Anchor using the first C and last F
        # ------------------------------------------------
        c_pos = seq.find(start_anchor)
        f_pos = seq.rfind(end_anchor)

        if (
            c_pos == -1
            or f_pos == -1
            or f_pos <= c_pos
        ):
            n_no_anchor += 1
            continue

        anchored_seq = seq[c_pos:f_pos + 1]

        # ------------------------------------------------
        # Find k-mer
        # ------------------------------------------------
        starts = [
            m.start()
            for m in re.finditer(
                f"(?={re.escape(kmer)})",
                anchored_seq,
            )
        ]

        if len(starts) == 0:
            n_no_kmer += 1
            continue

        if len(starts) > 1:
            n_multiple_kmer += 1

        kmer_start = _find_kmer_start(
            anchored_seq,
            kmer,
        )

        kmer_end = (
            kmer_start +
            len(kmer)
        )

        # K-mer should lie between C and F,
        # rather than include either anchor.
        if (
            kmer_start < 1
            or kmer_end > len(anchored_seq) - 1
        ):
            n_anchor_overlap += 1
            continue

        # ------------------------------------------------
        # Flanks excluding C, kmer and F
        # ------------------------------------------------
        left_flank = anchored_seq[
            1:kmer_start
        ]

        right_flank = anchored_seq[
            kmer_end:-1
        ]

        records.append({
            "_original_index": idx,
            "_cdr3_original": seq,
            "_cdr3_anchored": anchored_seq,
            "_left_flank": left_flank,
            "_right_flank": right_flank,
            "_left_length": len(left_flank),
            "_right_length": len(right_flank),
        })

    if len(records) == 0:
        raise ValueError(
            f"No sequences containing k-mer {kmer!r} "
            f"with valid {start_anchor}...{end_anchor} anchors."
        )

    alignment_df = pd.DataFrame(records)

    # ----------------------------------------------------
    # Determine required flank widths
    # ----------------------------------------------------
    observed_max_left = int(
        alignment_df["_left_length"].max()
    )

    observed_max_right = int(
        alignment_df["_right_length"].max()
    )

    if max_left is None:
        max_left = observed_max_left

    if max_right is None:
        max_right = observed_max_right

    if max_left < observed_max_left:
        raise ValueError(
            f"max_left={max_left} is smaller than the "
            f"observed maximum left flank "
            f"({observed_max_left})."
        )

    if max_right < observed_max_right:
        raise ValueError(
            f"max_right={max_right} is smaller than the "
            f"observed maximum right flank "
            f"({observed_max_right})."
        )

    # ----------------------------------------------------
    # Construct alignment
    #
    # C LEFT ---- KMER ---- RIGHT F
    #
    # Left flank:
    #   residues remain aligned relative to C.
    #
    # Right flank:
    #   residues remain aligned relative to F.
    #
    # Gaps therefore accumulate immediately around kmer.
    # ----------------------------------------------------
    aligned_sequences = []

    for _, row in alignment_df.iterrows():

        left = row["_left_flank"]
        right = row["_right_flank"]

        left_gaps = (
            "-" *
            (max_left - len(left))
        )

        right_gaps = (
            "-" *
            (max_right - len(right))
        )

        aligned = (
            start_anchor
            + left
            + left_gaps
            + kmer
            + right_gaps
            + right
            + end_anchor
        )

        aligned_sequences.append(aligned)

    alignment_df["_cdr3_aligned"] = (
        aligned_sequences
    )

    # ----------------------------------------------------
    # Bring V/J etc. back from original dataframe
    # ----------------------------------------------------
    aligned_df = (
        alignment_df
        .merge(
            plot_df,
            left_on="_original_index",
            right_index=True,
            how="left",
        )
        .reset_index(drop=True)
    )

    kmer_start_aligned = (
        1 + max_left
    )

    kmer_end_aligned = (
        kmer_start_aligned +
        len(kmer)
    )

    metadata = {
        "kmer": kmer,
        "n_sequences_input": len(plot_df),
        "n_sequences_used": len(aligned_df),

        "n_dropped_no_anchor": n_no_anchor,
        "n_dropped_no_kmer": n_no_kmer,
        "n_dropped_anchor_overlap": n_anchor_overlap,

        "n_multiple_kmer_occurrences":
            n_multiple_kmer,

        "max_left_flank": max_left,
        "max_right_flank": max_right,

        "aligned_length":
            len(aligned_sequences[0]),

        "kmer_start":
            kmer_start_aligned,

        # Python-style exclusive end
        "kmer_end":
            kmer_end_aligned,
    }

    return aligned_df, metadata


# ============================================================
# ALIGNMENT-AWARE PROBABILITY MATRIX
# ============================================================

def cdr3_probability_matrix_aligned(
    seqs,
    alphabet=AA_ALPHABET,
    gap_char="-",
):
    """
    Calculate amino-acid probabilities from aligned CDR3s.

    Gaps are NOT treated as amino acids.

    Importantly, probabilities are divided by the TOTAL number
    of sequences, not only the number of non-gap residues at
    each position.

    Therefore:
        sum(AA probabilities at position i)
        =
        fraction of sequences occupied by an amino acid there.

    This means gap-rich positions produce shorter probability
    stacks rather than artificially normalized stacks of height 1.
    """

    seqs = list(seqs)

    if len(seqs) == 0:
        raise ValueError(
            "No aligned CDR3 sequences supplied."
        )

    lengths = {
        len(seq)
        for seq in seqs
    }

    if len(lengths) != 1:
        raise ValueError(
            "Aligned CDR3 sequences must all "
            "have equal length."
        )

    n_sequences = len(seqs)
    aligned_length = len(seqs[0])

    mat = pd.DataFrame(
        0.0,
        index=np.arange(aligned_length),
        columns=alphabet,
    )

    for seq in seqs:

        for i, aa in enumerate(seq):

            if aa in alphabet:
                mat.loc[i, aa] += 1

    # Divide by ALL sequences.
    mat = mat / n_sequences

    occupancy = (
        mat.sum(axis=1)
    )

    gap_probability = (
        1 - occupancy
    )

    return (
        mat,
        occupancy,
        gap_probability,
    )


def probability_to_information_matrix_aligned(
    prob_mat,
):
    """
    Convert alignment-aware amino-acid probabilities to
    information content.

    For gap-containing positions:

      1. AA frequencies are normalized among occupied sequences
         to calculate Shannon entropy.

      2. Information content is multiplied by positional
         occupancy.

    Thus highly conserved residues occurring in only a small
    fraction of sequences do not generate artificially tall
    information stacks.
    """

    n_symbols = prob_mat.shape[1]

    occupancy = (
        prob_mat.sum(axis=1)
    )

    conditional_prob = (
        prob_mat
        .div(
            occupancy.replace(
                0,
                np.nan,
            ),
            axis=0,
        )
        .fillna(0)
    )

    H = -(
        conditional_prob
        *
        np.log2(
            conditional_prob.replace(
                0,
                np.nan,
            )
        )
    ).sum(
        axis=1
    ).fillna(0)

    R = (
        np.log2(n_symbols)
        - H
    )

    # Down-weight information content
    # by positional occupancy.
    effective_R = (
        R * occupancy
    )

    info_mat = (
        conditional_prob
        .mul(
            effective_R,
            axis=0,
        )
    )

    return info_mat


# ============================================================
# CDR3 LOGO AXIS
# ============================================================

def plot_aligned_cdr3_logo_axis(
    ax,
    prob_mat,
    info_mat,
    kmer,
    kmer_start,
    logo_type="information",
    highlight_kmer=False,
    min_probability_height=0.1,
):
    """
    Plot aligned CDR3 sequence logo.

    For probability logos, positions with non-zero total
    probability below `min_probability_height` are visually
    expanded to that minimum height for readability.

    The original probability matrix is NOT modified.

    A dashed horizontal marker at the minimum display height
    identifies positions that were visually expanded.
    """

    if logo_type not in {
        "information",
        "probability",
    }:
        raise ValueError(
            "logo_type must be "
            "'information' or 'probability'."
        )

    # --------------------------------------------------
    # Probability logo
    # --------------------------------------------------
    if logo_type == "probability":

        # Copy so that the real probability matrix
        # is never modified.
        display_mat = prob_mat.copy()

        occupancy = prob_mat.sum(axis=1)

        visually_scaled_positions = []

        for pos in prob_mat.index:

            true_height = occupancy.loc[pos]

            # Only scale positions containing at least
            # one amino acid.
            if (
                true_height > 0
                and true_height < min_probability_height
            ):

                scale_factor = (
                    min_probability_height /
                    true_height
                )

                display_mat.loc[pos] *= (
                    scale_factor
                )

                visually_scaled_positions.append(
                    pos
                )

        matrix = display_mat
        ylabel = "Probability"
        title = "CDR3 probability logo"
        ylim = (0, 1)

    # --------------------------------------------------
    # Information logo
    # --------------------------------------------------
    else:

        matrix = info_mat
        ylabel = "Bits"
        title = "CDR3 information logo"

        ylim = (
            0,
            np.log2(
                len(AA_ALPHABET)
            ),
        )

        visually_scaled_positions = []

    # --------------------------------------------------
    # Draw logo
    # --------------------------------------------------
    logomaker.Logo(
        matrix,
        ax=ax,
        color_scheme="chemistry",
    )

    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="y",
        labelsize=14,
    )

    ax.tick_params(
        axis="x",
        labelsize=12,
    )

    # --------------------------------------------------
    # Mark positions that were scaled to the
    # minimum visible height.
    #
    # Dashed line across the width of each affected
    # sequence-logo position.
    # --------------------------------------------------
    if (
        logo_type == "probability"
        and min_probability_height is not None
    ):

        for pos in visually_scaled_positions:

            ax.hlines(
                y=min_probability_height,
                xmin=pos - 0.45,
                xmax=pos + 0.45,
                linestyles="--",
                linewidth=0.8,
                color="black",
                zorder=10,
            )

    # --------------------------------------------------
    # Highlight k-mer
    # --------------------------------------------------
    if highlight_kmer:

        kmer_end = (
            kmer_start
            + len(kmer)
        )

        ax.axvspan(
            kmer_start - 0.5,
            kmer_end - 0.5,
            alpha=0.08,
        )

        ax.text(
            (
                kmer_start
                + kmer_end
                - 1
            ) / 2,
            ax.get_ylim()[1] * 0.98,
            kmer,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )


# ============================================================
# SINGLE LOGO
# ============================================================

def plot_cdr3_logo_aligned(
    df,
    cdr3_col,
    v_col,
    j_col,
    kmer,
    title=None,
    cdr3_logo_type="information",
    max_v_genes=15,
    max_j_genes=15,
    start_anchor="C",
    end_anchor="F",
    highlight_kmer=True,
    figsize=(12, 3),
    save_path=None,
):
    """
    Plot:

        V probability | aligned CDR3 | J probability

    CDR3 sequences are aligned using:

        initial C -> kmer -> terminal F

    with gaps inserted around the k-mer.
    """

    aligned_df, alignment = (
        prepare_kmer_aligned_sequences(
            df=df,
            cdr3_col=cdr3_col,
            v_col=v_col,
            j_col=j_col,
            kmer=kmer,
            start_anchor=start_anchor,
            end_anchor=end_anchor,
        )
    )

    seqs = (
        aligned_df["_cdr3_aligned"]
        .tolist()
    )

    n_sequences_used = len(seqs)

    (
        prob_mat,
        occupancy,
        gap_probability,
    ) = cdr3_probability_matrix_aligned(
        seqs
    )

    info_mat = (
        probability_to_information_matrix_aligned(
            prob_mat
        )
    )

    fig = plt.figure(
        figsize=figsize
    )

    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[
            1.2,
            5.5,
            1.2,
        ],
        wspace=0.35,
    )

    ax_v = fig.add_subplot(
        gs[0, 0]
    )

    ax_cdr3 = fig.add_subplot(
        gs[0, 1]
    )

    ax_j = fig.add_subplot(
        gs[0, 2]
    )

    # ----------------------------------------------------
    # V probability
    # ----------------------------------------------------
    plot_gene_usage_logo(
        ax=ax_v,
        values=aligned_df[v_col],
        title=_logo_title(
            f"{v_col} usage",
            n_sequences_used,
        ),
        max_genes=max_v_genes,
    )

    # ----------------------------------------------------
    # CDR3
    # ----------------------------------------------------
    plot_aligned_cdr3_logo_axis(
        ax=ax_cdr3,
        prob_mat=prob_mat,
        info_mat=info_mat,
        kmer=kmer,
        kmer_start=alignment[
            "kmer_start"
        ],
        logo_type=cdr3_logo_type,
        highlight_kmer=highlight_kmer,
    )

    ax_cdr3.set_title(
        _logo_title(
            (
                "CDR3 probability logo"
                if cdr3_logo_type
                == "probability"
                else
                "CDR3 information logo"
            ),
            n_sequences_used,
        )
    )

    # ----------------------------------------------------
    # J probability
    # ----------------------------------------------------
    plot_gene_usage_logo(
        ax=ax_j,
        values=aligned_df[j_col],
        title=_logo_title(
            f"{j_col} usage",
            n_sequences_used,
        ),
        max_genes=max_j_genes,
    )

    if title is not None:

        fig.suptitle(
            f"{title} — motif: {kmer}",
            fontsize=14,
            y=1.04,
        )

    fig.tight_layout()

    if save_path is not None:

        fig.savefig(
            save_path,
            bbox_inches="tight",
        )

    plt.show()

    result = {
        "title": title,
        "kmer": kmer,
        "n_sequences_used":
            n_sequences_used,

        "aligned_sequences":
            seqs,

        "aligned_dataframe":
            aligned_df,

        "probability_matrix":
            prob_mat,

        "information_matrix":
            info_mat,

        "occupancy":
            occupancy,

        "gap_probability":
            gap_probability,

        **alignment,
    }

    return fig, result


# ============================================================
# PANEL VERSION
# ============================================================

def plot_cdr3_logo_panel_aligned(
    axes,
    df,
    cdr3_col,
    v_col,
    j_col,
    kmer,
    title=None,
    cdr3_logo_type="information",
    max_v_genes=15,
    max_j_genes=15,
    start_anchor="C",
    end_anchor="F",
    highlight_kmer=True,
):
    """
    Plot one aligned V-CDR3-J panel onto three supplied axes.
    """

    aligned_df, alignment = (
        prepare_kmer_aligned_sequences(
            df=df,
            cdr3_col=cdr3_col,
            v_col=v_col,
            j_col=j_col,
            kmer=kmer,
            start_anchor=start_anchor,
            end_anchor=end_anchor,
        )
    )

    seqs = (
        aligned_df["_cdr3_aligned"]
        .tolist()
    )

    n_sequences_used = len(seqs)

    (
        prob_mat,
        occupancy,
        gap_probability,
    ) = cdr3_probability_matrix_aligned(
        seqs
    )

    info_mat = (
        probability_to_information_matrix_aligned(
            prob_mat
        )
    )

    ax_v, ax_cdr3, ax_j = axes

    # ----------------------------------------------------
    # V
    # ----------------------------------------------------
    plot_gene_usage_logo(
        ax=ax_v,
        values=aligned_df[v_col],
        title=_logo_title(
            f"{v_col} usage",
            n_sequences_used,
        ),
        max_genes=max_v_genes,
    )

    # ----------------------------------------------------
    # CDR3
    # ----------------------------------------------------
    plot_aligned_cdr3_logo_axis(
        ax=ax_cdr3,
        prob_mat=prob_mat,
        info_mat=info_mat,
        kmer=kmer,
        kmer_start=alignment[
            "kmer_start"
        ],
        logo_type=cdr3_logo_type,
        highlight_kmer=highlight_kmer,
    )

    ax_cdr3.set_title(
        _logo_title(
            (
                "CDR3 probability logo"
                if cdr3_logo_type
                == "probability"
                else
                "CDR3 information logo"
            ),
            n_sequences_used,
        )
    )

    # ----------------------------------------------------
    # J
    # ----------------------------------------------------
    plot_gene_usage_logo(
        ax=ax_j,
        values=aligned_df[j_col],
        title=_logo_title(
            f"{j_col} usage",
            n_sequences_used,
        ),
        max_genes=max_j_genes,
    )

    if title is not None:

        ax_cdr3.text(
            0.5,
            1.35,
            f"{title} — motif: {kmer}",
            transform=ax_cdr3.transAxes,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    return {
        "title": title,
        "kmer": kmer,

        "n_sequences_used":
            n_sequences_used,

        "aligned_sequences":
            seqs,

        "aligned_dataframe":
            aligned_df,

        "probability_matrix":
            prob_mat,

        "information_matrix":
            info_mat,

        "occupancy":
            occupancy,

        "gap_probability":
            gap_probability,

        **alignment,
    }


# ============================================================
# MULTI-PANEL VERSION -- NO BASELINE MATCHING
# ============================================================

def plot_cdr3_logo_multi_aligned(
    df_inputs,
    cdr3_col,
    v_col,
    j_col,
    kmer,
    cdr3_logo_type="information",
    max_v_genes=15,
    max_j_genes=15,
    start_anchor="C",
    end_anchor="F",
    highlight_kmer=True,
    figsize_per_df=(12, 3),
    save_path=None,
):
    """
    Plot multiple independently aligned datasets.

    df_inputs can be:

        {
            "Motif A": df_a,
            "Motif B": df_b,
        }

    or:

        [
            ("Motif A", df_a),
            ("Motif B", df_b),
        ]

    `kmer` can be:

        1. one string applied to every dataframe

        2. a dict keyed by dataset title:

           {
               "Motif A": "ABC",
               "Motif B": "XYZ",
           }

        3. a list/tuple in the same order as df_inputs.
    """

    # ----------------------------------------------------
    # Normalize input datasets
    # ----------------------------------------------------
    if isinstance(
        df_inputs,
        dict,
    ):

        df_items = list(
            df_inputs.items()
        )

    else:

        df_items = []

        for i, item in enumerate(
            df_inputs
        ):

            if isinstance(
                item,
                tuple,
            ):

                df_items.append(
                    item
                )

            else:

                df_items.append(
                    (
                        f"Dataset {i + 1}",
                        item,
                    )
                )

    n_dfs = len(df_items)

    if n_dfs == 0:
        raise ValueError(
            "df_inputs must contain "
            "at least one dataframe."
        )

    # ----------------------------------------------------
    # Resolve motif for each dataframe
    # ----------------------------------------------------
    def resolve_kmer(
        title,
        i,
    ):

        if isinstance(
            kmer,
            dict,
        ):

            if title not in kmer:
                raise KeyError(
                    f"No kmer supplied for "
                    f"dataset {title!r}."
                )

            return kmer[title]

        if isinstance(
            kmer,
            (list, tuple),
        ):

            if len(kmer) != n_dfs:
                raise ValueError(
                    "When kmer is a list/tuple, "
                    "its length must match df_inputs."
                )

            return kmer[i]

        return kmer

    # ----------------------------------------------------
    # Figure
    # ----------------------------------------------------
    fig = plt.figure(
        figsize=(
            figsize_per_df[0],
            figsize_per_df[1]
            * n_dfs,
        )
    )

    gs = fig.add_gridspec(
        nrows=n_dfs,
        ncols=3,
        width_ratios=[
            1.2,
            5.5,
            1.2,
        ],
        height_ratios=[
            1
        ] * n_dfs,
        wspace=0.35,
        hspace=1.0,
    )

    results = {}

    # ----------------------------------------------------
    # Plot each dataframe independently
    # ----------------------------------------------------
    for i, (
        title,
        df,
    ) in enumerate(df_items):

        this_kmer = resolve_kmer(
            title,
            i,
        )

        axes = np.array([
            fig.add_subplot(
                gs[i, 0]
            ),
            fig.add_subplot(
                gs[i, 1]
            ),
            fig.add_subplot(
                gs[i, 2]
            ),
        ])

        panel_result = (
            plot_cdr3_logo_panel_aligned(
                axes=axes,
                df=df,
                cdr3_col=cdr3_col,
                v_col=v_col,
                j_col=j_col,
                kmer=this_kmer,
                title=title,
                cdr3_logo_type=
                    cdr3_logo_type,
                max_v_genes=
                    max_v_genes,
                max_j_genes=
                    max_j_genes,
                start_anchor=
                    start_anchor,
                end_anchor=
                    end_anchor,
                highlight_kmer=
                    highlight_kmer,
            )
        )

        results[title] = (
            panel_result
        )

    fig.tight_layout()

    if save_path is not None:

        fig.savefig(
            save_path,
            bbox_inches="tight",
        )

    plt.show()

    return fig, results
