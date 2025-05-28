import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

def plot_joint_vs_marginal_violin(config, detector, mode: str):
    df = detector.get_data(mode)
    categories = ["face", "out_of_roi"]
    df = df[df["fixation_category"].isin(categories)].copy()
    df["joint"] = df["P(m1&m2)"]
    df["marginal"] = df["P(m1)*P(m2)"]

    violin_palette = config.violin_palette
    monkey_pairs = df["monkey_pair"].unique()
    monkey_color_dict = _get_monkey_palette(monkey_pairs)

    if mode == "overall":
        agg_df = _compute_median_df(df, ["monkey_pair", "fixation_category"])
        fig, axes = plt.subplots(1, len(categories), figsize=(12, 6), sharey=False)
        for i, category in enumerate(categories):
            ax = axes[i]
            _plot_violin_core(ax, agg_df, df, category, monkey_color_dict, violin_palette)
            ax.set_title(f"{category.capitalize()}")
        fig.suptitle("Fixation Probability Comparison — Overall", fontsize=16)
        filename = "joint_fixation_probabilities_overall.pdf"

    elif mode == "interactivity":
        agg_df = _compute_median_df(df, ["monkey_pair", "fixation_category", "interactivity"])
        interactivities = sorted(df["interactivity"].unique())
        fig, axs = plt.subplots(len(interactivities), len(categories), figsize=(12, 5 * len(interactivities)), sharey=False)
        if len(interactivities) == 1:
            axs = [axs]

        for i, interactivity in enumerate(interactivities):
            for j, category in enumerate(categories):
                ax = axs[i][j] if len(interactivities) > 1 else axs[0][j]
                sub_agg = agg_df[(agg_df["interactivity"] == interactivity) & (agg_df["fixation_category"] == category)]
                stat_df = df[(df["interactivity"] == interactivity) & (df["fixation_category"] == category)]
                _plot_violin_core(ax, sub_agg, stat_df, category, monkey_color_dict, violin_palette)
                ax.set_title(f"{interactivity} — {category}")
        fig.suptitle("Fixation Probability Comparison — By Interactivity", fontsize=16)
        filename = "joint_fixation_probabilities_by_interactivity.pdf"

    elif mode == "segments":
        agg_df = _compute_median_df(df, ["monkey_pair", "fixation_category", "interactivity", "segment_id"])
        interactivities = sorted(df["interactivity"].unique())
        fig, axs = plt.subplots(len(interactivities), len(categories), figsize=(12, 5 * len(interactivities)), sharey=False)
        if len(interactivities) == 1:
            axs = [axs]

        for i, interactivity in enumerate(interactivities):
            for j, category in enumerate(categories):
                ax = axs[i][j] if len(interactivities) > 1 else axs[0][j]
                sub_agg = agg_df[(agg_df["interactivity"] == interactivity) & (agg_df["fixation_category"] == category)]
                stat_df = df[(df["interactivity"] == interactivity) & (df["fixation_category"] == category)]
                _plot_violin_core(ax, sub_agg, stat_df, category, monkey_color_dict, violin_palette, id_var="segment_id")
                ax.set_title(f"{interactivity} — {category}")
        fig.suptitle("Fixation Probability Comparison — By Interactivity Segment", fontsize=16)
        filename = "joint_fixation_probabilities_by_interactivity_segment.pdf"

    else:
        raise ValueError(f"Invalid mode: {mode}")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _finalize_and_save(fig, config.plot_dir, filename)


def _plot_violin_core(ax, agg_df, stat_df, category, monkey_color_dict, violin_palette, id_var=None):
    id_vars = ["monkey_pair"] if id_var is None else ["monkey_pair", id_var]
    melted = agg_df.melt(
        id_vars=id_vars,
        value_vars=["P(m1)*P(m2)", "P(m1&m2)"],
        var_name="Probability Type",
        value_name="Probability"
    )

    sns.violinplot(
        data=melted,
        x="Probability Type",
        y="Probability",
        hue="Probability Type",
        palette=violin_palette,
        inner="quartile",
        legend=False,
        ax=ax
    )

    mp_data = {}
    for _, row in agg_df.iterrows():
        mp = row["monkey_pair"]
        y_vals = [row["P(m1)*P(m2)"], row["P(m1&m2)"]]
        if mp in mp_data:
            # average across segments if applicable
            mp_data[mp][0] += y_vals[0]
            mp_data[mp][1] += y_vals[1]
            mp_data[mp][2] += 1
        else:
            mp_data[mp] = [y_vals[0], y_vals[1], 1]

    mp_data = {k: [v[0]/v[2], v[1]/v[2]] for k, v in mp_data.items()}  # avg

    _overlay_medians(ax, mp_data, monkey_color_dict)
    _annotate_ks(ax, stat_df)


def _compute_median_df(df, group_cols):
    return df.groupby(group_cols).agg({
        "P(m1)*P(m2)": "median",
        "P(m1&m2)": "median"
    }).reset_index()


def _get_monkey_palette(monkey_pairs):
    palette = sns.color_palette("Set2", n_colors=len(monkey_pairs))
    return {mp: palette[i] for i, mp in enumerate(monkey_pairs)}


def _overlay_medians(ax, monkey_pair_data, monkey_color_dict):
    for mp, y_vals in monkey_pair_data.items():
        if not isinstance(y_vals, list) or len(y_vals) != 2:
            logger.warning(f"Invalid y_vals for {mp}: {y_vals}")
            continue
        jitter = 0.01 * (hash(mp) % 10 - 5)
        ax.plot([0 + jitter, 1 + jitter], y_vals, color=monkey_color_dict.get(mp, "gray"), alpha=0.4)
        ax.scatter([0 + jitter], [y_vals[0]], color=monkey_color_dict.get(mp, "gray"), s=30, alpha=0.7)
        ax.scatter([1 + jitter], [y_vals[1]], color=monkey_color_dict.get(mp, "gray"), s=30, alpha=0.7)


def _annotate_ks(ax, group_df):
    ks_stat, p_val = ks_2samp(group_df["P(m1)*P(m2)"], group_df["P(m1&m2)"])
    marker = _get_significance_marker(p_val)
    ax.text(0.5, 1.05, f"KS test: {marker}", transform=ax.transAxes, ha="center", fontsize=12)


def _get_significance_marker(p_val):
    if p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < 0.05:
        return '*'
    else:
        return 'NS'


def _finalize_and_save(fig, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    fig.patch.set_alpha(0)
    for ax in fig.get_axes():
        ax.set_facecolor("none")
    fig.savefig(save_path, format="pdf", transparent=True, bbox_inches="tight")
    plt.close(fig)
