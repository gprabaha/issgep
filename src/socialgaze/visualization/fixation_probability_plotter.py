import pdb

import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats.mstats import winsorize

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
        fig, axes = plt.subplots(1, len(categories), figsize=(12, 6), sharey=False)
        for i, category in enumerate(categories):
            ax = axes[i]
            sub_df = df[df["fixation_category"] == category].copy()
            _plot_violin_core(ax, sub_df, category, monkey_color_dict, violin_palette)
            ax.set_title(f"{category.capitalize()}")
        fig.suptitle("Fixation Probability Comparison — Overall", fontsize=16)
        filename = "joint_fixation_probabilities_overall.pdf"

    elif mode in {"interactivity", "segments"}:
        fig, axs = plt.subplots(2, len(categories), figsize=(12, 10), sharey=False)
        interactivities = ["interactive", "non_interactive"]

        for i, interactivity in enumerate(interactivities):
            for j, category in enumerate(categories):
                ax = axs[i][j]
                sub_df = df[
                    (df["interactivity"] == interactivity) &
                    (df["fixation_category"] == category)
                ].copy()
                _plot_violin_core(ax, sub_df, category, monkey_color_dict, violin_palette)
                ax.set_title(f"{interactivity.capitalize()} — {category}")
        title = "By Interactivity Segment" if mode == "segments" else "By Interactivity"
        filename = f"joint_fixation_probabilities_by_{mode}.pdf"
        fig.suptitle(f"Fixation Probability Comparison — {title}", fontsize=16)

    else:
        raise ValueError(f"Invalid mode: {mode}")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _finalize_and_save(fig, config.plot_dir, filename)



def _plot_violin_core(ax, full_df, category, monkey_color_dict, violin_palette):
    
    for col in ["P(m1)*P(m2)", "P(m1&m2)"]:
        full_df[col] = full_df.groupby("fixation_category")[col].transform(
            lambda x: pd.Series(winsorize(x, limits=[0.01, 0.01]), index=x.index)
        )

    melted = full_df.melt(
        id_vars=["monkey_pair"],
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
        # cut=0,
        legend=False,
        order=["P(m1)*P(m2)", "P(m1&m2)"],
        ax=ax
    )

    # Overlay medians: average per monkey_pair
    agg_df = full_df.groupby("monkey_pair")[["P(m1)*P(m2)", "P(m1&m2)"]].median().reset_index()
    mp_data = {
        row["monkey_pair"]: [row["P(m1)*P(m2)"], row["P(m1&m2)"]]
        for _, row in agg_df.iterrows()
    }

    _overlay_medians(ax, mp_data, monkey_color_dict)
    _annotate_ks(ax, full_df)


def _filter_outliers_iqr(df, group_col="Probability Type", value_col="Probability", k=3):
    """Remove outliers beyond k * IQR from each group."""
    filtered = []
    for group, group_df in df.groupby(group_col):
        q1 = group_df[value_col].quantile(0.25)
        q3 = group_df[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        filtered_df = group_df[(group_df[value_col] >= lower) & (group_df[value_col] <= upper)]
        filtered.append(filtered_df)
    return pd.concat(filtered)


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
