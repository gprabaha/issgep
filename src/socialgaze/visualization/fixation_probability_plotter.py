import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import ks_2samp
from socialgaze.utils.loading_utils import load_df_from_pkl


def plot_joint_vs_marginal_violin(config):
    df = load_df_from_pkl(config.fix_prob_df_path)
    categories = ["face", "out_of_roi"]
    df = df[df["fixation_category"].isin(categories)]
    df["joint"] = df["P(m1&m2)"]
    df["marginal"] = df["P(m1)*P(m2)"]

    agg_df = _compute_median_df(df, ["monkey_pair", "fixation_category"])
    monkey_pairs = agg_df["monkey_pair"].unique()
    monkey_color_dict = _get_monkey_palette(monkey_pairs)
    violin_palette = config.violin_palette

    fig, axes = plt.subplots(1, len(categories), figsize=(12, 6), sharey=False)

    for i, category in enumerate(categories):
        ax = axes[i]
        agg_cat_df = agg_df[agg_df["fixation_category"] == category]

        melted = agg_cat_df.melt(
            id_vars=["monkey_pair"],
            value_vars=["P(m1)*P(m2)", "P(m1&m2)"],
            var_name="Probability Type",
            value_name="Probability"
        )
        _draw_single_violin(ax, melted, violin_palette)

        mp_data = {
            row["monkey_pair"]: [row["P(m1)*P(m2)"], row["P(m1&m2)"]]
            for _, row in agg_cat_df.iterrows()
        }
        _overlay_medians(ax, mp_data, monkey_color_dict)

        _annotate_ks(ax, df[df["fixation_category"] == category])
        ax.set_title(f"{category.capitalize()}")

    fig.suptitle("Fixation Probability Comparison — Overall", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _finalize_and_save(fig, config.plot_dir, "joint_fixation_probabilities_overall.pdf")


def plot_joint_vs_marginal_violin_by_interactivity(config):
    df = load_df_from_pkl(config.fix_prob_df_by_interactivity_path)
    categories = ["face", "out_of_roi"]
    df = df[df["fixation_category"].isin(categories)]
    df["joint"] = df["P(m1&m2)"]
    df["marginal"] = df["P(m1)*P(m2)"]

    agg_df = _compute_median_df(df, ["monkey_pair", "fixation_category", "interactivity"])
    interactivities = df["interactivity"].unique()
    monkey_pairs = agg_df["monkey_pair"].unique()
    monkey_color_dict = _get_monkey_palette(monkey_pairs)
    violin_palette = config.violin_palette

    fig, axs = plt.subplots(len(interactivities), len(categories), figsize=(12, 5 * len(interactivities)), sharey=False)
    if len(interactivities) == 1:
        axs = [axs]

    for i, interactivity in enumerate(interactivities):
        for j, category in enumerate(categories):
            ax = axs[i][j] if len(interactivities) > 1 else axs[0][j]
            sub_agg_df = agg_df[
                (agg_df["interactivity"] == interactivity) &
                (agg_df["fixation_category"] == category)
            ]
            melted = sub_agg_df.melt(
                id_vars=["monkey_pair"],
                value_vars=["P(m1)*P(m2)", "P(m1&m2)"],
                var_name="Probability Type",
                value_name="Probability"
            )
            _draw_single_violin(ax, melted, violin_palette)

            mp_data = {
                row["monkey_pair"]: [row["P(m1)*P(m2)"], row["P(m1&m2)"]]
                for _, row in sub_agg_df.iterrows()
            }
            _overlay_medians(ax, mp_data, monkey_color_dict)

            stat_df = df[
                (df["fixation_category"] == category) &
                (df["interactivity"] == interactivity)
            ]
            _annotate_ks(ax, stat_df)

            ax.set_title(f"{interactivity} — {category}")

    fig.suptitle("Fixation Probability Comparison — By Interactivity", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _finalize_and_save(fig, config.plot_dir, "joint_fixation_probabilities_by_interactivity.pdf")


def _get_significance_marker(p_val):
    if p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < 0.05:
        return '*'
    else:
        return 'NS'


def _compute_median_df(df, group_cols):
    return df.groupby(group_cols).agg({
        "P(m1)*P(m2)": "median",
        "P(m1&m2)": "median"
    }).reset_index()


def _get_monkey_palette(monkey_pairs):
    palette = sns.color_palette("Set2", n_colors=len(monkey_pairs))
    return {mp: palette[i] for i, mp in enumerate(monkey_pairs)}


def _draw_single_violin(ax, melted, violin_palette):
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


def _overlay_medians(ax, monkey_pair_data, monkey_color_dict):
    for mp, row in monkey_pair_data.items():
        y_vals = row
        jitter = 0.01 * (hash(mp) % 10 - 5)
        ax.plot([0 + jitter, 1 + jitter], y_vals, color=monkey_color_dict[mp], alpha=0.4)
        ax.scatter([0 + jitter], [y_vals[0]], color=monkey_color_dict[mp], s=30, alpha=0.7)
        ax.scatter([1 + jitter], [y_vals[1]], color=monkey_color_dict[mp], s=30, alpha=0.7)


def _annotate_ks(ax, group_df):
    ks_stat, p_val = ks_2samp(group_df["P(m1)*P(m2)"], group_df["P(m1&m2)"])
    ax.text(0.5, 1.05, f"KS test: {_get_significance_marker(p_val)}",
            transform=ax.transAxes, ha="center", fontsize=12)


def _finalize_and_save(fig, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    fig.patch.set_alpha(0)
    for ax in fig.get_axes():
        ax.set_facecolor("none")
    fig.savefig(save_path, format="pdf", transparent=True, bbox_inches="tight")
    plt.close(fig)
