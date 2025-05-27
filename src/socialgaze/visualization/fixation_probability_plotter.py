import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import ks_2samp
from socialgaze.utils.loading_utils import load_df_from_pkl


def _get_significance_marker(p_val):
    if p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < 0.05:
        return '*'
    else:
        return 'NS'


def plot_joint_vs_marginal_violin(config):
    df = load_df_from_pkl(config.fix_prob_df_path)
    categories = ["face", "out_of_roi"]
    df = df[df["fixation_category"].isin(categories)]

    df["joint"] = df["P(m1&m2)"]
    df["marginal"] = df["P(m1)*P(m2)"]

    # Compute aggregated medians per monkey pair per category
    agg_df = df.groupby(["monkey_pair", "fixation_category"]).agg({
        "P(m1)*P(m2)": "median",
        "P(m1&m2)": "median"
    }).reset_index()

    # Monkey pair color map
    monkey_pairs = agg_df["monkey_pair"].unique()
    monkey_palette = sns.color_palette("Set2", n_colors=len(monkey_pairs))
    monkey_color_dict = {mp: monkey_palette[i] for i, mp in enumerate(monkey_pairs)}

    violin_palette = {"P(m1)*P(m2)": "#66c2a5", "P(m1&m2)": "#8da0cb"}

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


        # Overlay individual medians per monkey pair
        for mp in monkey_pairs:
            mp_row = agg_cat_df[agg_cat_df["monkey_pair"] == mp]
            if mp_row.empty:
                continue
            x_vals = [0, 1]
            y_vals = [mp_row["P(m1)*P(m2)"].values[0], mp_row["P(m1&m2)"].values[0]]
            jitter = 0.01 * (hash(mp) % 10 - 5)
            ax.plot([0 + jitter, 1 + jitter], y_vals, color=monkey_color_dict[mp], alpha=0.4)
            ax.scatter([0 + jitter], [y_vals[0]], color=monkey_color_dict[mp], s=30, alpha=0.7)
            ax.scatter([1 + jitter], [y_vals[1]], color=monkey_color_dict[mp], s=30, alpha=0.7)

        # Run-level KS test
        full_cat_df = df[df["fixation_category"] == category]
        ks_stat, p_val = ks_2samp(full_cat_df["P(m1)*P(m2)"], full_cat_df["P(m1&m2)"])
        marker = _get_significance_marker(p_val)
        ax.text(0.5, 1.05, f"KS test: {marker}", transform=ax.transAxes,
                ha="center", fontsize=12)

        ax.set_title(f"{category.capitalize()}")

    fig.suptitle("Fixation Probability Comparison — Overall", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(config.plot_dir, "joint_fixation_probabilities_overall.pdf")
    fig.patch.set_alpha(0)
    for ax in fig.get_axes():
        ax.set_facecolor("none")
    fig.savefig(save_path, format="pdf", transparent=True, bbox_inches="tight")
    plt.close(fig)




def plot_joint_vs_marginal_violin_by_interactivity(config):
    df = load_df_from_pkl(config.fix_prob_df_by_interactivity_path)
    categories = ["face", "out_of_roi"]
    df = df[df["fixation_category"].isin(categories)]

    df["joint"] = df["P(m1&m2)"]
    df["marginal"] = df["P(m1)*P(m2)"]

    agg_df = df.groupby(["monkey_pair", "fixation_category", "interactivity"]).agg({
        "P(m1)*P(m2)": "median",
        "P(m1&m2)": "median"
    }).reset_index()

    interactivities = df["interactivity"].unique()
    monkey_pairs = agg_df["monkey_pair"].unique()
    monkey_palette = sns.color_palette("Set2", n_colors=len(monkey_pairs))
    monkey_color_dict = {mp: monkey_palette[i] for i, mp in enumerate(monkey_pairs)}
    violin_palette = {"P(m1)*P(m2)": "#66c2a5", "P(m1&m2)": "#8da0cb"}

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

            for mp in monkey_pairs:
                row = sub_agg_df[sub_agg_df["monkey_pair"] == mp]
                if row.empty:
                    continue
                x_vals = [0, 1]
                y_vals = [row["P(m1)*P(m2)"].values[0], row["P(m1&m2)"].values[0]]
                jitter = 0.01 * (hash(mp) % 10 - 5)
                ax.plot([0 + jitter, 1 + jitter], y_vals, color=monkey_color_dict[mp], alpha=0.4)
                ax.scatter([0 + jitter], [y_vals[0]], color=monkey_color_dict[mp], s=30, alpha=0.7)
                ax.scatter([1 + jitter], [y_vals[1]], color=monkey_color_dict[mp], s=30, alpha=0.7)

            stat_df = df[
                (df["fixation_category"] == category) &
                (df["interactivity"] == interactivity)
            ]
            ks_stat, p_val = ks_2samp(stat_df["P(m1)*P(m2)"], stat_df["P(m1&m2)"])
            marker = _get_significance_marker(p_val)
            ax.text(0.5, 1.05, f"KS test: {marker}", transform=ax.transAxes,
                    ha="center", fontsize=12)

            ax.set_title(f"{interactivity} — {category}")

    fig.suptitle("Fixation Probability Comparison — By Interactivity", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(config.plot_dir, "joint_fixation_probabilities_by_interactivity.pdf")
    fig.patch.set_alpha(0)
    for ax in fig.get_axes():
        ax.set_facecolor("none")
    fig.savefig(save_path, format="pdf", transparent=True, bbox_inches="tight")
    plt.close(fig)
