import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from socialgaze.utils.loading_utils import load_df_from_pkl

def plot_joint_vs_marginal_violin(config):
    """Plots violin plots comparing P(m1&m2) vs P(m1)*P(m2) for 'face' and 'out_of_roi'."""
    df = load_df_from_pkl(config.fix_prob_df_path)

    relevant_cats = ["face", "out_of_roi"]
    df = df[df["fixation_category"].isin(relevant_cats)]

    df["joint"] = df["P(m1&m2)"]
    df["marginal"] = df["P(m1)*P(m2)"]

    df_melted = pd.melt(
        df,
        id_vars=["session_name", "run_number", "fixation_category"],
        value_vars=["joint", "marginal"],
        var_name="ProbabilityType",
        value_name="Probability"
    )

    for category in relevant_cats:
        plot_df = df_melted[df_melted["fixation_category"] == category]

        plt.figure(figsize=(8, 6))
        sns.violinplot(data=plot_df, x="ProbabilityType", y="Probability")
        plt.title(f"Overall — {category}")
        plt.ylabel("Probability")
        plt.xlabel("")

        save_dir = os.path.join(config.plot_dir, "violinplots_overall")
        os.makedirs(save_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"violin_{category}.png"))
        plt.close()


def plot_joint_vs_marginal_violin_by_interactivity(config):
    """Plots violin plots comparing P(m1&m2) vs P(m1)*P(m2) split by interactivity."""
    df = load_df_from_pkl(config.fix_prob_df_by_interactivity_path)

    relevant_cats = ["face", "out_of_roi"]
    df = df[df["fixation_category"].isin(relevant_cats)]

    df["joint"] = df["P(m1&m2)"]
    df["marginal"] = df["P(m1)*P(m2)"]

    df_melted = pd.melt(
        df,
        id_vars=["session_name", "run_number", "fixation_category", "interactivity"],
        value_vars=["joint", "marginal"],
        var_name="ProbabilityType",
        value_name="Probability"
    )

    for category in relevant_cats:
        plot_df = df_melted[df_melted["fixation_category"] == category]

        plt.figure(figsize=(10, 6))
        sns.violinplot(
            data=plot_df,
            x="interactivity",
            y="Probability",
            hue="ProbabilityType",
            split=True
        )
        plt.title(f"By Interactivity — {category}")
        plt.ylabel("Probability")
        plt.xlabel("Interactivity")
        plt.legend(title="", loc="upper right")

        save_dir = os.path.join(config.plot_dir, "violinplots_by_interactivity")
        os.makedirs(save_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"violin_{category}.png"))
        plt.close()
