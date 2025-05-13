import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import pandas as pd
from socialgaze.utils.path_utils import get_pc_plot_path
from matplotlib.cm import get_cmap


class ThreeDPlotter:
    def __init__(self, config):
        self.config = config

    def plot_pc_trajectories_all_trials(self, proj_df: pd.DataFrame, fit_name: str, transform_name: str, region: str):
        output_dir = get_pc_plot_path(self.config.pc_plot_base_dir, fit_name, transform_name, region)
        os.makedirs(output_dir, exist_ok=True)

        # === Group by (category, interactivity)
        has_inter = "is_interactive" in proj_df.columns and proj_df["is_interactive"].notna().any()
        has_trial = "trial_index" in proj_df.columns and proj_df["trial_index"].notna().any()

        grouped = defaultdict(list)
        for _, row in proj_df.iterrows():
            key = (row["category"], row.get("is_interactive", None))
            grouped[key].append(row)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        cmap = get_cmap("tab10")
        color_map = {}

        for idx, ((cat, inter), rows) in enumerate(grouped.items()):
            color = cmap(idx % 10)
            color_map[(cat, inter)] = color
            label = f"{cat}" + (f" | {inter}" if inter is not None else "")
            added_label = False

            # group by trial_index if present
            trial_groups = defaultdict(list)
            for row in rows:
                trial_groups[row.get("trial_index", None)].append(row)

            for trial, trial_rows in trial_groups.items():
                pc1 = next(row for row in trial_rows if row["pc_dimension"] == 0)["pc_timeseries"]
                pc2 = next(row for row in trial_rows if row["pc_dimension"] == 1)["pc_timeseries"]
                pc3 = next(row for row in trial_rows if row["pc_dimension"] == 2)["pc_timeseries"]

                ax.plot(pc1, pc2, pc3, color=color, alpha=0.5, label=label if not added_label else "")
                added_label = True

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"{region} | {fit_name} → {transform_name}")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        fig.tight_layout()

        fname = f"{region}_pc123.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=self.config.plot_dpi)
        plt.close(fig)
