#src/socialgaze/visualization/pca_plotter.py


import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.cm import get_cmap
from collections import defaultdict



class PCAPlotter:
    def __init__(self, plotting_config, pca_config):
        self.plotting_config = plotting_config
        self.pca_config = pca_config


    def plot_pc_trajectories_all_trials(self, proj_df, fit_name: str, transform_name: str, region: str):
        """
        Plots all trials across all (category, interactivity) combinations in a single 3D plot.
        Trials in the same (cat, inter) group share color and legend label.
        """

        grouped = defaultdict(list)
        for _, row in proj_df.iterrows():
            key = (
                row["category"],
                row.get("is_interactive", None),
                row.get("trial_index", None),
            )
            grouped[key].append(row)

        # Assign color per (cat, inter)
        color_map = {}
        label_flags = {}
        cmap = get_cmap("tab10")
        cat_inter_keys = sorted(set((k[0], k[1]) for k in grouped.keys()))
        for i, key in enumerate(cat_inter_keys):
            color_map[key] = cmap(i % 10)
            label_flags[key] = False

        # Plot
        fig = plt.figure(figsize=self.plotting_config.plot_size, dpi=self.plotting_config.plot_dpi)
        ax = fig.add_subplot(111, projection="3d")

        for (cat, inter, trial), rows in grouped.items():
            color = color_map[(cat, inter)]
            label = f"{cat}" + (f" | {inter}" if inter is not None else "")
            show_label = not label_flags[(cat, inter)]

            pc1 = next(row for row in rows if row["pc_dimension"] == 0)["pc_timeseries"]
            pc2 = next(row for row in rows if row["pc_dimension"] == 1)["pc_timeseries"]
            pc3 = next(row for row in rows if row["pc_dimension"] == 2)["pc_timeseries"]

            ax.plot(pc1, pc2, pc3, color=color, alpha=0.5, label=label if show_label else None)
            label_flags[(cat, inter)] = True

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"Region: {region} | Fit: {fit_name} → {transform_name}")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        fig.tight_layout()
        
        save_path = self.pca_config.get_static_pc_plot_path(fit_name, transform_name, region, self.plotting_config.plot_file_format)
        fig.savefig(save_path, dpi=self.plotting_config.plot_dpi)
        plt.close(fig)
        return save_path



    def animate_pc_trajectories_3d(
        self,
        proj_df: pd.DataFrame,
        fit_name: str,
        transform_name: str,
        region: str,
        rotation_speed: int = 2,
        n_frames: int = 180,
    ):
        """
        Generates an MP4 animation rotating the 3D PCA trajectories over time.

        Args:
            proj_df: The projection dataframe.
            fit_name: Name of PCA fit.
            transform_name: Name of projection transform.
            region: Brain region name.
            rotation_speed: Degrees per frame (default = 2).
            n_frames: Total number of frames (default = 180 for full rotation).
        """

        # === Group trials
        grouped = defaultdict(list)
        for _, row in proj_df.iterrows():
            key = (row["category"], row.get("is_interactive", None), row.get("trial_index", None))
            grouped[key].append(row)

        # === Assign color per (cat, inter)
        color_map = {}
        cmap = get_cmap("tab10")
        cat_inter_keys = sorted(set((k[0], k[1]) for k in grouped.keys()))
        for i, key in enumerate(cat_inter_keys):
            color_map[key] = cmap(i % 10)

        # === Create figure
        fig = plt.figure(figsize=self.plotting_config.plot_size, dpi=self.plotting_config.plot_dpi)
        ax = fig.add_subplot(111, projection="3d")
        lines = []

        for (cat, inter, trial), rows in grouped.items():
            color = color_map[(cat, inter)]
            pc1 = next(row for row in rows if row["pc_dimension"] == 0)["pc_timeseries"]
            pc2 = next(row for row in rows if row["pc_dimension"] == 1)["pc_timeseries"]
            pc3 = next(row for row in rows if row["pc_dimension"] == 2)["pc_timeseries"]

            line, = ax.plot(pc1, pc2, pc3, color=color, alpha=0.5)
            lines.append(line)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"{region} | {fit_name} → {transform_name}")

        def update(frame):
            ax.view_init(elev=30, azim=frame * rotation_speed)
            return lines

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=50,
            blit=False
        )

        save_path = self.pca_config.get_rotation_pc_plot_path(fit_name, transform_name, region)
        ani.save(save_path, fps=24, dpi=self.plotting_config.plot_dpi)
        plt.close(fig)
        return save_path


