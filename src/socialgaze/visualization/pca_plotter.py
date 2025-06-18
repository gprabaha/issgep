#src/socialgaze/visualization/pca_plotter.py

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.cm import get_cmap
from collections import defaultdict

from socialgaze.utils.path_utils import get_pc_trajectory_comparison_plot_dir


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


    def plot_pc_trajectory_comparisons(self, comparison_data: dict, fit_name: str, regions: list):
        """
        Generates comparison plots for a given fit, across all transforms and regions.
        Each region gets one figure with:
            - One row per transform
            - Columns: 3D trajectory plot + bar plots for comparison metrics
        """

        save_base = get_pc_trajectory_comparison_plot_dir(
            base_dir=self.pca_config.pc_trajectory_comparison_plots_base_dir,
            fit_name=fit_name,
            dated=True
        )
        os.makedirs(save_base, exist_ok=True)

        metric_names = [
            "euclidean_distance", 
            "vector_angle_deg", 
            "trajectory_length_diff", 
            "procrustes_disparity"
        ]
        transform_names = sorted(comparison_data.keys())
        plot_var_expl = True  # toggle
        n_extra = 1 if plot_var_expl else 0
        n_cols = 1 + len(metric_names) + n_extra


        for region in regions:
            fig, axes = [], []

            # Create mixed 3D + 2D subplot grid manually
            fig = plt.figure(figsize=(self.plotting_config.plot_size[0]*n_cols, self.plotting_config.plot_size[1]*len(transform_names)))
            axes = []
            for row_idx in range(len(transform_names)):
                row_axes = []
                for col_idx in range(n_cols):
                    if col_idx == 0:
                        ax = fig.add_subplot(len(transform_names), n_cols, row_idx * n_cols + col_idx + 1, projection="3d")
                    else:
                        ax = fig.add_subplot(len(transform_names), n_cols, row_idx * n_cols + col_idx + 1)
                    row_axes.append(ax)
                axes.append(row_axes)

            for row_idx, transform_name in enumerate(transform_names):
                transform_data = comparison_data[transform_name].get(region)
                if transform_data is None:
                    continue

                # --- 3D trajectory plot ---
                ax = axes[row_idx][0]
                proj_df = transform_data["projection_df"]

                grouped = defaultdict(list)
                for _, row in proj_df.iterrows():
                    key = row["category"]
                    grouped[key].append(row)

                cmap = get_cmap("tab10")
                color_map = {cat: cmap(i % 10) for i, cat in enumerate(sorted(grouped.keys()))}

                for cat, rows in grouped.items():
                    color = color_map[cat]
                    pc1 = next(row for row in rows if row["pc_dimension"] == 0)["pc_timeseries"]
                    pc2 = next(row for row in rows if row["pc_dimension"] == 1)["pc_timeseries"]
                    pc3 = next(row for row in rows if row["pc_dimension"] == 2)["pc_timeseries"]
                    ax.plot(pc1, pc2, pc3, color=color, alpha=0.6, label=cat)

                ax.set_title(f"{transform_name} | {region}")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_zlabel("PC3")
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

                # --- Metric bar plots ---
                comparison_metrics = transform_data["comparison_metrics"]
                for col_idx, metric_name in enumerate(metric_names, start=1):
                    ax_metric = axes[row_idx][col_idx]
                    labels = [f"{m['category_1']} vs {m['category_2']}" for m in comparison_metrics]
                    values = [m[metric_name] for m in comparison_metrics]
                    ax_metric.bar(range(len(values)), values, tick_label=labels)
                    ax_metric.set_title(metric_name.replace("_", " ").title())
                    ax_metric.tick_params(axis='x', labelrotation=45)

                # --- Variance explained per PC (per category) ---
                if plot_var_expl:
                    ax_var = axes[row_idx][-1]
                    meta = transform_data["projection_meta"]
                    var_expl_key = f"{region}_category_pc_var_explained"
                    if var_expl_key in meta:
                        per_cat_pc_var = meta[var_expl_key]  # dict: cat -> list of floats
                        cats = sorted(per_cat_pc_var.keys())
                        n_pcs = len(next(iter(per_cat_pc_var.values())))

                        width = 0.8 / len(cats)
                        x = np.arange(n_pcs)

                        for i, cat in enumerate(cats):
                            ax_var.bar(
                                x + i * width, 
                                per_cat_pc_var[cat], 
                                width=width, 
                                label=cat, 
                                alpha=0.7
                            )

                        ax_var.set_title("Variance Explained per PC")
                        ax_var.set_xlabel("PC")
                        ax_var.set_ylabel("Explained Variance")
                        ax_var.set_xticks(x + width * (len(cats)-1)/2)
                        ax_var.set_xticklabels([f"PC{i+1}" for i in range(n_pcs)], rotation=45)
                        ax_var.set_ylim(0, 1)
                        ax_var.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))


            fig.tight_layout()
            save_path = os.path.join(save_base, f"{region}_comparison.png")
            fig.savefig(save_path, dpi=self.plotting_config.plot_dpi)
            plt.close(fig)


