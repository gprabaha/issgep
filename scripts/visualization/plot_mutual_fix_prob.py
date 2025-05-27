# scripts/visualization/plot_fixation_probabilities.py

from socialgaze.config.base_config import BaseConfig
from socialgaze.config.fix_prob_config import FixProbConfig
from socialgaze.visualization.fixation_probability_plotter import (
    plot_joint_vs_marginal_violin,
    plot_joint_vs_marginal_violin_by_interactivity
)


def main():
    base_config = BaseConfig()
    fix_prob_config = FixProbConfig()

    # Plot overall (not separated by interactivity)
    plot_joint_vs_marginal_violin(fix_prob_config)

    # Plot separated by interactivity
    plot_joint_vs_marginal_violin_by_interactivity(fix_prob_config)


if __name__ == "__main__":
    main()
