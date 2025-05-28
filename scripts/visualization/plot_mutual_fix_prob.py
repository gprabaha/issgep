# scripts/visualization/plot_fixation_probabilities.py

from socialgaze.config.fix_prob_config import FixProbConfig
from socialgaze.features.fix_prob_detector import FixProbDetector
from socialgaze.visualization.fixation_probability_plotter import plot_joint_vs_marginal_violin


def main():
    fix_prob_config = FixProbConfig()
    fix_prob_detector = FixProbDetector(fixation_detector=None, config=fix_prob_config)

    # Plot all modes
    for mode in fix_prob_config.modes:
        print(f"\n=== Plotting fixation probability violins for mode = '{mode}' ===")
        plot_joint_vs_marginal_violin(config=fix_prob_config, detector=fix_prob_detector, mode=mode)


if __name__ == "__main__":
    main()
