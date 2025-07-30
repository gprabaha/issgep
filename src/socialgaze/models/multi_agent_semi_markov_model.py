import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from scipy.stats import kstest

class MultiAgentSemiMarkovAnalysis:
    """
    Advanced Semi-Markov analysis for multiple agents with comparison capabilities
    """
    
    def __init__(self):
        self.models = {}  # Store models for each agent
        
    def fit_agent_models(self, df, agents=None, distribution_type='lognorm'):
        """
        Fit separate models for each agent
        """
        if agents is None:
            agents = df['agent'].unique()
        
        for agent in agents:
            print(f"\nFitting model for agent: {agent}")
            agent_data = df[df['agent'] == agent].copy()
            
            model = SemiMarkovModel()
            model.fit(agent_data, distribution_type=distribution_type)
            self.models[agent] = model
        
        return self
    
    def compare_transition_matrices(self, figsize=(15, 5)):
        """
        Compare transition matrices across agents
        """
        n_agents = len(self.models)
        fig, axes = plt.subplots(1, n_agents, figsize=figsize)
        
        if n_agents == 1:
            axes = [axes]
        
        for i, (agent, model) in enumerate(self.models.items()):
            sns.heatmap(model.transition_matrix,
                       xticklabels=model.state_names,
                       yticklabels=model.state_names,
                       annot=True, fmt='.2f', cmap='Blues',
                       ax=axes[i])
            axes[i].set_title(f'Agent {agent}')
            if i == 0:
                axes[i].set_ylabel('From State')
            axes[i].set_xlabel('To State')
        
        plt.tight_layout()
        plt.show()
    
    def compare_dwell_times(self, figsize=(12, 8)):
        """
        Compare dwell time statistics across agents
        """
        # Collect dwell time statistics
        stats_data = []
        
        for agent, model in self.models.items():
            for state_idx, dist_info in model.dwell_distributions.items():
                state_name = model.state_names[state_idx]
                data = dist_info['data']
                
                stats_data.append({
                    'agent': agent,
                    'state': state_name,
                    'mean': np.mean(data),
                    'median': np.median(data),
                    'std': np.std(data),
                    'q25': np.percentile(data, 25),
                    'q75': np.percentile(data, 75)
                })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Mean dwell times
        pivot_mean = stats_df.pivot(index='state', columns='agent', values='mean')
        pivot_mean.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Mean Dwell Times')
        axes[0, 0].set_ylabel('Time units')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Standard deviation
        pivot_std = stats_df.pivot(index='state', columns='agent', values='std')
        pivot_std.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Dwell Time Standard Deviation')
        axes[0, 1].set_ylabel('Time units')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Median dwell times
        pivot_median = stats_df.pivot(index='state', columns='agent', values='median')
        pivot_median.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Median Dwell Times')
        axes[1, 0].set_ylabel('Time units')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Coefficient of variation
        stats_df['cv'] = stats_df['std'] / stats_df['mean']
        pivot_cv = stats_df.pivot(index='state', columns='agent', values='cv')
        pivot_cv.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Coefficient of Variation')
        axes[1, 1].set_ylabel('CV')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return stats_df
    
    def statistical_tests(self):
        """
        Perform statistical tests to compare agents
        """
        results = {}
        
        if len(self.models) != 2:
            print("Statistical tests currently support exactly 2 agents")
            return results
        
        agents = list(self.models.keys())
        model1, model2 = self.models[agents[0]], self.models[agents[1]]
        
        print(f"\n=== Statistical Comparison: {agents[0]} vs {agents[1]} ===")
        
        # Compare dwell time distributions for each state
        common_states = set(model1.state_names) & set(model2.state_names)
        
        for state in common_states:
            try:
                state_idx1 = model1.state_names.index(state)
                state_idx2 = model2.state_names.index(state)
                
                if state_idx1 in model1.dwell_distributions and state_idx2 in model2.dwell_distributions:
                    data1 = model1.dwell_distributions[state_idx1]['data']
                    data2 = model2.dwell_distributions[state_idx2]['data']
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p = stats.ks_2samp(data1, data2)
                    
                    # Mann-Whitney U test
                    mw_stat, mw_p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                    
                    results[state] = {
                        'ks_statistic': ks_stat,
                        'ks_p_value': ks_p,
                        'mw_statistic': mw_stat,
                        'mw_p_value': mw_p,
                        'significant_ks': ks_p < 0.05,
                        'significant_mw': mw_p < 0.05
                    }
                    
                    print(f"\n{state} state:")
                    print(f"  KS test: statistic={ks_stat:.3f}, p-value={ks_p:.3f}")
                    print(f"  MW test: statistic={mw_stat:.3f}, p-value={mw_p:.3f}")
                    if ks_p < 0.05:
                        print(f"  → Significant difference in dwell time distributions (KS)")
                    if mw_p < 0.05:
                        print(f"  → Significant difference in dwell time medians (MW)")
            
            except Exception as e:
                print(f"Could not test state {state}: {e}")
        
        return results
    
    def compute_sequence_metrics(self, df):
        """
        Compute additional sequence-level metrics
        """
        metrics = {}
        
        for agent in self.models.keys():
            agent_data = df[df['agent'] == agent]
            agent_metrics = {}
            
            # Group by session and run
            for (session, run), group in agent_data.groupby(['session_name', 'run_number']):
                group = group.sort_values('start')
                
                # Basic sequence metrics
                seq_length = len(group)
                total_duration = group['stop'].max() - group['start'].min()
                avg_dwell = (group['stop'] - group['start']).mean()
                
                # State transition entropy
                states = group['category'].values
                if len(states) > 1:
                    transitions = [(states[i], states[i+1]) for i in range(len(states)-1)]
                    transition_counts = Counter(transitions)
                    total_transitions = len(transitions)
                    
                    entropy = 0
                    for count in transition_counts.values():
                        p = count / total_transitions
                        entropy -= p * np.log2(p)
                else:
                    entropy = 0
                
                # State diversity (number of unique states / total states)
                diversity = len(set(states)) / len(states) if len(states) > 0 else 0
                
                agent_metrics[f"{session}_{run}"] = {
                    'sequence_length': seq_length,
                    'total_duration': total_duration,
                    'avg_dwell_time': avg_dwell,
                    'transition_entropy': entropy,
                    'state_diversity': diversity
                }
            
            metrics[agent] = agent_metrics
        
        return metrics
    
    def plot_sequence_metrics(self, df, figsize=(15, 10)):
        """
        Plot sequence-level metrics comparison
        """
        metrics = self.compute_sequence_metrics(df)
        
        # Convert to DataFrame for easier plotting
        plot_data = []
        for agent, agent_metrics in metrics.items():
            for session_run, vals in agent_metrics.items():
                row = {'agent': agent, 'session_run': session_run}
                row.update(vals)
                plot_data.append(row)
        
        metrics_df = pd.DataFrame(plot_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        metric_names = ['sequence_length', 'total_duration', 'avg_dwell_time', 
                       'transition_entropy', 'state_diversity']
        
        for i, metric in enumerate(metric_names):
            if i < 5:  # We have 5 metrics to plot
                row, col = i // 3, i % 3
                ax = axes[row, col]
                
                # Box plot for each agent
                agents = metrics_df['agent'].unique()
                data_to_plot = [metrics_df[metrics_df['agent'] == agent][metric].values 
                               for agent in agents]
                
                ax.boxplot(data_to_plot, labels=agents)
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_ylabel('Value')
        
        # Hide the last empty subplot
        axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        return metrics_df

# Usage example with your actual data structure
def analyze_fixation_data(fixation_df):
    """
    Complete analysis pipeline for your fixation data
    """
    print("Starting Semi-Markov analysis of fixation data...")
    
    # Initialize multi-agent analyzer
    analyzer = MultiAgentSemiMarkovAnalysis()
    
    # Fit models for each agent
    analyzer.fit_agent_models(fixation_df, distribution_type='lognorm')
    
    # Print summaries for each agent
    for agent, model in analyzer.models.items():
        print(f"\n{'='*20} Agent {agent} {'='*20}")
        model.get_model_summary()
    
    # Compare transition matrices
    print("\nComparing transition matrices...")
    analyzer.compare_transition_matrices()
    
    # Compare dwell time statistics
    print("\nComparing dwell time statistics...")
    dwell_stats = analyzer.compare_dwell_times()
    
    # Perform statistical tests (if 2 agents)
    if len(analyzer.models) == 2:
        test_results = analyzer.statistical_tests()
    
    # Analyze sequence-level metrics
    print("\nAnalyzing sequence-level metrics...")
    sequence_metrics = analyzer.plot_sequence_metrics(fixation_df)
    
    return analyzer, dwell_stats, sequence_metrics

# Example usage:
# analyzer, stats, metrics = analyze_fixation_data(fixation_df)