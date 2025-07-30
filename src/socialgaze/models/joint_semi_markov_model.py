import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class SemiMarkovModel:
    """Base Semi-Markov Model class"""
    
    def __init__(self, model_name="SemiMarkov"):
        self.states = None
        self.transition_matrix = None
        self.dwell_distributions = {}
        self.initial_state_probs = None
        self.label_encoder = LabelEncoder()
        self.state_names = None
        self.model_name = model_name
        self.log_likelihood = None
        self.n_parameters = 0
        self.n_observations = 0
        
    def prepare_sequences(self, df, agent=None):
        """Convert fixation dataframe to sequences of states and dwell times"""
        sequences = []
        dwell_times = []
        
        if agent is not None:
            df = df[df['agent'] == agent].copy()
        
        # Group by session and run to get individual sequences
        for (session, run), group in df.groupby(['session_name', 'run_number']):
            if agent is None:
                # For joint models, we need to ensure we have data from both agents
                agents_in_group = group['agent'].unique()
                if len(agents_in_group) < 2:
                    continue
            
            group = group.sort_values('start')
            
            # Extract state sequence and dwell times
            states = group['category'].values if agent is not None else group['joint_state'].values
            durations = (group['stop'] - group['start']).values
            
            if len(states) > 1:  # Need at least 2 states for transitions
                sequences.append(states)
                dwell_times.append(durations)
        
        return sequences, dwell_times
    
    def fit(self, df, agent=None, distribution_type='lognorm'):
        """Fit the semi-Markov model"""
        sequences, dwell_times = self.prepare_sequences(df, agent)
        
        if len(sequences) == 0:
            raise ValueError("No valid sequences found for fitting")
        
        # Get unique states
        all_states = []
        for seq in sequences:
            all_states.extend(seq)
        
        self.states = sorted(list(set(all_states)))
        self.state_names = self.states.copy()
        n_states = len(self.states)
        
        # Encode states as integers
        self.label_encoder.fit(self.states)
        
        # Initialize transition matrix
        self.transition_matrix = np.zeros((n_states, n_states))
        
        # Count transitions and collect dwell times by state
        state_dwell_times = defaultdict(list)
        initial_states = []
        total_transitions = 0
        
        for seq, dwells in zip(sequences, dwell_times):
            if len(seq) == 0:
                continue
                
            # Record initial state
            initial_states.append(seq[0])
            
            # Process each state in the sequence
            for i, (state, dwell_time) in enumerate(zip(seq, dwells)):
                state_idx = self.label_encoder.transform([state])[0]
                state_dwell_times[state_idx].append(dwell_time)
                
                # Count transition to next state
                if i < len(seq) - 1:
                    next_state = seq[i + 1]
                    next_state_idx = self.label_encoder.transform([next_state])[0]
                    self.transition_matrix[state_idx, next_state_idx] += 1
                    total_transitions += 1
        
        # Normalize transition matrix
        row_sums = self.transition_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        self.transition_matrix = self.transition_matrix / row_sums[:, np.newaxis]
        
        # Calculate initial state probabilities
        initial_state_counts = Counter(initial_states)
        self.initial_state_probs = np.zeros(n_states)
        for state, count in initial_state_counts.items():
            state_idx = self.label_encoder.transform([state])[0]
            self.initial_state_probs[state_idx] = count / len(initial_states)
        
        # Fit dwell time distributions for each state
        total_log_likelihood = 0
        n_params = 0
        n_obs = 0
        
        for state_idx in range(n_states):
            if len(state_dwell_times[state_idx]) > 0:
                dwell_data = np.array(state_dwell_times[state_idx])
                n_obs += len(dwell_data)
                
                # Fit specified distribution
                if distribution_type == 'lognorm':
                    params = stats.lognorm.fit(dwell_data)
                    ll = np.sum(stats.lognorm.logpdf(dwell_data, *params))
                    n_params += 3  # shape, location, scale
                elif distribution_type == 'gamma':
                    params = stats.gamma.fit(dwell_data)
                    ll = np.sum(stats.gamma.logpdf(dwell_data, *params))
                    n_params += 3  # shape, location, scale
                elif distribution_type == 'expon':
                    params = stats.expon.fit(dwell_data)
                    ll = np.sum(stats.expon.logpdf(dwell_data, *params))
                    n_params += 2  # location, scale
                else:
                    raise ValueError(f"Unsupported distribution: {distribution_type}")
                
                total_log_likelihood += ll
                
                self.dwell_distributions[state_idx] = {
                    'type': distribution_type,
                    'params': params,
                    'data': dwell_data,
                    'log_likelihood': ll
                }
        
        # Add parameters for transition matrix (non-zero entries)
        n_params += np.sum(self.transition_matrix > 0) - n_states  # subtract diagonal constraint
        
        # Add parameters for initial state probabilities
        n_params += n_states - 1  # sum to 1 constraint
        
        self.log_likelihood = total_log_likelihood
        self.n_parameters = n_params
        self.n_observations = n_obs
        
        return self
    
    def get_aic(self):
        """Calculate Akaike Information Criterion"""
        if self.log_likelihood is None:
            return None
        return 2 * self.n_parameters - 2 * self.log_likelihood
    
    def get_bic(self):
        """Calculate Bayesian Information Criterion"""
        if self.log_likelihood is None:
            return None
        return np.log(self.n_observations) * self.n_parameters - 2 * self.log_likelihood
    
    def get_model_summary(self):
        """Print model summary"""
        print(f"=== {self.model_name} Model Summary ===")
        print(f"Number of states: {len(self.states)}")
        print(f"States: {self.state_names}")
        print(f"Log-likelihood: {self.log_likelihood:.2f}")
        print(f"Number of parameters: {self.n_parameters}")
        print(f"AIC: {self.get_aic():.2f}")
        print(f"BIC: {self.get_bic():.2f}")
        print(f"Number of observations: {self.n_observations}")


class JointSemiMarkovAnalysis:
    """Analysis framework for comparing individual vs joint models"""
    
    def __init__(self):
        self.individual_models = {}
        self.joint_model = None
        self.joint_data = None
        
    def create_joint_states(self, df):
        """Create joint state representations from M1 and M2 data"""
        print("Creating joint state sequences...")
        
        # Get all unique sessions and runs
        sessions_runs = df[['session_name', 'run_number']].drop_duplicates()
        
        joint_sequences = []
        
        for _, (session, run) in sessions_runs.iterrows():
            # Get data for this session/run
            session_data = df[(df['session_name'] == session) & 
                            (df['run_number'] == run)].copy()
            
            # Check if we have both agents
            agents = session_data['agent'].unique()
            if 'm1' not in agents or 'm2' not in agents:
                continue
            
            m1_data = session_data[session_data['agent'] == 'm1'].sort_values('start')
            m2_data = session_data[session_data['agent'] == 'm2'].sort_values('start')
            
            # Create synchronized joint states using time windows
            joint_seq = self._synchronize_sequences(m1_data, m2_data, session, run)
            if len(joint_seq) > 0:
                joint_sequences.append(joint_seq)
        
        # Combine all joint sequences
        self.joint_data = pd.concat(joint_sequences, ignore_index=True)
        return self.joint_data
    
    def _synchronize_sequences(self, m1_data, m2_data, session, run, time_window=100):
        """
        Synchronize M1 and M2 sequences by creating joint states based on temporal overlap
        """
        joint_sequence = []
        
        # Get all time points where either agent starts or stops a fixation
        all_times = set()
        for _, row in m1_data.iterrows():
            all_times.add(row['start'])
            all_times.add(row['stop'])
        for _, row in m2_data.iterrows():
            all_times.add(row['start'])
            all_times.add(row['stop'])
        
        all_times = sorted(list(all_times))
        
        # Create time segments and determine joint states
        for i in range(len(all_times) - 1):
            segment_start = all_times[i]
            segment_end = all_times[i + 1]
            segment_mid = (segment_start + segment_end) / 2
            
            # Find what M1 and M2 are doing during this segment
            m1_state = self._get_state_at_time(m1_data, segment_mid)
            m2_state = self._get_state_at_time(m2_data, segment_mid)
            
            if m1_state is not None and m2_state is not None:
                joint_state = f"{m1_state}|{m2_state}"
                duration = segment_end - segment_start
                
                if duration > 0:  # Only include non-zero duration segments
                    joint_sequence.append({
                        'session_name': session,
                        'run_number': run,
                        'agent': 'joint',
                        'start': segment_start,
                        'stop': segment_end,
                        'joint_state': joint_state,
                        'm1_state': m1_state,
                        'm2_state': m2_state,
                        'category': joint_state  # For compatibility with existing code
                    })
        
        return pd.DataFrame(joint_sequence)
    
    def _get_state_at_time(self, agent_data, time_point):
        """Get the state of an agent at a specific time point"""
        for _, row in agent_data.iterrows():
            if row['start'] <= time_point <= row['stop']:
                return row['category']
        return None
    
    def fit_individual_models(self, df, distribution_type='lognorm'):
        """Fit individual models for M1 and M2"""
        print("Fitting individual models...")
        
        for agent in ['m1', 'm2']:
            print(f"  Fitting model for {agent}...")
            model = SemiMarkovModel(model_name=f"Individual-{agent.upper()}")
            model.fit(df, agent=agent, distribution_type=distribution_type)
            self.individual_models[agent] = model
    
    def fit_joint_model(self, df, distribution_type='lognorm'):
        """Fit joint model using synchronized data"""
        print("Fitting joint model...")
        
        # Create joint state data
        joint_data = self.create_joint_states(df)
        
        # Fit joint model
        self.joint_model = SemiMarkovModel(model_name="Joint")
        self.joint_model.fit(joint_data, agent=None, distribution_type=distribution_type)
    
    def compare_models(self):
        """Compare individual vs joint models using information criteria"""
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        # Individual model results
        m1_model = self.individual_models['m1']
        m2_model = self.individual_models['m2']
        
        print("\nIndividual Models:")
        print("-" * 30)
        m1_model.get_model_summary()
        print()
        m2_model.get_model_summary()
        
        # Combined individual model metrics
        combined_ll = m1_model.log_likelihood + m2_model.log_likelihood
        combined_params = m1_model.n_parameters + m2_model.n_parameters
        combined_obs = m1_model.n_observations + m2_model.n_observations
        combined_aic = 2 * combined_params - 2 * combined_ll
        combined_bic = np.log(combined_obs) * combined_params - 2 * combined_ll
        
        print(f"\nCombined Individual Models:")
        print(f"Combined Log-likelihood: {combined_ll:.2f}")
        print(f"Combined Parameters: {combined_params}")
        print(f"Combined AIC: {combined_aic:.2f}")
        print(f"Combined BIC: {combined_bic:.2f}")
        
        # Joint model results
        print(f"\nJoint Model:")
        print("-" * 30)
        self.joint_model.get_model_summary()
        
        # Model comparison
        print(f"\n" + "="*40)
        print("COMPARISON SUMMARY")
        print("="*40)
        
        aic_improvement = combined_aic - self.joint_model.get_aic()
        bic_improvement = combined_bic - self.joint_model.get_bic()
        ll_improvement = self.joint_model.log_likelihood - combined_ll
        
        print(f"Log-likelihood improvement (Joint vs Combined): {ll_improvement:.2f}")
        print(f"AIC improvement (Joint vs Combined): {aic_improvement:.2f}")
        print(f"BIC improvement (Joint vs Combined): {bic_improvement:.2f}")
        
        # Interpretation
        print(f"\nInterpretation:")
        if aic_improvement > 0:
            print(f"✓ Joint model is BETTER by AIC (Δ = {aic_improvement:.2f})")
        else:
            print(f"✗ Individual models are BETTER by AIC (Δ = {-aic_improvement:.2f})")
            
        if bic_improvement > 0:
            print(f"✓ Joint model is BETTER by BIC (Δ = {bic_improvement:.2f})")
        else:
            print(f"✗ Individual models are BETTER by BIC (Δ = {-bic_improvement:.2f})")
        
        # Statistical significance test (likelihood ratio test)
        if ll_improvement > 0:
            # Degrees of freedom = difference in number of parameters
            df_diff = self.joint_model.n_parameters - combined_params
            if df_diff > 0:
                lr_statistic = 2 * ll_improvement
                p_value = 1 - stats.chi2.cdf(lr_statistic, df_diff)
                print(f"\nLikelihood Ratio Test:")
                print(f"LR statistic: {lr_statistic:.2f}")
                print(f"Degrees of freedom: {df_diff}")
                print(f"P-value: {p_value:.4f}")
                
                if p_value < 0.05:
                    print(f"✓ Joint model is SIGNIFICANTLY better (p < 0.05)")
                else:
                    print(f"✗ No significant improvement (p ≥ 0.05)")
        
        return {
            'individual_m1': {
                'log_likelihood': m1_model.log_likelihood,
                'n_parameters': m1_model.n_parameters,
                'aic': m1_model.get_aic(),
                'bic': m1_model.get_bic()
            },
            'individual_m2': {
                'log_likelihood': m2_model.log_likelihood,
                'n_parameters': m2_model.n_parameters,
                'aic': m2_model.get_aic(),
                'bic': m2_model.get_bic()
            },
            'combined_individual': {
                'log_likelihood': combined_ll,
                'n_parameters': combined_params,
                'aic': combined_aic,
                'bic': combined_bic
            },
            'joint': {
                'log_likelihood': self.joint_model.log_likelihood,
                'n_parameters': self.joint_model.n_parameters,
                'aic': self.joint_model.get_aic(),
                'bic': self.joint_model.get_bic()
            },
            'improvements': {
                'log_likelihood': ll_improvement,
                'aic': aic_improvement,
                'bic': bic_improvement
            }
        }
    
    def plot_joint_states_analysis(self, figsize=(15, 10)):
        """Visualize joint state patterns"""
        if self.joint_data is None:
            print("No joint data available. Run fit_joint_model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Joint state frequency
        joint_state_counts = self.joint_data['joint_state'].value_counts()
        ax1 = axes[0, 0]
        joint_state_counts.head(15).plot(kind='bar', ax=ax1)
        ax1.set_title('Most Common Joint States')
        ax1.set_xlabel('Joint State (M1|M2)')
        ax1.set_ylabel('Frequency')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Transition matrix heatmap for joint model
        ax2 = axes[0, 1]
        # Show only top states for readability
        top_states = joint_state_counts.head(10).index
        state_indices = [self.joint_model.state_names.index(state) for state in top_states if state in self.joint_model.state_names]
        
        if len(state_indices) > 1:
            sub_matrix = self.joint_model.transition_matrix[np.ix_(state_indices, state_indices)]
            sns.heatmap(sub_matrix, 
                       xticklabels=[self.joint_model.state_names[i] for i in state_indices],
                       yticklabels=[self.joint_model.state_names[i] for i in state_indices],
                       annot=True, fmt='.2f', cmap='Blues', ax=ax2)
            ax2.set_title('Joint State Transitions (Top 10 States)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.tick_params(axis='y', rotation=0)
        
        # 3. Individual vs joint state diversity
        ax3 = axes[1, 0]
        m1_states = len(self.individual_models['m1'].states)
        m2_states = len(self.individual_models['m2'].states)
        joint_states = len(self.joint_model.states)
        
        ax3.bar(['M1 Individual', 'M2 Individual', 'Joint'], 
                [m1_states, m2_states, joint_states])
        ax3.set_title('Number of States in Each Model')
        ax3.set_ylabel('Number of States')
        
        # 4. Model comparison metrics
        ax4 = axes[1, 1]
        comparison_results = self.compare_models()
        
        models = ['M1', 'M2', 'Combined\nIndividual', 'Joint']
        aics = [comparison_results['individual_m1']['aic'],
                comparison_results['individual_m2']['aic'],
                comparison_results['combined_individual']['aic'],
                comparison_results['joint']['aic']]
        
        bars = ax4.bar(models, aics)
        ax4.set_title('Model Comparison (AIC)')
        ax4.set_ylabel('AIC (lower = better)')
        
        # Highlight the best model
        best_idx = np.argmin(aics)
        bars[best_idx].set_color('red')
        
        plt.tight_layout()
        plt.show()

# Main analysis function
def run_joint_vs_individual_analysis(fixation_df, distribution_type='lognorm'):
    """
    Complete analysis comparing individual vs joint semi-Markov models
    
    Parameters:
    fixation_df: DataFrame with your fixation data
    distribution_type: 'lognorm', 'gamma', or 'expon'
    
    Returns:
    analyzer: JointSemiMarkovAnalysis object with fitted models
    results: Dictionary with comparison results
    """
    
    print("Starting Joint vs Individual Semi-Markov Analysis")
    print("="*60)
    
    # Initialize analyzer
    analyzer = JointSemiMarkovAnalysis()
    
    # Fit individual models
    analyzer.fit_individual_models(fixation_df, distribution_type=distribution_type)
    
    # Fit joint model
    analyzer.fit_joint_model(fixation_df, distribution_type=distribution_type)
    
    # Compare models
    results = analyzer.compare_models()
    
    # Create visualizations
    analyzer.plot_joint_states_analysis()
    
    return analyzer, results

# Example usage:
# analyzer, results = run_joint_vs_individual_analysis(fixation_df)