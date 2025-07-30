import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

class SemiMarkovModel:
    """
    Semi-Markov Model for eye fixation data with state transitions and dwell time distributions
    """
    
    def __init__(self):
        self.states = None
        self.transition_matrix = None
        self.dwell_distributions = {}
        self.initial_state_probs = None
        self.label_encoder = LabelEncoder()
        self.state_names = None
        
    def prepare_sequences(self, df):
        """
        Convert fixation dataframe to sequences of states and dwell times
        """
        sequences = []
        dwell_times = []
        
        # Group by session and run to get individual sequences
        for (session, run, agent), group in df.groupby(['session_name', 'run_number', 'agent']):
            group = group.sort_values('start')
            
            # Extract state sequence and dwell times
            states = group['category'].values
            durations = (group['stop'] - group['start']).values
            
            sequences.append(states)
            dwell_times.append(durations)
            
        return sequences, dwell_times
    
    def fit(self, df, distribution_type='lognorm'):
        """
        Fit the semi-Markov model to the data
        
        Parameters:
        df: DataFrame with fixation data
        distribution_type: Type of distribution to fit for dwell times ('lognorm', 'gamma', 'expon')
        """
        print("Preparing sequences...")
        sequences, dwell_times = self.prepare_sequences(df)
        
        # Get unique states
        all_states = []
        for seq in sequences:
            all_states.extend(seq)
        
        self.states = sorted(list(set(all_states)))
        self.state_names = self.states.copy()
        n_states = len(self.states)
        
        # Encode states as integers
        self.label_encoder.fit(self.states)
        
        print(f"Found {n_states} states: {self.states}")
        
        # Initialize transition matrix
        self.transition_matrix = np.zeros((n_states, n_states))
        
        # Count transitions and collect dwell times by state
        state_dwell_times = defaultdict(list)
        initial_states = []
        
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
        
        # Normalize transition matrix
        row_sums = self.transition_matrix.sum(axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        self.transition_matrix = self.transition_matrix / row_sums[:, np.newaxis]
        
        # Calculate initial state probabilities
        initial_state_counts = Counter(initial_states)
        self.initial_state_probs = np.zeros(n_states)
        for state, count in initial_state_counts.items():
            state_idx = self.label_encoder.transform([state])[0]
            self.initial_state_probs[state_idx] = count / len(initial_states)
        
        # Fit dwell time distributions for each state
        print("Fitting dwell time distributions...")
        for state_idx in range(n_states):
            if len(state_dwell_times[state_idx]) > 0:
                dwell_data = np.array(state_dwell_times[state_idx])
                
                # Fit specified distribution
                if distribution_type == 'lognorm':
                    params = stats.lognorm.fit(dwell_data)
                elif distribution_type == 'gamma':
                    params = stats.gamma.fit(dwell_data)
                elif distribution_type == 'expon':
                    params = stats.expon.fit(dwell_data)
                else:
                    raise ValueError(f"Unsupported distribution: {distribution_type}")
                
                self.dwell_distributions[state_idx] = {
                    'type': distribution_type,
                    'params': params,
                    'data': dwell_data
                }
        
        print("Model fitting complete!")
        return self
    
    def generate_sequence(self, max_length=100, random_state=None):
        """
        Generate a synthetic sequence from the fitted model
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        if self.states is None:
            raise ValueError("Model must be fitted before generating sequences")
        
        # Start with initial state
        current_state = np.random.choice(len(self.states), p=self.initial_state_probs)
        sequence = []
        dwell_times = []
        
        for _ in range(max_length):
            # Add current state
            sequence.append(self.states[current_state])
            
            # Generate dwell time for current state
            if current_state in self.dwell_distributions:
                dist_info = self.dwell_distributions[current_state]
                if dist_info['type'] == 'lognorm':
                    dwell_time = stats.lognorm.rvs(*dist_info['params'])
                elif dist_info['type'] == 'gamma':
                    dwell_time = stats.gamma.rvs(*dist_info['params'])
                elif dist_info['type'] == 'expon':
                    dwell_time = stats.expon.rvs(*dist_info['params'])
                
                dwell_times.append(max(1, int(dwell_time)))  # Ensure positive integer
            else:
                dwell_times.append(100)  # Default dwell time
            
            # Transition to next state
            next_state = np.random.choice(len(self.states), 
                                        p=self.transition_matrix[current_state])
            current_state = next_state
        
        return sequence, dwell_times
    
    def plot_transition_matrix(self, figsize=(10, 8)):
        """
        Plot the transition matrix as a heatmap
        """
        plt.figure(figsize=figsize)
        sns.heatmap(self.transition_matrix, 
                   xticklabels=self.state_names,
                   yticklabels=self.state_names,
                   annot=True, fmt='.3f', cmap='Blues')
        plt.title('State Transition Matrix')
        plt.xlabel('To State')
        plt.ylabel('From State')
        plt.tight_layout()
        plt.show()
    
    def plot_dwell_distributions(self, figsize=(15, 10)):
        """
        Plot dwell time distributions for each state
        """
        n_states = len(self.dwell_distributions)
        n_cols = 3
        n_rows = (n_states + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (state_idx, dist_info) in enumerate(self.dwell_distributions.items()):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            data = dist_info['data']
            state_name = self.state_names[state_idx]
            
            # Plot histogram
            ax.hist(data, bins=30, density=True, alpha=0.7, color='skyblue')
            
            # Plot fitted distribution
            x = np.linspace(data.min(), data.max(), 100)
            if dist_info['type'] == 'lognorm':
                y = stats.lognorm.pdf(x, *dist_info['params'])
            elif dist_info['type'] == 'gamma':
                y = stats.gamma.pdf(x, *dist_info['params'])
            elif dist_info['type'] == 'expon':
                y = stats.expon.pdf(x, *dist_info['params'])
            
            ax.plot(x, y, 'r-', linewidth=2, label=f'Fitted {dist_info["type"]}')
            ax.set_title(f'Dwell Time Distribution - {state_name}')
            ax.set_xlabel('Dwell Time')
            ax.set_ylabel('Density')
            ax.legend()
        
        # Hide empty subplots
        for i in range(n_states, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def get_model_summary(self):
        """
        Print summary statistics of the fitted model
        """
        print("=== Semi-Markov Model Summary ===")
        print(f"Number of states: {len(self.states)}")
        print(f"States: {self.state_names}")
        print("\nInitial state probabilities:")
        for i, prob in enumerate(self.initial_state_probs):
            print(f"  {self.state_names[i]}: {prob:.3f}")
        
        print("\nDwell time distribution parameters:")
        for state_idx, dist_info in self.dwell_distributions.items():
            state_name = self.state_names[state_idx]
            params = dist_info['params']
            mean_dwell = np.mean(dist_info['data'])
            print(f"  {state_name} ({dist_info['type']}): params={params}, mean_dwell={mean_dwell:.1f}")

# Example usage
def example_usage():
    """
    Example of how to use the Semi-Markov model with your fixation data
    """
    
    # Assuming your dataframe is called fixation_df
    # fixation_df = pd.read_csv('your_data.csv')  # Load your data
    
    # Create sample data for demonstration
    np.random.seed(42)
    sample_data = {
        'session_name': ['01022019'] * 100,
        'run_number': [1] * 50 + [2] * 50,
        'agent': ['m1'] * 100,
        'start': np.cumsum(np.random.exponential(50, 100)),
        'category': np.random.choice(['face', 'out_of_roi', 'eyes_nf'], 100, p=[0.4, 0.4, 0.2])
    }
    sample_data['stop'] = sample_data['start'] + np.random.lognormal(4, 1, 100)
    fixation_df = pd.DataFrame(sample_data)
    
    # Fit the semi-Markov model
    model = SemiMarkovModel()
    model.fit(fixation_df, distribution_type='lognorm')
    
    # Print model summary
    model.get_model_summary()
    
    # Plot visualizations
    model.plot_transition_matrix()
    model.plot_dwell_distributions()
    
    # Generate synthetic sequences
    synthetic_seq, synthetic_dwells = model.generate_sequence(max_length=20, random_state=42)
    print("\nSynthetic sequence:")
    for state, dwell in zip(synthetic_seq[:10], synthetic_dwells[:10]):
        print(f"  {state}: {dwell} time units")
    
    return model

# Run the example
if __name__ == "__main__":
    model = example_usage()