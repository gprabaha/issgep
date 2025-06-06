

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# --- Core Functions (same as before, compacted) ---

def estimate_probabilities_and_errors_with_crosscorr(a: np.ndarray, b: np.ndarray):
    N = len(a)
    a, b = a.astype(bool), b.astype(bool)
    ab = a & b
    p_a, p_b, p_ab = np.mean(a), np.mean(b), np.mean(ab)
    r_a = acf(a, nlags=1, fft=True)[1]
    r_b = acf(b, nlags=1, fft=True)[1]
    r_ab = acf(ab, nlags=1, fft=True)[1]
    cross_ab = np.corrcoef(a, b)[0, 1]
    shifted_ab = np.corrcoef(a[:-1], b[1:])[0, 1]
    N_eff_ab = N * ((1 - r_a) / (1 + r_a))**0.25 * ((1 - r_b) / (1 + r_b))**0.25 * ((1 - r_ab) / (1 + r_ab))**0.5
    N_eff_ab = max(1, N_eff_ab)
    se_a = np.sqrt(p_a * (1 - p_a) / (N * (1 - r_a) / (1 + r_a)))
    se_b = np.sqrt(p_b * (1 - p_b) / (N * (1 - r_b) / (1 + r_b)))
    se_ab = np.sqrt(p_ab * (1 - p_ab) / N_eff_ab)
    return {
        "P(A)": (p_a, se_a),
        "P(B)": (p_b, se_b),
        "P(AB)": (p_ab, se_ab),
        "cross_corr_0": cross_ab,
        "cross_corr_lag1": shifted_ab,
        "N_eff_AB": N_eff_ab
    }

def sliding_window_dependence(a, b, window_size=1000, step_size=100):
    indices = range(0, len(a) - window_size + 1, step_size)
    results = []
    for start in indices:
        aw, bw = a[start:start + window_size], b[start:start + window_size]
        p_a, p_b = np.mean(aw), np.mean(bw)
        p_ab = np.mean(aw & bw)
        delta = p_ab - (p_a * p_b)
        results.append((start + window_size // 2, delta))
    return np.array(results)

def gmm_on_deltas(deltas, k_range=range(1, 6)):
    deltas_reshaped = deltas.reshape(-1, 1)
    models, bics, silhouettes = [], [], []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=0).fit(deltas_reshaped)
        labels = gmm.predict(deltas_reshaped)
        models.append((gmm, labels))
        bics.append(gmm.bic(deltas_reshaped))
        silhouettes.append(silhouette_score(deltas_reshaped, labels) if k > 1 else np.nan)
    best_idx = np.argmin(bics)
    return models[best_idx], bics, silhouettes


def fit_hmm_to_joint_state(a, b, n_states=2):
    """Fit a discrete HMM to the joint state of A and B"""
    joint_states = (a.astype(int) << 1) | b.astype(int)  # 2-bit state: 00, 01, 10, 11 -> 0, 1, 2, 3
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, random_state=0)
    model.fit(joint_states.reshape(-1, 1))
    state_sequence = model.predict(joint_states.reshape(-1, 1))
    return model, state_sequence

# Example usage (a and b must be defined as binary arrays of same length)
# a, b = <binary numpy arrays>
# results = estimate_probabilities_and_errors(a, b)
# observed_pab, pab_shuffled, p_val = permutation_test_pab(a, b)
# sliding_results = sliding_window_dependence(a, b)
# gmm, gmm_labels = gmm_on_dependence_deltas(sliding_results[:, 4])
# hmm_model, hmm_states = fit_hmm_to_joint_state(a, b)


# --- Sample Data Generation ---

np.random.seed(0)
T = 30000  # 30 seconds at 1 ms resolution
a = np.random.rand(T) < 0.01  # ~1% on-state
b = np.roll(a, 50) ^ (np.random.rand(T) < 0.005)  # introduce slight desync and noise

# --- Run Analysis ---

print("\n--- PROBABILITY & CORRELATION ESTIMATES ---")
estimates = estimate_probabilities_and_errors_with_crosscorr(a, b)
for k, (v, e) in estimates.items():
    if 'P' in k:
        print(f"{k}: {v:.4f} ± {e:.4f}")
    else:
        print(f"{k}: {v:.4f}")

# --- Try multiple sliding window sizes ---
window_sizes = [500, 1000, 2000]
fig, axs = plt.subplots(len(window_sizes), 2, figsize=(12, 3 * len(window_sizes)))

for i, w in enumerate(window_sizes):
    delta_arr = sliding_window_dependence(a, b, window_size=w, step_size=w//10)
    midpoints, deltas = delta_arr[:, 0], delta_arr[:, 1]
    
    # Plot delta trace
    axs[i, 0].plot(midpoints / 1000, deltas, label=f'window={w}ms')
    axs[i, 0].set_title(f"Δ = P(AB) - P(A)P(B), window={w}ms")
    axs[i, 0].set_xlabel("Time (s)")
    axs[i, 0].set_ylabel("Δ")
    axs[i, 0].axhline(0, color='gray', linestyle='--')

    # GMM
    (gmm, labels), bics, sils = gmm_on_deltas(deltas)
    axs[i, 1].scatter(midpoints / 1000, deltas, c=labels, cmap='tab10', s=10)
    axs[i, 1].set_title(f"GMM (k={gmm.n_components}) | BIC min @ k={np.argmin(bics)+1}")
    axs[i, 1].set_xlabel("Time (s)")
    axs[i, 1].set_ylabel("Δ")

plt.tight_layout()
plt.show()

# --- Plot BICs for last window size ---
plt.figure(figsize=(6, 4))
plt.plot(range(1, 6), bics, marker='o')
plt.title("GMM BIC Scores")
plt.xlabel("Number of Components")
plt.ylabel("BIC")
plt.grid(True)
plt.show()



