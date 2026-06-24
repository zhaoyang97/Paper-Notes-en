---
title: >-
  [Paper Note] Causal Discovery from Conditionally Stationary Time Series
description: >-
  [ICML 2025][Time Series][Causal Discovery] Proposed SDCI (State-Dependent Causal Inference)—a causal discovery method for conditionally stationary time series. It models non-stationary behavior using discrete latent state variables to perform state-dependent causal structure recovery, with its effectiveness validated on physical particle systems, gene regulatory networks, and NBA player motion prediction.
tags:
  - "ICML 2025"
  - "Time Series"
  - "Causal Discovery"
  - "Non-stationary Time Series"
  - "Conditional Stationarity"
  - "Graph Neural Networks"
  - "Latent State"
date: 2026-05-08
content_hash: fbe1c5083ec0923e
---

# Causal Discovery from Conditionally Stationary Time Series

**Conference**: ICML 2025  
**arXiv**: [2110.06257](https://arxiv.org/abs/2110.06257)  
**Code**: None  
**Area**: Time Series  
**Keywords**: Causal Discovery, Non-stationary Time Series, Conditional Stationarity, Graph Neural Networks, Latent State

## TL;DR
Proposed SDCI (State-Dependent Causal Inference)—a causal discovery method for conditionally stationary time series. It models non-stationary behavior using discrete latent state variables to perform state-dependent causal structure recovery, with its effectiveness validated on physical particle systems, gene regulatory networks, and NBA player motion prediction.

## Background & Motivation

**Background**: Traditional time-series causal discovery methods (Granger Causality, TiMINo, etc.) assume stationarity, meaning the causal structure remains unchanged over time. However, real-world data are often non-stationary.

**Limitations of Prior Work**:
   - Methods modeling non-stationary noise (Huang et al.) assume a fixed causal structure with only the noise changing.
   - Methods modeling time-varying effects assume the causal graph is invariant, with only the effect strengths varying.
   - Methods using discrete latent variables (Saggioro et al.) only handle simple cases where the states are state-independent or fully observable.

**Key Challenge**: Real-world causal relationships are typically "state-dependent." For instance, a basketball player's behavior depends on the preceding game situations, and the interaction rules of physical particles are completely different before and after a collision.

**Goal**: To recover causal graphs from non-stationary time series where the causal structure varies with discrete latent states.

**Key Insight**: Modeling non-stationarity as "conditional stationarity"—namely, the system is stationary conditioned on the latent states. Three scenarios are considered: (a) observable states; (b) states determined by observations; (c) states dependent on history (recurrent states).

**Core Idea**: A Conditional Summary Graph to serve as a compact causal representation of conditionally stationary time series, paired with a practical algorithm based on GNNs and discrete latent variables.

## Method

### Overall Architecture
Core components of SDCI:
1. **Causal Graph Representation**: Conditional Summary Graph $\mathcal{G}^{CS}_{1:K} = \{\mathcal{G}^{CS}_k, k=1,...,K\}$, where each state $k$ corresponds to a causal graph.
2. **Latent State Inference**: Inferring the current state $u_t$ based on observations.
3. **Causal Structure Learning**: Utilizing GNNs to learn a causal adjacency matrix for each state.
4. **Prediction**: Predicting the next step based on the inferred states and the corresponding causal graphs.

### Key Designs

1. **Conditional Summary Graph**:

    - **Function**: To compactly represent state-dependent causal structures.
    - **Mechanism**: The Full Time Graph is time-varying under non-stationarity, making direct learning exponentially complex. The Conditional Summary Graph groups all time steps by state, associating each group with a static causal graph.
    - **Formalization**: $i \to j \in \mathcal{E}^{CS}_k$ if and only if $x_t^{(i)}$ causally influences $x_{t+1}^{(j)}$ under state $k$.
    - **Design Motivation**: To reduce the exponential complexity to $O(K \cdot N^2)$, where $K$ is the number of states and $N$ is the number of variables.

2. **State Inference Mechanism**:

    - **Function**: To infer discrete latent states from observations.
    - **"State-determined" scenario**: The state is directly determined by the current observation, i.e., $u_t = g(x_t)$.
    - **"State-recurrent" scenario**: The state depends on history, inferred by an RNN encoder as $q(u_t | x_{1:t})$.
    - **Mechanism**: Variational inference using Gumbel-Softmax to make discrete states differentiable.
    - **Design Motivation**: Different scenarios require distinct inference mechanisms, unified under a variational framework.

3. **Identifiability Proof**:

    - **Function**: To rigorously prove that the Conditional Summary Graph can be uniquely recovered under specific conditions.
    - **Mechanism**: Extending the identifiability of the Additive Noise Model (ANM) to state-dependent settings.
    - **Theorem Content**: Under conditional stationarity + ANM + sufficient excitation conditions, the Conditional Summary Graph is identifiable.
    - **Design Motivation**: To provide theoretical guarantees for the practical algorithm.

4. **GNN Causal Structure Learning**:

    - **Function**: To parameterize state-dependent causal transition functions.
    - **Mechanism**: Each state $k$ has an independent adjacency matrix $A_k$, and GNN layers propagate information according to $A_k$.
    - **Training Objective**: Reconstruction loss + sparsity regularization ($L_1$ on $A_k$) + KL divergence (latent state prior).
    - **Design Motivation**: The message passing in GNNs naturally corresponds to the propagation of causal influences.

### Loss & Training
- Total loss = Reconstruction loss (predicting $x_{t+1}$) + Sparsity penalty ($\lambda \sum_k \|A_k\|_1$) + KL divergence ($D_{KL}(q(u_t) || p(u_t))$)
- Alternating updates via EM algorithm: the E-step infers latent states, while the M-step optimizes causal graphs and transition functions.
- Gumbel-Softmax temperature annealing enables differentiable training for discrete states.

## Key Experimental Results

### Main Results
Particle interaction dataset (Spring system with collisions):

| Method | AUROC (Causal Graph) | State Classification Accuracy | Prediction MSE |
|------|-------------|-------------|---------|
| NRI (Static Causal Graph) | 0.78 | - | 0.042 |
| dNRI (Time-varying Causal Graph) | 0.82 | 0.71 | 0.035 |
| **SDCI** | **0.93** | **0.89** | **0.021** |

### Gene Regulatory Networks

| Method | AUROC | AUPRC |
|------|-------|-------|
| Granger (Stationary Assumption) | 0.62 | 0.48 |
| PCMCI | 0.68 | 0.55 |
| **SDCI** | **0.81** | **0.72** |

### NBA Player Motion Prediction

| Method | Prediction Error (m) ↓ |
|------|-------------|
| LSTM | 0.85 |
| NRI | 0.72 |
| **SDCI** | **0.58** |

### Ablation Study

| Configuration | Causal Graph AUROC | Description |
|------|------------|------|
| Fixed Causal Graph (No State) | 0.78 | Degenerates to NRI |
| Continuous States | 0.85 | Less distinct than discrete states |
| **Discrete States (SDCI)** | **0.93** | Optimal |
| K=2 States | 0.90 | Sufficient to capture collision/non-collision |
| K=5 States | 0.93 | Finer granularity |
| No Sparsity Regularization | 0.81 | Overconnected |

### Key Findings
- The AUROC of causal graph recovery improves from 0.82 (baseline) to 0.93, demonstrating the significant value of state-dependent modeling.
- Discrete states fit causal structure transitions better than continuous states (since changes in causal graphs are discrete switches).
- Improvements on NBA data suggest that causal modeling aids practical prediction (not merely graph recovery metrics).
- The "state-recurrent" scenario is more challenging than the "state-determined" one, but the results remain robust.
- The identifiability proof provides theoretical guarantees, ensuring that the algorithm converges to the correct causal graph (when assumptions are met).

## Highlights & Insights
- The **conditional stationarity assumption** is an elegant relaxation for non-stationary time series; it is more tractable than "complete non-stationarity" and closer to reality than "stationarity."
- The Conditional Summary Graph, as a causal representation, balances compactness and expressiveness; having one graph per state is far more efficient than having one graph per time step.
- The physically-inspired experimental design (where collisions change causal relationships) is highly intuitive; particles move independently before collision and repel each other after collision.
- The improvement from causal discovery to prediction indicates that the learned causal structure is genuinely useful, rather than just yielding gains in graph recovery metrics.
- The finding that "discrete states outperform continuous states" holds independent value, as changes in causal structures are inherently discrete.

## Limitations & Future Work
- The number of states $K$ needs to be specified a priori; automated determination would be more practical.
- Only the first-order Markov assumption is considered; higher-order dependencies remain unaddressed.
- The assumptions of no latent confounding variables and no instantaneous effects are relatively strong.
- The scalability of GNNs is limited when the number of variables $N$ is very large.
- The identifiability proof is based on Gaussian ANM; theoretical guarantees for non-Gaussian scenarios require extension.
- The "ground truth" of causal relationships in real-world gene regulatory networks may not be fully reliable.

## Related Work & Insights
- **vs NRI/dNRI**: NRI assumes a static causal graph, and dNRI allows time-varying structures but lacks a concept of state; SDCI decouples different causal patterns using discrete states.
- **vs Granger Causality**: Granger Causality assumes stationarity and linearity, whereas SDCI handles non-stationarity and non-linearity.
- **vs PCMCI**: PCMCI handles time-lagged causality but assumes stationarity, while SDCI relaxes the stationarity assumption.
- **vs Markov Switching Models**: Traditional MSMs do not conduct causal discovery (only modeling state switching); SDCI jointly learns state transitions and causal structures.
- **Insight**: The framework of conditional stationarity + discrete latent variables can be generalized to other scenarios requiring the discovery of "regime-dependent relationships," such as distinguishing between bullish and bearish causal structures in financial markets.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Conditionally stationary causal discovery is an important theoretical and algorithmic contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthesis, biological, and sports data, validated in multiple scenarios.
- Writing Quality: ⭐⭐⭐⭐ The theoretical framework is clear, and the taxonomy is useful.
- Value: ⭐⭐⭐⭐⭐ Advances the frontier of non-stationary causal discovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models](../../NeurIPS2025/time_series/causaldynamics_a_large-scale_benchmark_for_structural_discovery_of_dynamical_cau.md)
- [\[AAAI 2026\] Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing](../../AAAI2026/time_series/towards_non-stationary_time_series_forecasting_with_temporal_stabilization_and_f.md)
- [\[NeurIPS 2025\] Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction](../../NeurIPS2025/time_series/neural_mjd_neural_non-stationary_merton_jump_diffusion_for_time_series_predictio.md)
- [\[ICML 2025\] Channel Normalization for Time Series Channel Identification](channel_normalization_for_time_series_channel_identification.md)
- [\[ICML 2025\] Lyapunov Learning at the Onset of Chaos](lyapunov_learning_at_the_onset_of_chaos.md)

</div>

<!-- RELATED:END -->
