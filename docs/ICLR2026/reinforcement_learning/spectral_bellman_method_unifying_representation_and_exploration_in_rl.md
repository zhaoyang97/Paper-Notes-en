---
title: >-
  [Paper Note] Spectral Bellman Method: Unifying Representation and Exploration in RL
description: >-
  [ICLR 2026][Reinforcement Learning][Representation Learning] This paper proposes the Spectral Bellman Method (SBM), which derives a spectral relationship between the Bellman operator and feature covariance structure from the zero Intrinsic Bellman Error (IBE) condition, leading to a novel representation learning objective that naturally unifies representation learning and Thompson Sampling–based exploration.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Representation Learning
  - Exploration
  - Bellman Error
  - Spectral Decomposition
  - Thompson Sampling
date: 2026-05-08
content_hash: 3c44e23f44fd8a60
---

# Spectral Bellman Method: Unifying Representation and Exploration in RL

**Conference**: ICLR 2026
**arXiv**: [2507.13181](https://arxiv.org/abs/2507.13181)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Representation Learning, Exploration, Bellman Error, Spectral Decomposition, Thompson Sampling

## TL;DR

This paper proposes the Spectral Bellman Method (SBM), which derives a spectral relationship between the Bellman operator and feature covariance structure from the zero Intrinsic Bellman Error (IBE) condition, leading to a novel representation learning objective that naturally unifies representation learning and Thompson Sampling–based exploration.

## Background & Motivation

Efficient reinforcement learning in complex environments faces two core challenges: **learning effective representations** and **performing efficient exploration**. Existing methods typically treat these as independent problems, yet they share deep connections—good representations should simultaneously support accurate value estimation and strategic data collection.

Most existing representation learning approaches (autoencoders, contrastive learning, predictive models, successor features, etc.) are motivated from a model-learning perspective, lacking alignment with the core structure of RL (Bellman updates). A key theoretical framework is the **Intrinsic Bellman Error (IBE)**:

- IBE quantifies how well a feature space supports value-based RL.
- Zero IBE implies that the function space is closed under the Bellman operator, a generalization of the Linear MDP assumption.
- Features satisfying zero IBE can support efficient exploration.

However, directly learning low-IBE representations faces severe difficulties:
1. Direct minimization of IBE leads to a complex min-max-min optimization problem.
2. The Bellman operator is highly nonlinear with respect to $Q_\theta$.
3. Simple MSE objectives neither exploit spectral properties nor encourage structured features.

**Core Insight**: Under the zero IBE condition, there exists a fundamental **spectral relationship** between the transformation of Q-functions under the Bellman operator and the feature covariance structure. This spectral relationship can be leveraged to design practical learning objectives.

## Method

### Overall Architecture

The core theoretical pipeline of SBM:

1. **Zero IBE Condition** → function space is closed under the Bellman operator
2. **Spectral Analysis** → reveals alignment between the SVD of the Bellman transformation matrix and the feature covariance matrix
3. **Power Iteration Analogy** → derives an alternating optimization objective (SBM Loss)
4. **Covariance Structure** → naturally embeds Thompson Sampling exploration

### Key Designs

1. **Bellman Spectral Decomposition Theorem**: Structural revelation under zero IBE

    - Core finding: Under the zero IBE condition, defining the weighted feature matrix $\Phi_P$ and the weighted post-Bellman parameter matrix $\tilde{\Theta}_P$, the SVD of the Bellman transformation matrix $\mathcal{T}\bar{Q} = \Phi_P \tilde{\Theta}_P$ is directly related to the feature covariance matrix $\Lambda = \mathbb{E}[\phi(s,a)\phi(s,a)^\top]$.
    - The nonzero singular values coincide with the eigenvalues of $\Lambda$; the left singular vectors correspond to weighted features, and the right singular vectors correspond to weighted parameters.
    - Key corollary: $\Lambda_1 = \Lambda_2 = \Lambda$, i.e., the feature covariance and the post-Bellman parameter covariance are aligned.

2. **SBM Loss**: A practical learning objective based on power iteration

    - Motivated by the power iteration method for SVD, an alternating optimization objective is derived:
    - $\mathcal{L}(\phi, \tilde{\theta}) = \mathcal{L}_1(\phi) + \mathcal{L}_2(\tilde{\theta}) + \mathcal{L}_{orth}(\phi, \tilde{\theta})$
    - **Representation loss** $\mathcal{L}_1(\phi)$: updates $\phi$ to align with the Bellman-transformed Q-values, regularized by the current parameter covariance $\Lambda_{2,t}$.
    - **Parameter loss** $\mathcal{L}_2(\tilde{\theta})$: updates $\tilde{\theta}$ to best represent the Bellman transformation results, regularized by the current feature covariance $\Lambda_{1,t}$.
    - **Orthogonality regularization** $\mathcal{L}_{orth}$: enforces orthogonality across feature dimensions.
    - Key theorem (Proposition 2): minimizing the SBM Loss is equivalent to performing a power iteration update.

3. **Advantages of SBM Loss over Bellman MSE**:

    - The quadratic term $\|\phi(s,a)\|_{\hat{\Lambda}}^2$ in the MSE objective uses a **single-sample noisy estimate** $\hat{\Lambda}$.
    - SBM's quadratic term uses a **moving-average covariance** $\Lambda_{2,t}$, providing robust batch-statistic regularization.
    - The decoupled structure $\mathcal{L}_1 + \mathcal{L}_2$ naturally supports alternating optimization (the intrinsic structure of power iteration), yielding greater stability than simultaneous MSE optimization.
    - SBM explicitly includes orthogonality regularization.

4. **Thompson Sampling Exploration**: Naturally driven by the feature covariance

    - Given learned features $\phi$, a precision matrix is constructed: $\Sigma = \lambda I + \sum_{(s,a) \in \mathcal{D}} \phi(s,a)\phi(s,a)^\top$.
    - Before each rollout, a parameter sample is drawn from the posterior: $\hat{\theta}_{TS} \sim \mathcal{N}(\hat{\theta}_{LS}, \sigma_{exp} \Sigma^{-1})$.
    - Naturally compatible with low-IBE representations—the feature covariance structure simultaneously encodes value estimation uncertainty and exploration directions.
    - Also compatible with UCB-based methods (using the same $\Sigma$).

### Loss & Training

The full algorithm (Algorithm 2) alternates among three phases in each iteration:

1. **Data Collection** (Thompson Sampling): sample $\hat{\theta}_{TS} \sim \mathcal{N}(\hat{\theta}_t, \sigma_{exp} \Sigma_t^{-1})$ and collect data using the greedy policy $\pi_{\hat{\theta}_{TS}}$.
2. **Policy Optimization**: update $\theta$ using the standard Q-learning loss $\mathcal{L}_{QL}(\theta; \phi) = \mathbb{E}[(\mathcal{T}Q_{\theta^-}(s,a) - \phi(s,a)^\top\theta)^2]$.
3. **Representation Learning**: update features $\phi$ using the SBM Loss, with the parameter distribution centered at the current Q-parameters $\nu(\theta) = \mathcal{N}(\hat{\theta}_{t+1}, \sigma_{rep}^2 I)$.

$\tilde{\theta}(\theta)$ is implemented as a residual network: $\tilde{\theta}(\theta) = \theta + \Delta(\theta)$, where $\Delta$ is a trainable MLP.

Covariance matrices are updated via exponential moving averages to ensure stability.

## Key Experimental Results

### Main Results

| Method | Atari 57 Mean | Atari 57 Median | Atari Explore Mean | Atari Explore Median |
|---|---|---|---|---|
| DQN | 1.61 | 0.63 | 0.22 | 0.03 |
| SBM + DQN (ε-greedy) | 1.83 | 0.81 | 0.36 | 0.08 |
| **SBM + DQN (TS)** | **1.91** | **0.98** | **0.42** | **0.21** |
| R2D2 | 3.2 | 1.02 | 0.42 | 0.25 |
| SBM + R2D2 (ε-greedy) | 3.3 | 1.14 | 0.45 | 0.26 |
| **SBM + R2D2 (TS)** | **3.51** | **1.32** | **0.67** | **0.31** |

(Scores are Human Normalized Scores at 100M steps.)

### Ablation Study

| Configuration | Description |
|---|---|
| SBM + ε-greedy | Gain from representation learning alone |
| SBM + TS | Synergistic gain from representation + exploration |
| DQN → +SBM | +19% Mean HNS (Atari 57), +91% Mean HNS (Explore) |
| R2D2 → +SBM+TS | +10% Mean HNS (Atari 57), +60% Mean HNS (Explore) |

### Key Findings

1. **Synergy between representation learning and exploration**: The gains from SBM + TS significantly exceed those from either component alone, validating the superiority of the unified framework.
2. **Greater advantage on hard exploration games**: Improvements on the Atari Explore subset (Montezuma's Revenge, Pitfall!, etc.) are proportionally far larger than those on the full 57-game benchmark.
3. **Compatibility with multi-step operators**: SBM extends naturally to the Retrace(λ) objective and integrates seamlessly with R2D2's distributed training.
4. **Co-evolution of representation and policy optimization**: improved features → better value estimation → better policy → better data → further improved features.

## Highlights & Insights

1. **Balance between theoretical depth and practicality**: Starting from IBE theory to derive spectral objectives, the final algorithm requires only a simple modification to existing Q-learning—adding an auxiliary SBM loss.
2. **Elegance of the unified perspective**: A single spectral objective simultaneously drives three aspects—features with low Bellman error, structured covariance, and covariance-based exploration noise.
3. **Avoidance of hard optimization problems**: Direct IBE minimization requires min-max-min optimization; SBM reformulates this as power-iteration-style alternating optimization, which is simple and converges rapidly.
4. **The comparative analysis of SBM Loss vs. MSE** is incisive: by expanding MSE and comparing it term-by-term with SBM, the paper clearly demonstrates the advantage of moving-average covariance over single-sample estimation.

## Limitations & Future Work

1. **Sensitivity to the parameter distribution $\nu(\theta)$**: Careful tuning of $\sigma_{rep}$ is required, and different environments may demand different configurations.
2. **Validation limited to Atari**: The method has not been evaluated on continuous control tasks (e.g., MuJoCo), and its generality remains to be confirmed.
3. **Incomplete theoretical analysis**: Convergence proofs and analysis of behavior under nonzero IBE have not been established.
4. **Approximation of $\tilde{\theta}$**: The current approach approximates $\tilde{\theta}(\theta)$ with a residual MLP; better parameterizations may further improve performance.
5. **No direct comparison with other representation learning methods (CURL, SPR, etc.)**: It is therefore difficult to assess the absolute advantage of the spectral objective over contrastive or predictive learning objectives.

## Related Work & Insights

- **Intrinsic Bellman Error (IBE)**: Zanette et al. (2020a) introduced the IBE concept and established regret bounds for planning-based algorithms; this paper makes IBE-guided representation learning practically viable for the first time.
- **Linear MDP**: Jin et al. (2020) assumed linear transitions and rewards, which are difficult to satisfy in practice; IBE relaxes this condition.
- **Spectral decomposition for representations**: The spectral decomposition work of Ren et al. (2022/2023) is a direct predecessor, but this paper goes further by establishing connections to power iteration and Thompson Sampling exploration.
- **Inspiration**: This framework demonstrates how practical algorithms can be designed from the core mathematical structure of RL (spectral properties of the Bellman operator), rather than relying on general-purpose self-supervised heuristic objectives.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Discovering the connection between IBE and spectral structure, and deriving the practical SBM Loss, constitutes an original contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Full Atari 57 suite plus a hard exploration subset, combined with two baselines (DQN and R2D2), though continuous control and broader comparisons are lacking.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear, and the chain from theorems to algorithm is complete.
- Value: ⭐⭐⭐⭐⭐ — Provides a theoretically grounded new paradigm for representation learning in RL, with significant implications for future work.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)
- [\[ICLR 2026\] Stackelberg Coupling of Online Representation Learning and Reinforcement Learning](stackelberg_coupling_of_online_representation_learning_and_reinforcement_learnin.md)
- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](../../AAAI2026/reinforcement_learning/first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[ICLR 2026\] A Unifying View of Coverage in Linear Off-Policy Evaluation](a_unifying_view_of_coverage_in_linear_off-policy_evaluation.md)
- [\[ICLR 2026\] Emergence of Spatial Representation in an Actor-Critic Agent with Hippocampus-Inspired Sequence Generator](emergence_of_spatial_representation_in_an_actor-critic_agent_with_hippocampus-in.md)

<!-- RELATED:END -->
