---
title: >-
  [Paper Note] Conformal Online Learning of Deep Koopman Linear Embeddings
description: >-
  [NEURIPS2025][LLM Evaluation][Koopman operator] This paper proposes the COLoKe framework, which reinterprets conformal prediction as a model consistency diagnostic tool. Parameter updates are triggered only when the Koop…
tags:
  - "NEURIPS2025"
  - "LLM Evaluation"
  - "Koopman operator"
  - "online learning"
  - "conformal prediction"
  - "dynamical systems"
  - "deep learning"
date: 2026-05-08
content_hash: d432ff86726ac5c9
---

# Conformal Online Learning of Deep Koopman Linear Embeddings

**Conference**: NEURIPS2025  
**arXiv**: [2511.12760](https://arxiv.org/abs/2511.12760)  
**Code**: [ben2022lo/COLoKe](https://github.com/ben2022lo/COLoKe)  
**Area**: LLM Evaluation  
**Keywords**: Koopman operator, online learning, conformal prediction, dynamical systems, deep learning

## TL;DR
This paper proposes the COLoKe framework, which reinterprets conformal prediction as a model consistency diagnostic tool. Parameter updates are triggered only when the Koopman model's prediction error exceeds a dynamically calibrated threshold, enabling efficient online Koopman linear embedding learning for nonlinear dynamical systems.

## Background & Motivation
Koopman operator theory provides a powerful framework for lifting nonlinear dynamical systems into an infinite-dimensional function space for linear analysis. The core idea is that although the dynamics in state space are nonlinear, the evolution of observable functions along trajectories can be exactly described by a linear operator — the Koopman operator. In finite-dimensional approximations, this is equivalent to finding a feature map $\Phi$ and a matrix $K$ such that $\Phi(x_{t+1}) = K \Phi(x_t)$.

Primary limitations of existing methods:

- **Offline methods** (DMD, EDMD, deep Koopman autoencoders, etc.) assume full one-time data access and are unsuitable for streaming data scenarios.
- **Online methods** (Online DMD, Online EDMD) rely on linear observables or fixed dictionaries, limiting their expressive capacity.
- **Neural network-based online methods** (OnlineAE, DKLT, etc.) lack principled update strategies, typically performing a fixed number of gradient steps per time step regardless of necessity, leading to overfitting or computational waste.

In practical applications (robotic control, real-time prediction, online monitoring, etc.), data arrives as a stream and the system may undergo distribution drift over time. This motivates the need for an online learning strategy that is both adaptive and avoids unnecessary computation.

## Core Problem
How can one adaptively determine **when to update** and **how much to update** the parameters of a Koopman embedding model in a streaming data setting, maintaining long-term prediction accuracy while avoiding overfitting and computational waste?

## Method

### 1. Deep Koopman Embedding Architecture
The feature map is designed to preserve the original state structure:
$$\Phi_{\theta}(x) = [x, \tilde{\Phi}_{\theta}(x)]^\top$$
where $\tilde{\Phi}_{\theta}$ is a learnable neural network component. This design ensures the lifted representation retains raw state information while enhancing expressiveness through learned nonlinear embeddings, simultaneously eliminating the need for an explicit decoder.

### 2. Multi-Step Prediction Loss
At each time step $t$, a sliding window $\mathcal{D}_t = \{x_{t-w}, \ldots, x_t\}$ of size $w$ is maintained. The training loss accumulates errors over all valid multi-step prediction pairs within the window:
$$\mathcal{L}_t(\theta, K) = \sum_{(s,\tau) \in \mathcal{I}_t} \sum_{j=1}^{\tau} \|\Phi_\theta(x_{s+\tau}) - K^j \Phi_\theta(x_{s+\tau-j})\|^2$$
Multi-step prediction facilitates the identification of persistent spectral patterns and approximate Koopman eigenfunctions, thereby improving long-term prediction capability.

### 3. Conformal Update Mechanism (Core Contribution)
Traditional conformal prediction is used to construct prediction intervals for uncertainty quantification. This paper **reinterprets** it as a model consistency diagnostic tool.

**Prediction consistency score**: The consistency score of the current model on a new observation $x_t$ is defined as:
$$s_t = \ell_{t-w,w}(\theta_t, K_t) = \sum_{\tau=1}^{w} \|\Phi_{\theta_t}(x_t) - K_t^\tau \Phi_{\theta_t}(x_{t-\tau})\|^2$$

**Adaptive threshold**: A Conformal PI controller dynamically adjusts the threshold $q_t$:
$$q_{t+1} = q_t + \gamma(e_t - \alpha) + r_t\left(\sum_{i=1}^{t}(e_i - \alpha)\right)$$
where $e_t = \mathbf{1}\{s_t > q_t\}$ is a binary error signal, $\gamma$ is the learning rate, and $r_t$ is an integral correction term.

**Update decision**:
- If $s_t \leq q_t$: the model remains consistent with the data — **no update**.
- If $s_t > q_t$: the model is no longer consistent — **gradient updates are triggered** and iterated until $s_t \leq q_t$.

Key perspective shift: rather than assessing whether a new data point conforms to the model (traditional conformal prediction), the framework assesses whether the current model parameters remain consistent with the new data.

### 4. Theoretical Guarantees
Under standard online learning assumptions (smooth loss functions and bounded variation of the dynamic oracle path), the following dynamic regret bound is established:
$$\sum_{t=1}^{T}[\mathcal{L}_t(\theta_t, K_t) - \mathcal{L}_t(\theta_t^*, K_t^*)] \leq \mathcal{O}(\alpha h(T) + V_T + S_T)$$
where $h(T)$ is a sublinear function, and $V_T$ and $S_T$ denote the total variation and squared variation of the oracle path, respectively.

## Key Experimental Results

### Datasets
- **Synthetic**: Single Attractor, Duffing oscillator, Van der Pol oscillator, Lorenz system
- **Real-world**: Electricity Transformer (ETD), EEG Motor Movement, Turbulence (CASES-99)

### Main Results (Table 2, Generalization Error)

| Dataset | ODMD | OnlineAE | OLoKe (w/o conformal) | **COLoKe** |
|--------|------|----------|---------------------|------------|
| Single Attractor | 1.1e-3 | 1.0e-2 | 2.1e-6 | **2.4e-7** |
| Duffing | 2.5e-4 | 8.7e-3 | 5.5e-5 | **3.1e-6** |
| VdP | 2.1e-3 | 1.7e-2 | 6.6e-4 | **3.8e-4** |
| Lorenz | 2.7e-1 | 5.9e-1 | 7.6e-3 | **6.5e-3** |

### Key Findings
- COLoKe achieves best or near-best performance on all synthetic and real-world datasets.
- On the chaotic Lorenz system, (C)OLoKe improves over baseline methods by nearly **two orders of magnitude**.
- The conformal triggering mechanism systematically improves upon OLoKe with fixed-step updates.
- Spectral analysis confirms: on analytically solvable systems, COLoKe recovers accurate Koopman eigenvalues (e.g., true values $\{-1, -0.05, -0.1\}$, estimated as $\{-1.0091, -0.04996, -0.1001\}$).

### Computational Efficiency
Pareto frontier analysis on the high-dimensional EEG dataset demonstrates that COLoKe simultaneously achieves lower online error and shorter execution time without manual tuning, outperforming all fixed-iteration OLoKe variants (1/5/10/50/100/150 steps).

## Highlights & Insights
1. **Perspective innovation**: Reinterpreting conformal prediction from "assessing whether new data conforms to the model" to "assessing whether the current model remains consistent with the data" — a concise yet profound shift.
2. **Adaptive updates**: Automatically determines when and how much to update, eliminating the need for manual iteration tuning while avoiding overfitting.
3. **Decoder-free design**: By embedding the raw state within the lifted representation, reconstruction constraints are implicitly incorporated into the consistency loss, removing the need for a decoder.
4. **Theoretical grounding**: Dynamic regret bounds are formally established.
5. **Methodological generality**: The conformal online learning framework is not restricted to Koopman settings and extends to more general online non-convex learning problems.

## Limitations & Future Work
1. **Assumption (A3) lacks first-principles derivation**: The condition of sublinear growth of the cumulative threshold is supported only empirically and has not been rigorously proven; the authors acknowledge this as an open problem.
2. **Restricted to autonomous systems**: The current method assumes dynamics driven by a time-invariant map $T$ and does not cover non-autonomous systems (i.e., systems driven by time-varying external forcing).
3. **Sensitivity to window size $w$**: The window size is a pre-specified hyperparameter that may require different settings for different systems.
4. **Inherent limitations of linear Koopman approximation**: For highly nonlinear or chaotic systems, the accuracy ceiling of finite-dimensional linear approximations remains.
5. **Lack of comprehensive comparison with offline deep Koopman methods**: Comparison with offline methods is limited to a single system.

## Related Work & Insights

| Method | No History Required | Online Updates | Adaptive Embedding | Built-in Reconstruction |
|------|---------|---------|-----------|---------|
| Online DMD | ✓ | ✓ | ✗ | ✓ |
| R-EDMD | ✗ | ✓ | ✗ | ✗ |
| OnlineAE | ✓ | ✓ | ✓ | ✗ |
| DKLT | ✓ | Batch only | ✓ | ✗ |
| **COLoKe** | **✓** | **✓** | **✓** | **✓** |

COLoKe is the only method satisfying all four properties simultaneously. Compared to OnlineAE (fixed-step update strategy), COLoKe achieves principled adaptive updates via the conformal mechanism.

Key transferable insights:
- **New application paradigm for conformal prediction**: Traditionally used for uncertainty quantification and prediction set construction, this paper repurposes it as a model diagnostic/triggering condition — a paradigm transferable to other online learning settings (e.g., online fine-tuning of LLMs, adaptive inference).
- **The "when to update" problem**: A pervasive yet often overlooked issue in continual learning and online adaptation; conformal scores provide an elegant automatic decision mechanism.
- **Integration of Koopman theory and deep learning**: Combining the structured prior of operator theory (linear evolution) with the flexibility of deep learning represents a promising technical direction.

## Rating
- Novelty: ⭐⭐⭐⭐ — Novel perspective of using conformal prediction as a model update trigger.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers synthetic and real-world datasets, including spectral analysis validation and efficiency comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with rigorous mathematical exposition.
- Value: ⭐⭐⭐⭐ — Highly generalizable framework with broad implications for the online learning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Computable Universal Online Learning](computable_universal_online_learning.md)
- [\[ICLR 2026\] Deep FlexQP: Accelerated Nonlinear Programming via Deep Unfolding](../../ICLR2026/learning_theory/deep_flexqp_accelerated_nonlinear_programming_via_deep_unfolding.md)
- [\[NeurIPS 2025\] Prediction-Powered Semi-Supervised Learning with Online Power Tuning](prediction-powered_semi-supervised_learning_with_online_power_tuning.md)
- [\[NeurIPS 2025\] Learning-Augmented Online Bipartite Fractional Matching](learning-augmented_online_bipartite_fractional_matching.md)
- [\[NeurIPS 2025\] Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression](transfer_learning_for_benign_overfitting_in_high-dimensional_linear_regression.md)

</div>

<!-- RELATED:END -->
