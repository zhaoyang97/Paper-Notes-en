---
title: >-
  [Paper Note] Understanding Sharpness Dynamics in NN Training with a Minimalist Example: The Effects of Dataset Difficulty, Depth, Stochasticity, and More
description: >-
  [ICML2025][Optimization][sharpness dynamics] Proposed a minimalist model using a "deep linear network with a single neuron per layer" to systematically study progressive sharpening and edge of stability phenomena. The concept of dataset difficulty $Q$ is introduced, and both upper and lower bounds of sharpness at global optima are derived. The theoretical analysis uncovers the impact mechanisms of dataset size, network depth, batch size, and learning rate on sharpness dynamic…
tags:
  - "ICML2025"
  - "Optimization"
  - "sharpness dynamics"
  - "progressive sharpening"
  - "edge of stability"
  - "deep linear networks"
  - "dataset difficulty"
date: 2026-05-08
content_hash: 8bc2309b850c4af2
---

# Understanding Sharpness Dynamics in NN Training with a Minimalist Example: The Effects of Dataset Difficulty, Depth, Stochasticity, and More

**Conference**: ICML2025  
**arXiv**: [2506.06940](https://arxiv.org/abs/2506.06940)  
**Code**: Not provided  
**Area**: Optimization  
**Keywords**: sharpness dynamics, progressive sharpening, edge of stability, deep linear networks, dataset difficulty

## TL;DR

Proposed a minimalist model using a "deep linear network with a single neuron per layer" to systematically study progressive sharpening and edge of stability phenomena. The concept of dataset difficulty $Q$ is introduced, and both upper and lower bounds of sharpness at global optima are derived. The theoretical analysis uncovers the impact mechanisms of dataset size, network depth, batch size, and learning rate on sharpness dynamics.

## Background & Motivation

- **Progressive sharpening**: When training deep networks with GD, the maximum eigenvalue of the loss Hessian (sharpness) progressively increases and eventually oscillates around $2/\eta$ (edge of stability, EoS).
- Cohen et al. (2021) showed through extensive empirical studies that sharpness dynamics are influenced by dataset size, network depth, batch size, and learning rate, but the underlying mechanism lacks theoretical explanations.
- Prior theoretical works (Wang et al. 2022, Agarwala et al. 2023, Marion & Chizat 2024) either only provide upper bounds, are restricted to specific regimes, or are limited to synthetic data, failing to fully quantify the effects of these factors.
- **Core Motivation**: To construct a minimalist model that is simple enough for rigorous analysis while faithfully reproducing the empirical sharpness behaviors observed in actual training.

## Method

### Minimalist Model Design

A deep linear network with a single neuron per layer (depth $D \geq 2$):

$$f(x;\theta) = (x^\top u) \prod_{i=1}^{D-1} v_i$$

Where $u \in \mathbb{R}^d$ represents the weights of the first layer, $v_i \in \mathbb{R}$ represents the scalar weights of the subsequent layers, yielding a total parameter size of $p = d + D - 1$. The MSE loss is employed:

$$L(\theta) = \frac{1}{2N} \|Xu \prod_{i=1}^{D-1} v_i - y\|^2$$

Sharpness is defined as $S(\theta) = \lambda_{\max}(\nabla^2 L(\theta))$.

### Core Concept: Dataset Difficulty

Given the SVD of the data matrix $X = \sum_{i=1}^r \sigma_i e_i w_i^\top$, and the projection coefficients of the labels on the left singular vectors $d_i = e_i^\top y$, dataset difficulty is defined as:

$$Q := \sum_{i=1}^{r} \frac{d_i^2}{\sigma_i^2}$$

Intuitively, $Q$ characterizes the "total distance" required for the model to perfectly fit the data. The larger the label component $d_i$ to fit in each direction, and the smaller the corresponding singular value $\sigma_i$, the more difficult the fitting process. $Q$ depends solely on the dataset and is independent of the architecture or optimizer.

### Derivation of Sharpness Bounds

**Two-layer case ($D=2$)**: Defining the layer imbalance as $C(\theta) = \|\Pi_W u\|^2 - v_1^2$, the following holds at the global optimum $\theta^\star$:

$$\frac{1}{N}\left[\sigma_1^2 (v_1^\star)^2 + \frac{d_1^2}{(v_1^\star)^2}\right] \leq S(\theta^\star) \leq \frac{1}{N}\left[\sigma_1^2 (v_1^\star)^2 + \frac{\sum_i d_i^2}{(v_1^\star)^2}\right]$$

Where $(v_1^\star)^2 = \frac{\sqrt{C(\theta^\star)^2 + 4Q} - C(\theta^\star)}{2}$. The sharpness increases with $Q$ and decreases with $C(\theta^\star)$.

**General depth ($D \geq 2$, under balanced condition)**:

$$\frac{1}{N}\left[\sigma_1^2 Q^{\frac{D-1}{D}} + (D-1) d_1^2 Q^{-\frac{1}{D}}\right] \leq S(\theta^\star) \leq \frac{1}{N}\left[\sigma_1^2 Q^{\frac{D-1}{D}} + (D-1) \sum_i d_i^2 \cdot Q^{-\frac{1}{D}}\right]$$

The dominant term is defined as the **predicted sharpness**: $\hat{S}_D = \frac{\sigma_1^2}{N} Q^{\frac{D-1}{D}}$.

### Effect of Optimizers on Sharpness

- **Gradient Flow**: The layer imbalance $C(\theta(t))$ is a conserved quantity (for $D=2$) or remains balanced (for $D>2$); hence, the converged sharpness can be directly predicted from the initialization.
- **GD/SGD**: $C$ increases progressively. Theorem 5.9 provides the exact formula for the increment of $C$ after one step of GD and SGD, proving that SGD increases $C$ more than GD ($\Psi_2 \geq \Psi_1$, $\Omega_2 \geq \Omega_1$). This additional increment is proportional to $\frac{N-B}{B}$—smaller batch sizes and larger learning rates make $C$ grow faster, leading to smaller final sharpness (i.e., reduced progressive sharpening).

## Key Experimental Results

### Minimalist Model Reproducing Phenomenon 1

| Factor | Effect | Experimental Setup |
|------|------|----------|
| Dataset size ↑ | sharpness ↑ | CIFAR10 2-label, $N$ from 100 to 1000 |
| Network depth ↑ | sharpness ↑ | $D$ from 2 to 5 |
| Batch size ↑ | sharpness ↑ | SGD, $B$ from 10 to $N$ |
| Learning rate ↑ (small $B$) | sharpness ↓ | SGD, $\eta$ from 0.01 to 0.3 |

### Numerical Validation of Dataset Difficulty $Q$

| Dataset | $N=100$ | $N=300$ | $N=1000$ |
|--------|---------|---------|----------|
| CIFAR10 | 0.22 | 1.70 | 44.44 |
| SVHN | 1.16 | 21.13 | 859.4 |
| Google Speech | 0.26 | 1.67 | 26.34 |

$Q$ increases sharply as $N$ grows, echoing the trend of progressive sharpening.

### Correlation between Predicted Sharpness and Actual Sharpness

- 5-layer linear network (width 2048) on CIFAR10: The correlation coefficient between $\hat{S}_D$ and $S(\theta(\infty))$ reaches **0.99**.
- 4-layer tanh network (width 1024) on SVHN: The correlation coefficient is **0.81**.
- Predictions remain effective even in scenarios deviating from theoretical assumptions, such as non-linear models, wide networks, and unbalanced initialization.

### Reproduction of Edge of Stability

The minimalist model successfully reproduces: (1) sharpness rising to $2/\eta$ followed by oscillation; (2) non-monotonic loss decay (including spikes); (3) oscillation amplitude decaying over time, which most minimalist models in prior works fail to reproduce simultaneously.

## Highlights & Insights

- **dataset difficulty $Q$**: A purely data-dependent scalar that uniformly quantifies the impact of dataset size on sharpness, showing strong predictive power even in actual non-linear networks.
- **High fidelity of the minimalist model**: With only $d + D - 1$ parameters, it successfully reproduces representative characteristics of actual training such as progressive sharpening, EoS, loss spikes, and decaying oscillations.
- **Analysis of layer imbalance $C$**: Discloses the quantitative mechanism where SGD noise suppresses sharpness growth by increasing $C$, and the reduction of batch size / learning rate effects can be attributed to the regulation of the increment of $C$.
- **Two-sided bounds**: Jointly derives both upper and lower sharpness bounds (while prior studies mostly focus on upper bounds), with the numerical upper and lower bounds being very tight.

## Limitations & Future Work

- Theoretical analysis is restricted to **linear activations** and **one neuron per layer**. Although experiments show some transferability to non-linear networks, rigorous guarantees are lacking.
- Theoretical analysis of the decaying oscillation during the EoS phase remains incomplete, and the authors observe that this behavior is highly correlated with **numerical precision** (high precision may lead to loss explosion).
- Behaviors under **cross-entropy loss** are not discussed (as margin maximization causes sharpness to decrease in late stages, making direct comparison difficult).
- A unified explanation for the impact of **network width** is not provided (with inconsistent trends under MSE and CE losses).
- The SGD analysis is limited to $D=2$; the evolution of $C$ in deep SGD setups requires further extension.

## Related Work & Insights

- **Cohen et al. (2021)**: Formalized the empirical benchmark for progressive sharpening and EoS, which this work builds upon theoretically.
- **Damian et al. (2023)**: Explained EoS via a self-stabilization mechanism, assuming progressive sharpening has already occurred.
- **Marion & Chizat (2024)**: Analyzed sharpness upper bounds in deep linear networks; this work provides tighter two-sided bounds.
- **Wang et al. (2022)**: Substituted sharpness using the output-layer norm, which is limited to specific intervals.
- **Agarwala & Pennington (2024)**: Analyzed the sharpness of SGD in quadratic regression models; this work achieves similar conclusions under more general settings.

## Rating

- Novelty: ⭐⭐⭐⭐ — The concept of dataset difficulty is novel, and the high fidelity of the minimalist model is impressive.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Systematically validated across multiple datasets, architectures, and optimizers, with convincing scatter plot correlation analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ — Well-organized, theories and experiments are well-interleaved, and the charts are concise.
- Value: ⭐⭐⭐⭐ — Provides practical theoretical tools and a clean analytical framework for understanding sharpness dynamics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Training Dynamics of In-Context Learning in Linear Attention](training_dynamics_of_in-context_learning_in_linear_attention.md)
- [\[ICLR 2026\] Towards Understanding the Calibration Benefits of Sharpness-Aware Minimization](../../ICLR2026/optimization/towards_understanding_the_calibration_benefits_of_sharpness-aware_minimization.md)
- [\[ICML 2025\] Tilted Sharpness-Aware Minimization](tilted_sharpness-aware_minimization.md)
- [\[ICML 2025\] How Transformers Learn Regular Language Recognition: A Theoretical Study on Training Dynamics and Implicit Bias](how_transformers_learn_regular_language_recognition_a_theoretical_study_on_train.md)
- [\[ICML 2025\] Understanding Mode Connectivity via Parameter Space Symmetry](understanding_mode_connectivity_via_parameter_space_symmetry.md)

</div>

<!-- RELATED:END -->
