---
title: >-
  [Paper Note] Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules
description: >-
  [NeurIPS 2025 Spotlight][Optimization][scaling law] This work establishes Functional Scaling Laws (FSL) in power-law kernel regression models. By introducing the concept of "intrinsic time," it uniformly characterizes the full loss trajectory under any learning rate schedule. Explicit scaling relationships under constant, exponential decay, and WSD schedules are derived in both data-limited and compute-limited regimes, theoretically explaining the empirical superiority of WSD…
tags:
  - "NeurIPS 2025 Spotlight"
  - "Optimization"
  - "scaling law"
  - "learning rate schedule"
  - "kernel regression"
  - "loss dynamics"
  - "WSD schedule"
date: 2026-05-08
content_hash: de8ebdcac00ed5ed
---

# Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2509.19189](https://arxiv.org/abs/2509.19189)  
**Code**: None  
**Area**: Optimization  
**Keywords**: scaling law, learning rate schedule, kernel regression, loss dynamics, WSD schedule

## TL;DR
This work establishes Functional Scaling Laws (FSL) in power-law kernel regression models. By introducing the concept of "intrinsic time," it uniformly characterizes the full loss trajectory under any learning rate schedule. Explicit scaling relationships under constant, exponential decay, and WSD schedules are derived in both data-limited and compute-limited regimes, theoretically explaining the empirical superiority of WSD over pure decay.

## Background & Motivation

**Background**: Kaplan et al. discovered that LLM pre-training loss follows a power-law relationship of $L(M,D) = L_0 + C_M M^{-\alpha_M} + C_D D^{-\alpha_D}$, making scaling laws a foundational principle for guiding LLM development. Theoretical explanations are primarily conducted on kernel/linear regression proxy models.

**Limitations of Prior Work**:
   - Existing theories only target final loss, leaving unexplained whether the full loss trajectory also follows scaling laws.
   - The role of the learning rate schedule (LRS) has not been systematically characterized—while gradient flow, constant-LR SGD, and exponential decay have been studied individually, a unified theory is lacking.
   - WSD (Warmup-Stable-Decay) schedules perform exceptionally well in practice (e.g., DeepSeek-V3, Kimi-K2), but the mechanism behind their advantage remains unclear.

**Key Challenge**: LRS simultaneously affects signal learning and noise injection/dissipation. Different scheduling strategies trade off these two effects differently, requiring a unified framework.

**Key Insight**: Introduce the concept of "intrinsic time" $t = \int_0^\tau \varphi(r) dr$ (cumulative learning rate) to decouple the influence of the LRS from the drift term of the SDE, retaining it only in the diffusion term.

**Core Idea**: FSL decomposes the loss into irreducible error + approximation error + signal learning term + noise convolution term, where the LRS enters only through the convolutional functional in the last term.

## Method

### Overall Architecture
Consider a power-law kernel (PLK) regression model: $y = ⟨\boldsymbol{\phi}(\mathbf{x}), \boldsymbol{\theta}^*⟩ + \epsilon$, with a feature covariance spectrum $\lambda_j \asymp j^{-\beta}$ (capacity exponent $\beta > 1$) and target coefficient decay $|\theta_j^*|^2 \asymp j^{-1}\lambda_j^{s-1}$ (difficulty exponent $s > 0$). The model is optimized using a model of width $M$ via one-pass SGD.

### Key Designs

1. **Intrinsic Time Reparameterization**:

    - **Function**: Rescale physical time $\tau$ into intrinsic time $t$ according to the LRS.
    - **Mechanism**: Define $t = T(\tau) = \int_0^\tau \varphi(r) dr$. After change of variables, the original SDE $d\bar{\mathbf{v}}_\tau = -\varphi(\tau)\nabla\mathcal{R} d\tau + \varphi\sqrt{h/b \cdot \Sigma} d\mathbf{B}_\tau$ becomes $d\boldsymbol{\nu}_t = -\nabla\mathcal{R} dt + \sqrt{\gamma(t)\Sigma} d\mathbf{B}_t$, where $\gamma(t) = h/(\varphi \cdot b)$ under intrinsic time. The LRS completely vanishes from the drift term.
    - **Design Motivation**: Intrinsic time conveys training progress more faithfully than the number of iteration steps, making the signal learning component independent of the LRS.

2. **Functional Scaling Law (FSL)**:

    - **Function**: Characterize the full loss trajectory instead of only the final loss.
    - **Mechanism**:
    $$\mathbb{E}[\mathcal{R}(\boldsymbol{\nu}_t)] - \frac{\sigma^2}{2} \asymp M^{-s\beta} + e(t) + \int_0^t \mathcal{K}(t-z)[e(z)+\sigma^2]\gamma(z) dz$$
   where $e(t) = (1+t)^{-s}$ represents the signal learning term, and $\mathcal{K}(t) = (1+t)^{-(2-1/\beta)}$ is the memory/forgetting kernel.
    - **Design Motivation**: The four terms correspond to irreducible error, approximation error, signal learning, and noise accumulation/dissipation, respectively. The LRS only enters the last term through the convolution.

3. **Noise Structure Analysis (Lemma 4.8)**:

    - **Function**: Precisely characterize the anisotropic structure of gradient noise.
    - **Mechanism**: $(2\rho_- \mathcal{E}(\mathbf{v}) + \sigma^2)\nabla^2\mathcal{R} \preceq \Sigma(\mathbf{v}) \preceq (2\rho_+ \mathcal{E}(\mathbf{v}) + \sigma^2)\nabla^2\mathcal{R}$. The noise energy along each direction is proportional to risk $\times$ the curvature in that direction.
    - **Design Motivation**: This structure allows the noise term to be bounded by the loss itself, forming a self-consistent analysis.

4. **Explicit Scaling of Three LRSs**:

    - **Function**: Derive closed-form scaling relations for constant, exponential decay, and WSD schedules.
    - **Mechanism**: Substitute the LRS into the convolution term of the FSL, and calculate the optimal allocation under data-limited (fixed $M$) and compute-limited (jointly optimizing $M$ and $D$) regimes, respectively.
    - **Design Motivation**: Quantify the difference in scaling efficiency across different LRSs to explain the empirical advantage of WSD.

### Loss & Training
- Model capacity $\beta > 1$, task difficulty $s > 0$. Relative difficulty distinction: easy ($s \geq 1-1/\beta$) vs hard ($s < 1-1/\beta$).
- The hypercontractivity assumption (Assumption 2.1) ensures that fourth-order moments are controlled by second-order moments.
- Supports two settings: top-$M$ feature selection and random-$M$ feature projection.

## Key Experimental Results

### Comparison of Data-Optimal Scaling for Three LRSs

| LRS | Easy regime | Hard regime |
|-----|------------|------------|
| Constant | $D^{-s/(s+1)}$ | $D^{-s/(s+1)}$ |
| Exp-decay | $D^{-s\beta/(1+s\beta)}(\log D)^{s\beta/(1+s\beta)}$ | $D^{-s}(\log D)^s$ |
| WSD | $D^{-s\beta/(1+s\beta)}(\log D)^{(s\beta-s)/(1+s\beta)}$ | $D^{-s}$ |

### Comparison of Compute-Optimal Scaling

| LRS | Easy regime | Hard regime |
|-----|------------|------------|
| Constant | — | $C^{-s\beta/(1+s\beta+\beta)}$ |
| Exp-decay | $C^{-s\beta/(2+s\beta)}(\log C)^{...}$ | $C^{-s\beta/(1+\beta)}(\log C)^{...}$ |
| WSD | $C^{-s\beta/(2+s\beta)}(\log C)^{...}$ | $C^{-s\beta/(1+\beta)}$ |

### Key Findings
- **WSD > Exp-decay > Constant**: WSD removes the $\log$ factor in the hard regime, while the Constant schedule lacks an additional $\beta$ factor.
- **High-capacity models are more efficient**: For a fixed task (constant $\alpha = s\beta$), when $\beta$ increases (reducing capacity), $s$ is larger, leading to faster signal learning.
- **Data should scale more than model size**: In compute-optimal training, data should scale more aggressively than the model size.
- **Peak learning rate should match the budget**: The optimal peak LR should scale appropriately with the training budget.
- LLM experiments (0.1B-1B) validate that FSL can serve as a proxy model for fitting and predicting loss trajectories.

## Highlights & Insights
- **Concept of Intrinsic Time**: Completely decoupling the influence of the LRS from signal learning is the key innovation for achieving a unified analysis. Intuitively, this concept corresponds to the "effective training steps".
- **Physical Intuition of the Forgetting Kernel**: $\mathcal{K}(t) \asymp t^{-(2-1/\beta)}$ indicates that high-capacity models (small $\beta$) forget injected noise more slowly, explaining why larger models require more delicate LRS design.
- **Mechanism of WSD's Advantage**: The long stable phase accumulates sufficient signal learning, while the decaying phase at the absolute end efficiently dissipates accumulated noise; the two phases complement each other.
- **Connection to Multi-Power-Law Models**: Via integration by parts, FSL can be transformed to be approximately equivalent to the empirical MPL model of Tissue et al.

## Limitations & Future Work
- The analysis is based on kernel regression (quadratic loss), which has a gap with the cross-entropy loss of actual LLMs.
- The continuous-time SDE approximation requires a sufficiently small step size, which might not hold in practical training.
- The analysis of random-$M$ features only covers $s \leq 1$, leaving the cases for easier tasks where $s > 1$ unresolved.
- The precision of the $\eqsim$ (constant factor) in FSL might be insufficient for practical fitting, requiring additional fitting parameters in LLM experiments.

## Related Work & Insights
- **vs Bordelon & Pehlevan (2024)**: They only analyzed gradient flow, whereas ours covers SGD with any LRS.
- **vs Sorscher et al. (2024) / Paquette et al.**: They analyzed constant-LRS SGD, whereas ours unifies and extends it.
- **vs Lin et al. (2024)**: They analyzed exponential decay LRS, whereas ours further covers WSD.
- **vs Tissue et al. (2024)**: They proposed empirical MPL models, whereas ours provides theoretical explanations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The concept of intrinsic time and the FSL framework are significant theoretical innovations, unifying scattered prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐ Numerical validation on kernel regression is thorough, and LLM experiments (0.1B-1B) enhance practical relevance.
- Writing Quality: ⭐⭐⭐⭐⭐ The physical interpretation of each term in the FSL is exceptionally clear, building an excellent bridge between theory and practice.
- Value: ⭐⭐⭐⭐⭐ It provides a solid theoretical foundation for understanding and designing LLM learning rate schedules.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](../../ICLR2026/optimization/convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)
- [\[NeurIPS 2025\] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[ICML 2026\] Muon in Associative Memory Learning: Training Dynamics and Scaling Laws](../../ICML2026/optimization/muon_in_associative_memory_learning_training_dynamics_and_scaling_laws.md)
- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](learning_theory_for_kernel_bilevel_optimization.md)

</div>

<!-- RELATED:END -->
