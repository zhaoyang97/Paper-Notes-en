---
title: >-
  [Paper Note] Universal 1/3 Time Scaling in Learning Peaked Distributions
description: >-
  [ICML 2026][Interpretability][Neural Scaling Laws] By analyzing the mathematical properties of softmax and cross-entropy when learning peaked probability distributions…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Neural Scaling Laws"
  - "Time Scaling Exponent"
  - "Power-law Convergence"
  - "Softmax-Cross Entropy"
  - "Peaked Distribution"
date: 2026-05-08
content_hash: 24e64fca6f476a1e
---

# Universal 1/3 Time Scaling in Learning Peaked Distributions

**Conference**: ICML 2026  
**arXiv**: [2602.03685](https://arxiv.org/abs/2602.03685)  
**Code**: https://github.com/liuyz0/TimeScaling  
**Area**: Interpretability / Model Scaling Laws / Optimization Dynamics  
**Keywords**: Neural Scaling Laws, Time Scaling Exponent, Power-law Convergence, Softmax-Cross Entropy, Peaked Distribution

## TL;DR
By analyzing the mathematical properties of softmax and cross-entropy when learning peaked probability distributions, the paper reveals the fundamental reason for the universal $1/3$ power-law decay in LLM training loss—an architectural-level optimization bottleneck independent of data structure.

## Background & Motivation

**Background**: Neural scaling laws (e.g., Chinchilla's law) empirically observe that LLM training loss decays in a power law with time/data volume, but the fundamental mechanism remains unclear. Existing theories mostly attribute this to power-law structures in data or feature frequency distributions.

**Limitations of Prior Work**: Why is the exponent close to $0.28$ rather than other values? Why does this exponent remain consistent across different model sizes? It is difficult to explain such "universality" solely from data distribution. Additionally, existing theoretical analyses of MSE loss cannot capture the specific nonlinearity of softmax and cross-entropy.

**Key Challenge**: On one hand, LLMs must output a "peaked" next-token distribution (low entropy) for accurate prediction; on the other hand, softmax produces power-law decaying loss and gradients at low temperatures (large logit variance)—the combination of these two forms a fundamental optimization bottleneck.

**Goal**: To identify the architectural root of slow LLM training rather than the data root, and derive the universal power-law exponent.

**Key Insight**: Starting from the concept of "universality" in statistical physics, a minimal model (single-layer softmax + cross-entropy) is solved exactly via low-temperature expansion, and its applicability is then verified on actual LLMs.

**Core Idea**: Softmax and cross-entropy loss inevitably lead to a power-law decay of $L \sim \tau^{-1/3}$ when learning peaked distributions, independent of the specific data structure.

## Method

### Overall Architecture
A three-tier progressive architecture: "Minimal Model $\rightarrow$ Theoretical Analysis $\rightarrow$ LLM Verification." First, a teacher-student model is constructed to prove that softmax + cross-entropy generates power laws; second, the power-law exponent is derived exactly using low-temperature expansion; finally, predictions are verified on real models like Pythia/OLMo.

### Key Designs

1.  **Aligned Student Ansatz**:
    - **Function**: Through observing weight evolution trajectories, it is assumed that student weights always align with the direction of teacher weights, with only norm changes.
    - **Mechanism**: Due to zero initialization and small learning rates, student weights can rapidly align in direction during early stages and subsequently focus on norm growth. This simplifies complex nonlinear dynamics into the evolution of a single variable $\beta$ (inverse temperature): $\frac{d\beta}{d\tau} = -\frac{c_{\text{eff}}}{n}\frac{dL(\beta)}{d\beta}$.
    - **Design Motivation**: Verifications show that even with initial misalignment, the student gradually aligns and eventually enters the $\beta \sim \tau^{1/3}$ regime.

2.  **Low-Temperature Expansion and Free Energy Analysis**:
    - **Function**: Under peaked distributions ($\beta \gg c_0 = \sqrt{2\ln n}$), use Taylor series to expand free energy $F(\beta)$ and internal energy $U(\beta)$.
    - **Mechanism**: $F(\beta) = -c_0 - c_1\beta^{-1} - c_2\beta^{-2} + \cdots$, leading to $L \approx c_2\beta^{-1}$ and $-\frac{dL}{d\beta} \approx c_2\beta^{-2}$. Substituting into gradient flow yields $\beta \sim \tau^{1/3}$, hence $L \sim \tau^{-1/3}$.
    - **Design Motivation**: This is the root of universality—the expansion coefficients are independent of the energy distribution form, requiring only the low-temperature condition, so the exponent $1/3$ holds for all peaked distributions.

3.  **Precise Mapping of Dynamic Time $\tau$**:
    - **Function**: Unify the complex learning rate schedules of adaptive optimizers like Adam into a single dynamic time $\tau = \int_0^t \eta_{t'} dt'$.
    - **Mechanism**: Training curves with different learning rates coincide on the $(L, \tau)$ plane, indicating that $\tau$ rather than step count $t$ is the fundamental variable.
    - **Design Motivation**: Explains why the Chinchilla scaling law exponent (using data volume) is slightly smaller than $1/3$ ($0.28$); the root cause is the nonlinear relationship between learning rate scheduling and data volume.

## Key Experimental Results

### Main Results: Toy Model Verification

| Inverse Temperature Range | Loss Decay Type | Fitted Exponent | Description |
| :--- | :--- | :--- | :--- |
| High Temperature (Small $\beta^*$) | Exponential Decay | N/A | Non-power law |
| Mid Temperature ($c_0 < \beta < \beta^*$) | Power Law | $-1/3$ | Theoretical prediction interval |
| Low Temperature ($\beta \approx \beta^*$) | Saturation | N/A | Student near convergence |

### Ablation Study

| Configuration | Observation | Conclusion |
| :--- | :--- | :--- |
| Fixed LR, scan $\beta^*$ | Power law more obvious at high $\beta^*$ | Peaked distribution triggers $1/3$ scaling |
| Weight decay + low LR | Loss still $\tau^{-1/3}$ but $\beta$ stops growing | Parameter rotation can also produce power laws |
| Different initial ratios | All align and enter $1/3$ regime | Robustness of Aligned Student Ansatz |

### Key Findings
- LLM Verification: Fitting $L = \frac{c_\tau}{\tau^{\alpha_\tau}} + L_{\backslash\tau}$ on Pythia yields $\alpha_\tau \approx 0.33 \pm 0.02$ (theoretical value $1/3$).
- Curves of different model sizes coincide under the $\tau$ coordinate, demonstrating universality.
- The growth exponent of logit standard deviation is $0.38 \approx 1/3$, verifying the low-temperature entry phase.

## Highlights & Insights
- **Architecture as Root**: First rigorous proof that softmax + cross-entropy itself (rather than data) leads to power laws—a universal physical law at the optimization level.
- **Power of Low-Temperature Expansion**: Analytically extracting power-law exponents from complex nonlinear systems through Taylor expansion and extreme value distribution theory.
- **Unification of Dynamic vs. Static Perspectives**: Proving that Chinchilla's static scaling law actually reflects dynamic time scaling ($\tau^{-1/3}$).

## Limitations & Future Work
- Theoretical limitations—Assumes gradient flow dynamics, does not account for finite mini-batch noise and large learning rate effects; the Aligned Student Ansatz fails if parameters deviate significantly.
- Experimental limitations—Verified only on Pythia/OLMo with limited sample sizes; logits are not strictly i.i.d. Gaussian.
- Future directions: Designing optimizers sensitive to parameter rotation to break the $1/3$ bottleneck; exploring architectures that reduce logit entropy (e.g., hierarchical vocabulary).

## Related Work & Insights
- **vs. Data Structure Theory (Bordelon et al. 2024)**: This paper proves that softmax + cross-entropy generates power laws even without power-law data.
- **vs. MSE Loss Theory**: MSE does not produce power laws, highlighting the special role of softmax.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to reveal the causal relationship between softmax-cross entropy and power laws, overturning the consensus that "power laws originate from data."
- Experimental Thoroughness: ⭐⭐⭐⭐ Toy models are complete, but LLM verification lacks more scales.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain with sound physical intuition.
- Value: ⭐⭐⭐⭐⭐ Provides a feasible direction for improving LLM scaling laws.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Tokenizing Single-Channel EEG with Time-Frequency Motif Learning](../../ICLR2026/interpretability/tokenizing_single-channel_eeg_with_time-frequency_motif_learning.md)
- [\[AAAI 2026\] Universal Safety Controllers with Learned Prophecies](../../AAAI2026/interpretability/universal_safety_controllers_with_learned_prophecies.md)
- [\[ICML 2026\] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)
- [\[NeurIPS 2025\] Towards Scaling Laws for Symbolic Regression](../../NeurIPS2025/interpretability/towards_scaling_laws_for_symbolic_regression.md)
- [\[ICML 2026\] Physics from Video: Identifiability of Time-Invariant Second-Order ODEs under Minimal Trajectory Conditions](physics_from_video_identifiability_of_time-invariant_second-order_odes_under_min.md)

</div>

<!-- RELATED:END -->
