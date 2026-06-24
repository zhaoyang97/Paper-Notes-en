---
title: >-
  [Paper Note] Universal 1/3 Time Scaling in Learning Spiked Distributions
description: >-
  [ICML 2026][Interpretability][Neural scaling laws] By analyzing the mathematical properties of softmax and cross-entropy when learning spiked probability distributions, this paper reveals the fundamental cause of the universal 1/3 power-law decay in LLM training loss—an optimization bottleneck at the architectural level independent of data structure.
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Neural scaling laws"
  - "time scaling exponent"
  - "power-law convergence"
  - "softmax-cross entropy"
  - "spiked distribution"
date: 2026-05-08
content_hash: 17036b575bab1ae3
---

# Universal 1/3 Time Scaling in Learning Spiked Distributions

**Conference**: ICML 2026  
**arXiv**: [2602.03685](https://arxiv.org/abs/2602.03685)  
**Code**: https://github.com/liuyz0/TimeScaling  
**Area**: Interpretability / Model Scaling Laws / Optimization Dynamics  
**Keywords**: Neural scaling laws, time scaling exponent, power-law convergence, softmax-cross entropy, spiked distribution

## TL;DR
By analyzing the mathematical properties of softmax and cross-entropy when learning spiked probability distributions, this paper reveals the fundamental cause of the universal 1/3 power-law decay in LLM training loss—an optimization bottleneck at the architectural level independent of data structure.

## Background & Motivation

**Background**: Neural scaling laws (such as the Chinchilla law) empirically observe that LLM training loss decays according to a power law with respect to time or data volume, but the fundamental mechanism underlying this phenomenon remains unclear. Most existing theories attribute this to the power-law structure or feature frequency distribution within the data.

**Limitations of Prior Work**: Why is the exponent close to 0.28 instead of other values? Why does this exponent remain consistent across different model sizes? Such "universality" is difficult to explain solely through data distribution. Furthermore, existing theoretical analyses based on MSE loss fail to capture the specific nonlinearity of softmax and cross-entropy.

**Key Challenge**: On one hand, LLMs must output a "spiked" next-token distribution (low entropy) for accurate prediction; on the other hand, softmax produces power-law decaying loss and gradients at low temperatures (large logit variance). The combination of these two factors forms a fundamental optimization bottleneck.

**Goal**: Identify the architectural rather than data-driven roots of slow LLM training and derive the universal power-law exponent.

**Key Insight**: Starting from the concept of "universality" in statistical physics, the authors utilize a minimal model (single-layer softmax + cross-entropy), solve it exactly using low-temperature expansion, and then verify its applicability to real-world LLMs.

**Core Idea**: Softmax and cross-entropy loss inevitably lead to a $L \sim \tau^{-1/3}$ power-law decay when learning spiked distributions, regardless of the specific data structure.

## Method

### Overall Architecture
The paper aims to answer why the power-law exponent of LLM training loss is approximately 0.28 and consistent across model sizes. It shifts the explanation from "data" to "architecture." The argumentation follows a three-layer approach: "minimal model → theoretical derivation → real LLM verification." First, a teacher-student model with single-layer softmax and cross-entropy is used to isolate interference from data structures, proving it inherently generates power-law loss. Then, low-temperature expansion is used to precisely solve for the exponent. Finally, the theoretical predictions are confirmed on Pythia and OLMo models.

### Key Designs

**1. Aligned Student Ansatz: Reducing high-dimensional nonlinear dynamics to univariate evolution**

Analyzing the full trajectory of student weights during training is a complex high-dimensional nonlinear problem. The authors observe that under zero initialization and small learning rates, student weights quickly align their **direction** with the teacher weights in the early stages; subsequently, training consists almost entirely of growth in the **norm**. Thus, one can assume the student is always aligned with the teacher, reducing the dynamics to the evolution of a single variable—the inverse temperature $\beta$ (the norm of student weights). The gradient flow is written as $\frac{d\beta}{d\tau} = -\frac{c_{\text{eff}}}{n}\frac{dL(\beta)}{d\beta}$. This simplification holds because verification shows that even with misaligned initial directions, the student gradually aligns and enters a $\beta \sim \tau^{1/3}$ growth regime, making the hypothesis robust to initial conditions.

**2. Low-temperature Expansion and Free Energy Analysis: Solving for the universal 1/3 exponent**

With univariate evolution, the key is to determine the form of $L(\beta)$. Borrowing a perspective from statistical physics, the paper treats loss as a combination of free energy and internal energy. It utilizes the low-temperature condition corresponding to "learning spiked distributions," where $\beta \gg c_0 = \sqrt{2\ln n}$ ($n$ is the vocabulary or class size). In this low-temperature region, the free energy is Taylor-expanded in terms of $\beta^{-1}$: $F(\beta) = -c_0 - c_1\beta^{-1} - c_2\beta^{-2} + \cdots$, leading to $L \approx c_2\beta^{-1}$ and $-\frac{dL}{d\beta} \approx c_2\beta^{-2}$. Substituting these into the gradient flow yields $\beta \sim \tau^{1/3}$, and thus $L \sim \tau^{-1/3}$. This step is the source of "universality": the expansion coefficients depend only on the low-temperature condition and not on the specific shape of the energy distribution, so the 1/3 exponent is identical for any spiked distribution.

**3. Exact Mapping of Dynamic Time $\tau$: Unifying learning rate schedules and explaining the Chinchilla exponent discrepancy**

While the theory is formulated via gradient flow, real training uses Adam and complex learning rate (LR) schedules, meaning the step count $t$ is not a clean variable. The paper uses dynamic time $\tau = \int_0^t \eta_{t'}\,dt'$ (the integral of the learning rate over time) as a substitute: training curves under different LR settings overlap on the $(L, \tau)$ plane. This suggests $\tau$ is the fundamental variable governing loss decay. This mapping also explains a long-standing question: the Chinchilla scaling law measures an exponent of approximately 0.28 relative to data volume, which is slightly smaller than the theoretical 1/3. The root cause is the nonlinear relationship between the LR schedule and data volume, rather than a theoretical bias.

## Key Experimental Results

### Main Results: Toy Model Verification

| Inverse Temperature Range | Loss Decay Type | Fitted Exponent | Explanation |
|-----------|-----------|--------|------|
| High Temp (small $\beta^*$) | Exponential | N/A | Non-power law |
| Mid Temp ($c_0 < \beta < \beta^*$) | Power Law | $-1/3$ | Theoretical prediction range |
| Low Temp ($\beta \approx \beta^*$) | Saturation | N/A | Student nearing convergence |

### Ablation Study

| Configuration | Observed Phenomenon | Conclusion |
|------|--------|------|
| Fixed LR, scanning $\beta^*$ | Clearer power law at high $\beta^*$ | Spiked distribution triggers 1/3 scaling |
| Weight decay + low LR | Loss remains $\tau^{-1/3}$ but $\beta$ does not grow | Parameter rotation can also produce power laws |
| Different initialization scales | All align and enter the 1/3 regime | Robustness of the Aligned Student Ansatz |

### Key Findings
- LLM Verification: Fitting $L = \frac{c_\tau}{\tau^{\alpha_\tau}} + L_{\backslash\tau}$ on Pythia yields $\alpha_\tau \approx 0.33 \pm 0.02$ (matching the theoretical 1/3).
- Loss curves for different model sizes overlap under $\tau$ coordinates, demonstrating universality.
- The growth exponent of logit standard deviation is approximately 0.38 ≈ 1/3, verifying the entry into the low-temperature phase.

## Highlights & Insights
- **Architecture as the Root**: Provides the first rigorous proof that softmax and cross-entropy themselves (rather than data) lead to power laws—a universal physical law at the optimization level.
- **Power of Low-temperature Expansion**: Analytically extracts power-law exponents from complex nonlinear systems using Taylor expansion and extreme value distribution theory.
- **Unification of Dynamic and Static Perspectives**: Proves that Chinchilla’s static scaling law actually reflects dynamic time scaling ($\tau^{-1/3}$).

## Limitations & Future Work
- Theoretical Limitations: Assumes gradient flow dynamics and does not consider the effects of finite batch noise or large learning rates; the Aligned Student Ansatz fails when parameters significantly deviate.
- Experimental Limitations: Verified primarily on Pythia/OLMo with limited sample sizes; logits are not strictly i.i.d. Gaussian.
- Future Directions: Design optimizers sensitive to parameter rotation to break the 1/3 bottleneck; explore architectures that reduce logit entropy (e.g., hierarchical vocabularies).

## Related Work & Insights
- **vs. Data Structure Theory (Bordelon et al. 2024)**: This paper proves that softmax + cross-entropy generates power laws even in the absence of power-law data.
- **vs. MSE Loss Theory**: MSE does not generate power-law decay, highlighting the unique role of the softmax layer.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Resolves the causal relationship between softmax-cross entropy and power laws, challenging the consensus that "power laws originate from data."
- Experimental Thoroughness: ⭐⭐⭐⭐ Toy models are comprehensive, though LLM verification could include more scales.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain with strong physical intuition.
- Value: ⭐⭐⭐⭐⭐ Provides a viable direction for improving LLM scaling laws.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Universal Safety Controllers with Learned Prophecies](../../AAAI2026/interpretability/universal_safety_controllers_with_learned_prophecies.md)
- [\[ICLR 2026\] Small Transformers Don't Need LayerNorm at Inference Time: Scaling LayerNorm Removal to GPT-2 XL and Implications for Mechanistic Interpretability](../../ICLR2026/interpretability/small_transformers_dont_need_layernorm_at_inference_time_scaling_layernorm_remov.md)
- [\[ICLR 2026\] Tokenizing Single-Channel EEG with Time-Frequency Motif Learning](../../ICLR2026/interpretability/tokenizing_single-channel_eeg_with_time-frequency_motif_learning.md)
- [\[NeurIPS 2025\] Superposition Yields Robust Neural Scaling](../../NeurIPS2025/interpretability/superposition_yields_robust_neural_scaling.md)
- [\[ICLR 2026\] Universal Properties of Activation Sparsity in Modern Large Language Models](../../ICLR2026/interpretability/universal_properties_of_activation_sparsity_in_modern_large_language_models.md)

</div>

<!-- RELATED:END -->
