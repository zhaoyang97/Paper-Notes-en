---
title: >-
  [Paper Note] Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?
description: >-
  [ICLR 2026][Optimization & Theory][SignSGD] This work systematically analyzes the scaling laws of SignSGD under the Power-Law Random Features (PLRF) model, revealing two unique effects—drift-normalization and noise-reshaping—and proving that SignSGD's compute-optimal slope can exceed that of SGD in noise-dominated scenarios.
tags:
  - ICLR 2026
  - Optimization & Theory
  - SignSGD
date: 2026-05-08
content_hash: 02c07f5e528df89a
---
# Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?

**Conference**: ICLR 2026  
**arXiv**: [2603.02069](https://arxiv.org/abs/2603.02069)  
**Code**: None  
**Area**: Optimization Theory  
**Keywords**: SignSGD, Scaling Laws, Linear Regression, Random Features, Learning Rate Scheduling

## TL;DR

This work systematically analyzes the scaling laws of SignSGD under the Power-Law Random Features (PLRF) model, revealing two unique effects—drift-normalization and noise-reshaping—and proving that SignSGD's compute-optimal slope can exceed that of SGD in noise-dominated scenarios.

## Background & Motivation

SignSGD is a core component of adaptive optimizers like Adam, using only the sign of the gradient for parameter updates. While Adam/AdamW is the de facto standard in large-scale language model training, theoretical understanding of why and when SignSGD outperforms SGD remains limited.

**Limitations of Prior Work**:
1. Most traditional SignSGD analyses focus on convex optimization or simple settings, failing to account for scaling law phenomena observed in modern deep learning.
2. Paquette et al. (2024) analyzed scaling laws for SGD in PLRF models but did not cover SignSGD.
3. There is a lack of precise characterization of conditions under which SignSGD outperforms SGD.

**Core Problem**:
- How does the population risk of SignSGD scale with model size, training steps, and learning rate?
- Under compute-optimal configurations, when do the scaling behaviors of SignSGD and SGD diverge?
- What unique impact does the warmup-stable-decay (WSD) learning rate schedule have on SignSGD?

The authors choose the Power-Law Random Features model as the analytical framework because it captures two key dimensions—feature decay and target decay—which are central to understanding scaling laws.

## Method

### Overall Architecture

The analysis is built on the analytically tractable Power-Law Random Features (PLRF) toy model. Specifically, a linear model acts on random sketched features $S\mathbf{x}$ to fit a target. The eigenvalues of the data feature $\mathbf{x}$ covariance decay as $i^{-2\alpha}$ ($\alpha$ is the feature decay index), and the optimal solution coefficients along each feature direction decay as $i^{-\beta}$ ($\beta$ is the target decay index). Optimization is performed using one-pass SignSGD, with generalization measured by population risk $R(M,N,\gamma_0)$, where $M$ is model size, $N$ is training steps, and $\gamma_0$ is the learning rate. This setup allows characterizing modern scaling laws using only two scalars, $\alpha$ and $\beta$, representing "how easy features are to learn" and "how complex the target is," thereby compressing the scaling behavior of SignSGD into an analytical function of $(M,N,\gamma_0,\alpha,\beta)$ for direct comparison with known SGD results. The analytical chain involves: converting non-linear sign updates into solvable ODEs to obtain a four-term closed-form risk formula; comparing this with SGD formulas to isolate SignSGD-specific effects; solving for compute-optimal configurations to determine when it outperforms SGD; and incorporating WSD scheduling to evaluate its impact on scaling slopes.

### Key Designs

**1. Four-term closed-form risk scaling formula: Analytical treatment of sign updates**

The SignSGD update $\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \gamma_k\,\mathrm{sign}(\mathbf{g}_k)$ is highly non-linear due to the sign operator. The authors use a second-order Taylor expansion on the quadratic objective combined with sign–Gaussian identities to represent the update in each mode as "drift + quadratic noise." This is transformed into a continuous-time ODE and solved using deterministic equivalents, yielding an asymptotic formula for population risk:

$$R(M,N,\gamma_0) \asymp \underbrace{A(M)}_{\text{Approximation Error}} + \underbrace{D^{\text{sign}}_{\text{al}}}_{\text{Aligned Feature Loss}} + \underbrace{D^{\text{sign}}_{\text{dis}}}_{\text{Distorted Feature Loss}} + \underbrace{N^{\text{sign}}}_{\text{Noise Term}}.$$

These four terms correspond to: approximation error $A(M)\asymp M^{-2\alpha+\max(0,1-2\beta)}$ determined by capacity; aligned/distorted feature losses $D^{\text{sign}}_{\text{al}}, D^{\text{sign}}_{\text{dis}}$ from drift decay; and quadratic noise $N^{\text{sign}}=\gamma_0^2 M^{2-\min(1,2\alpha)}$ from the Taylor expansion. This structure allows direct term-by-term comparison with SGD scaling.

**2. Two unique effects: Drift-normalization and noise-reshaping**

Comparing the formula with SGD shows that the sign operator modifies the structure in two places. The first is **drift-normalization**: the SignSGD drift term is $\frac{4\gamma_k}{\pi\sqrt{L(k)}}\lambda_i$, containing a $1/\sqrt{L(k)}$ self-normalization and a $M^{\min(\alpha,1/2)}$ diagonal preconditioning factor. This changes the effective "flow time" from $N\gamma_0$ to $\gamma_0\int_0^N L(u)^{-1/2}\,\mathrm{d}u$, accelerating convergence when $L\lesssim 1$ and resulting in a strictly larger $N$-exponent for drift terms compared to SGD. The second is **noise-reshaping**: the quadratic noise term in SignSGD lacks the multiplicative $L(k)$ factor found in SGD, meaning noise no longer decays with $N$ and lacks the $(N\gamma_0)^{-(4\alpha-1)/(2\alpha)}$ structure. It also gains additional $M$-dependence from working in the $\overline{K}$ feature basis. Noise-reshaping is the primary mechanism allowing SignSGD to overcome SGD's "noise bottleneck."

**3. Compute-optimal scaling laws: Optimal $M$ and $N$ allocation**

With fixed total compute $f = MN$, the learning rate is expressed as $\gamma_0 = M^{-e}$ and model size as $M=f^x$. Solving for $(e,x)$ to minimize risk yields the compute-optimal slope $\eta(\alpha,\beta)$ ($R\asymp f^{-\eta}$). Similar to Chinchilla laws, this is derived for SignSGD for the first time. Results show that in the SGD noise-bottleneck phase (Phase III–IV), SignSGD's compute-optimal slope is steeper and its optimal model size is larger; in other phases, their performance is comparable. A practical finding: SignSGD's optimal learning rate exponent $e^*$ is always larger than SGD's, meaning SignSGD requires smaller learning rates to balance noise suppression and drift maintenance.

**4. WSD scheduling analysis: Efficiency of warmup-stable-decay with SignSGD**

WSD (warmup-stable-decay) is a standard LLM training schedule currently lacking theoretical grounding. By incorporating these three stages into the analytical risk formula, the authors find that the "stable" phase maintains drift speed while the "decay" phase suppresses late-stage noise, improving the noise upper bound relative to a constant learning rate. When feature decay is fast ($\alpha>0.5$) and target decay is slow (Area Aa$^*$), WSD yields compute-optimal slopes strictly greater than both constant learning rates and SGD. This provides the first theoretical explanation for WSD's effectiveness under SignSGD: it leverages late-stage decay to harvest gains from noise-reshaping.

## Key Experimental Results

### Main Results

**Compute-optimal slope comparison: SignSGD vs SGD**

| Parameter Region | SignSGD Slope | SGD Slope | Winner | Description |
| :--- | :--- | :--- | :--- | :--- |
| Noise-dominated (High $N$ relative to $M$) | Steeper | Shallower | SignSGD | Benefits from noise-reshaping effect |
| Bias-dominated (Low $N$ relative to $M$) | Similar/Shallower | Similar | SGD or Tie | Drift-normalization may be disadvantageous |
| Balanced Region | Transitional | Transitional | Depends | Competition between the two effects |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Varying $\beta$ with fixed $\alpha$ | Slope Change | Slower target decay leads to greater SignSGD advantage |
| Varying $\alpha$ with fixed $\beta$ | Slope Change | Feature decay has similar impacts on both |
| With/Without WSD | Noise Reduction | WSD is highly effective when $\alpha$ is large and $\beta$ is small |
| Different Learning Rates | Optimal Config | SignSGD $e^*$ is always larger than SGD (uses smaller learning rates) |

### Key Findings

1. **Explicit conditions for SignSGD superiority**: SignSGD provides better compute-optimal scaling in noise-dominated regions (high step to model size ratio). This aligns with empirical observations where Adam outperforms SGD in large-scale training.
2. **Separable effects**: Drift-normalization and noise-reshaping have explicit mathematical forms, allowing independent analysis of their contributions to scaling laws.
3. **Theoretical support for WSD**: Provides the first theoretical explanation for WSD's effectiveness in SignSGD, showing it sharpens scaling slopes by minimizing noise terms.
4. **Power-law dominance**: The parameters $\alpha$ (feature decay) and $\beta$ (target decay) entirely determine the relative performance of SignSGD and SGD.

## Highlights & Insights

1. **Fills theoretical gap**: Establishes a complete scaling law theory for SignSGD, providing a contrast to existing SGD analyses.
2. **Discovery of unique effects**: Drift-normalization and noise-reshaping are critical concepts for understanding adaptive optimizers, potentially generalizable to broader settings.
3. **Practical guidance**: Identifies noise-dominated regions as the domain where SignSGD/Adam surpases SGD, matching the reality of large-scale LLM training.
4. **Theoretical foundation for WSD**: Explains a widely used but theoretically undersupported scheduling strategy.
5. **Analytical thoroughness**: A comprehensive 89-page theoretical study covering all regions of the parameter space with 25 supporting figures.

## Limitations & Future Work

1. **Simplified model assumptions**: Linear regression with random features differs significantly from deep neural networks. Non-linear effects and parameter coupling are ignored.
2. **One-pass assumption**: Analyzes single-epoch training, whereas practical training often involves multiple epochs.
3. **Lack of momentum**: SignSGD is a simplified version of Adam; incorporating first and second-order momentum could alter scaling behaviors.
4. **Gaussian-sketched features**: The Gaussian assumption may not capture structural nuances in real-world data.
5. **Non-linear extension**: Extending analysis to multi-layer networks is an important but challenging future direction.
6. **Empirical validation**: Further empirical work is needed to verify if theoretical predictions match observed scaling in complex deep learning tasks.

## Related Work & Insights

- **Paquette et al. (2024)**: Scaling law analysis for SGD in PLRF models; serves as the primary baseline.
- **Chinchilla Laws (Hoffmann et al., 2022)**: Original work on compute-optimal configurations; this work extends the logic to SignSGD.
- **Neural Scaling Laws (Kaplan et al., 2020)**: Scaling relationships for model size and data; this work adds the optimizer perspective.
- **Insight**: Understanding the impact of optimizers on scaling laws is as vital as understanding architecture and data; simple non-linear operations like "sign" can yield complex beneficial effects.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic analysis of SignSGD scaling laws; introduces drift-normalization and noise-reshaping.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Highly detailed theoretical analysis (89 pages) with numerical verification, though lacks deep learning experiments.
- Writing Quality: ⭐⭐⭐⭐ — Theoretically rigorous, though the extreme length may impact readability.
- Value: ⭐⭐⭐⭐ — Provides critical insights for understanding theoretical advantages of adaptive optimizers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)
- [\[NeurIPS 2025\] Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules](../../NeurIPS2025/optimization/functional_scaling_laws_in_kernel_regression_loss_dynamics_and_learning_rate_sch.md)
- [\[ICLR 2026\] Distributionally Robust Linear Regression with Block Lewis Weights](distributionally_robust_linear_regression_with_block_lewis_weights.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](../../NeurIPS2025/optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[NeurIPS 2025\] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](../../NeurIPS2025/optimization/learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)

</div>

<!-- RELATED:END -->
