---
title: >-
  [Paper Note] Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization
description: >-
  [ICLR 2026][Optimization][Sharpness-Aware Minimization] This work provides an in-depth analysis of the implicit bias of SAM trained on linear diagonal networks, revealing a qualitative shift induced by depth from $L=1$ to $L=2$: the limit direction of $\ell_\infty$-SAM is highly sensitive to initialization, while $\ell_2$-SAM exhibits a "weak-to-strong" **sequential feature amplification** phenomenon. The results indicate that analyses focusing solely on the $t\to\infty$ limi…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Sharpness-Aware Minimization"
  - "Implicit Bias"
  - "Linear Diagonal Networks"
  - "Feature Amplification"
  - "Depth-Induced"
date: 2026-05-08
content_hash: ef5c7c6e4e2c7d1c
---

# Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization

**Conference**: ICLR 2026  
**arXiv**: [2603.08290](https://arxiv.org/abs/2603.08290)  
**Code**: None  
**Area**: Optimization Theory  
**Keywords**: Sharpness-Aware Minimization, Implicit Bias, Linear Diagonal Networks, Feature Amplification, Depth-Induced

## TL;DR

This work provides an in-depth analysis of the implicit bias of SAM trained on linear diagonal networks, revealing a qualitative shift induced by depth from $L=1$ to $L=2$: the limit direction of $\ell_\infty$-SAM is highly sensitive to initialization, while $\ell_2$-SAM exhibits a "weak-to-strong" **sequential feature amplification** phenomenon. The results indicate that analyses focusing solely on the $t\to\infty$ limit are insufficient to reveal the complete dynamic behavior of SAM.

## Background & Motivation

### Background
Sharpness-Aware Minimization (SAM) seeks flat minima by minimizing the worst-case loss within a neighborhood, significantly improving generalization in practice. Previous theoretical works primarily analyzed the implicit bias of SAM in settings with finite minima (e.g., squared loss), while the case where the loss infimum is at infinity (e.g., logistic loss) remains insufficiently understood.

### Motivation
The authors examine the implicit bias of SAM when training on linearly separable binary classification data (logistic loss) using an $L$-layer linear diagonal network. Surprising findings include:

- **Depth $L=1$ (Linear Model)**: Both $\ell_\infty$-SAM and $\ell_2$-SAM converge to the $\ell_2$ maximum margin classifier, consistent with Gradient Descent (GD).
- **Depth $L=2$**: Behavior undergoes a qualitative change—even on a single-sample dataset $\{(\boldsymbol\mu, +1)\}$ where $\boldsymbol\mu=(1,2)$, the SAM trajectory can deviate from the $\ell_1$ maximum margin direction favored by GD.

This observation reveals that adding just one layer fundamentally alters the implicit bias of SAM.

## Method

### Overall Architecture

The paper does not propose a new algorithm but characterizes the continuous-time implicit bias of $\ell_\infty$-SAM and $\ell_2$-SAM on an $L$-layer linear diagonal network $f(\mathbf{x}) = \langle \boldsymbol\beta(\boldsymbol\theta), \mathbf{x}\rangle$, where the equivalent predictor $\boldsymbol\beta(\boldsymbol\theta) = \bigodot_{\ell=1}^L \mathbf{w}^{(\ell)}$ is the element-wise product of weights across layers. Training is conducted with logistic loss $\ell(u)=\log(1+\exp(-u))$ on linearly separable data. The analysis uses a single-sample dataset $\mathcal{D}_{\boldsymbol\mu}=\{(\boldsymbol\mu,+1)\}$ as the minimal solvable model. Using SAM flow ODEs and a rescaled flow for simplified analysis, limit directions and finite-time trajectories are derived based on depth $L$ and perturbation norm types.

### Key Designs

**1. Degenerate Conclusion for Depth 1: Establishing a Baseline for "Where SAM Becomes Special"**

To argue "depth-induced" changes, differences present in a single layer must first be excluded. Theorem 3.1 proves that for almost all linearly separable datasets, any perturbation radius $\rho$, and any initialization, the direction of $\ell_\infty$-SAM flow converges to the $\ell_2$ maximum margin direction, identical to GD. In the single-sample setting, the $\ell_\infty$-SAM trajectory even coincides pointwise with GD. This confirms that on linear models ($L=1$), the implicit biases of SAM and GD are indistinguishable, isolating the source of subsequent counter-intuitive phenomena to depth itself rather than the SAM perturbation form.

**2. $\ell_\infty$-SAM for Depth $L\ge 2$: Perturbation Radius as a Coordinate-Level Gating Threshold**

With an additional layer, $\ell_\infty$-SAM can select minor features. In single-sample datasets, Theorem 3.2 shows the evolution of each coordinate $\beta_j(t)$ is entirely determined by its initial value $\alpha_j$ relative to $\rho$: it is suppressed if $\alpha_j<\rho$ ($\beta_j(t)\to 0$ for even $L$, $\to\rho^L$ for odd $L$), stays constant if $\alpha_j=\rho$, and diverges for $\alpha_j>\rho$. Because coordinates are decoupled, the limit direction is determined by the unique winning coordinate $j^*=\arg\max_{j:\alpha_j>\rho}\mu_j(\alpha_j-\rho)^{L-2}$ (Corollary 3.5). This means by tuning initialization, $\ell_\infty$-SAM can converge to **any standard basis vector direction**, including minor directions with small feature weights $\mu_j$—a sharp contrast to GD, which always locks onto major features. Here, $\rho$ acts as a per-coordinate switch.

**3. $\ell_2$-SAM for Depth 2: Same Limit as GD, but "Sequential Feature Amplification" in Finite Time**

This is the most core and counter-intuitive finding. Theorem 4.2 first shows that the limit direction of $\ell_2$-SAM flow is still the $\ell_1$ maximum margin solution, identical to GD—looking only at $t\to\infty$ would lead to the misinterpretation that "SAM makes no difference." However, finite-time dynamics are entirely different: across the time dimension (Theorem 4.4), the predictor $\boldsymbol\beta(t)$ **depends on minor coordinates early on** and only **gradually shifts to major coordinates** as training progresses. Increasing the initialization scale (Theorem 4.5) triggers the same "minor-to-major" transition. The root cause is the gradient normalization factor in the $\ell_2$-SAM perturbation, which relatively amplifies smaller coordinates in the gradient, giving $\beta_j$ higher early growth rates. As training continues, major coordinates with larger feature weights $\mu_j$ eventually catch up. This "minor first, major last" phenomenon is a concrete counter-example showing that limit analysis is insufficient to characterize SAM.

**4. Rescaled Flow: Canceling Common Velocity Terms for Closed-Form Solutions**

Precise trajectory characterization relies on a technical tool. In the single-sample SAM flow, there is always a positive loss derivative term $-\ell'(\langle\boldsymbol\beta(\hat{\boldsymbol\theta}(t)),\boldsymbol\mu\rangle)>0$, which is a common scalar for all coordinates, affecting only the speed of evolution but not the direction. Canceling this term yields a rescaled flow, equivalent to a reparameterization of time. The spatial trajectory is preserved exactly, allowing for closed-form analysis of coordinate-level evolution equations. This is the premise for the precise trajectories in Theorems 3.2 and 4.4.

## Key Experimental Results

### Main Results
- **Synthetic Experiments**: A 2D single-sample dataset $\boldsymbol\mu=(1,2)$ clearly demonstrates trajectory differences between GD vs. $\ell_\infty$-SAM vs. $\ell_2$-SAM.
- Multi-sample datasets verify that theoretical predictions hold in practical settings.
- Depth $L=3$ experiments confirm that $\ell_\infty$-SAM becomes even more sensitive to initialization in deeper networks.

### CNN Experiments (MNIST)

| Method | Grad-CAM Observation | Description |
|------|-------------|------|
| GD | Focuses on primary digit pixels | Conventional behavior |
| $\ell_2$-SAM | Emphasizes background/weak pixel regions | Consistent with "minor-to-major" theory |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| $L=1$, any $\rho$ | Consistent with GD | SAM does not change bias |
| $L=2$, $\ell_\infty$, $\alpha_j < \rho$ | $\beta_j \to 0$ | Coordinate suppressed |
| $L=2$, $\ell_\infty$, $\alpha_j > \rho$ | Exponential growth | Minor features can be selected |
| $L=2$, $\ell_2$ | "Minor first, major last" | Sequential feature amplification |

### Key Findings
- For $L\geq 2$, $\ell_\infty$-SAM is extremely sensitive to initialization, where the perturbation radius $\rho$ serves as a coordinate-level "gating" threshold.
- The finite-time behavior of $\ell_2$-SAM differs fundamentally from its infinite-time limit, suggesting limit analysis is insufficient.
- Discrete SAM updates align closely with continuous-time SAM flow, validating the practical relevance of the theory.

## Highlights & Insights

1. **"Minor first, major last" phenomenon**: Reveals a counter-intuitive finite-time behavior of $\ell_2$-SAM—the optimizer focuses on weak features before strong ones, driven by gradient normalization.
2. **Finite Time vs. Infinite Time**: Provides a clear example showing that implicit bias analysis focusing only on $t\to\infty$ may miss critical dynamic information, advocating for more finite-time analysis.
3. **Qualitative Effect of Depth**: Adding just one layer ($L=1\to L=2$) completely changes SAM's behavior, revealing a deep interaction between depth and optimization algorithms.
4. **Precise Trajectory Characterization**: The independent coordinate evolution in $\ell_\infty$-SAM allows for exact trajectory representation, providing an elegant theoretical simplification.

## Limitations & Future Work

- Theoretical analysis is limited to the simplified **linear diagonal network** model, which differs from actual deep non-linear networks.
- Multi-sample dataset analysis faces additional technical hurdles and is currently limited to experimental validation.
- The limit direction theorem for $\ell_2$-SAM (Theorem 4.2) depends on the assumption of directional convergence.
- The impact of sequential feature amplification on generalization in actual deep learning remains unclear.
- Analysis does not cover SAM variants such as ASAM or GSAM.

## Related Work & Insights

- **Soudry et al. (2018)**: Classical result of GD converging to $\ell_2$ max margin on linear models.
- **Gunasekar et al. (2018)**: GD on linear diagonal networks biases towards $\ell_1$ sparse solutions.
- **Pesme & Flammarion (2023)**: Saddle-to-saddle dynamics in GD—similar staged learning, but with a different mechanism.
- **Foret et al. (2020)**: Original SAM paper.
- **Insight**: Finite-time analysis is crucial for understanding optimizer behavior. Future work should focus more on dynamic changes during training rather than just convergence results.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — "Sequential feature amplification" is a brand new and counter-intuitive discovery.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated with synthetic data and MNIST/CNN, though lacking large-scale experiments.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Rigorous theoretical derivation, clear illustrations, and thought-provoking comparison between finite and infinite time.
- **Value**: ⭐⭐⭐⭐ — Significantly advances theoretical understanding of SAM, though guidance for practical applications is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hyperbolic Aware Minimization: Implicit Bias for Sparsity](hyperbolic_aware_minimization_implicit_bias_for_sparsity.md)
- [\[ICLR 2026\] Towards Understanding the Calibration Benefits of Sharpness-Aware Minimization](towards_understanding_the_calibration_benefits_of_sharpness-aware_minimization.md)
- [\[ICLR 2026\] MASAM: Multimodal Adaptive Sharpness-Aware Minimization for Heterogeneous Data Fusion](masam_multimodal_adaptive_sharpness-aware_minimization_for_heterogeneous_data_fu.md)
- [\[ICLR 2026\] Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models](bi-lora_efficient_sharpness-aware_minimization_for_fine-tuning_large-scale_model.md)
- [\[ICLR 2026\] Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime](implicit_bias_of_per-sample_adam_on_separable_data_departure_from_the_full-batch.md)

</div>

<!-- RELATED:END -->
