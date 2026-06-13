---
title: >-
  [Paper Note] Bayesian Gated Non-Negative Contrastive Learning
description: >-
  [ICML 2026][Optimization][Contrastive learning] Addressing the optimization conflicts (gradient oscillation) caused by shared background features in Non-Negative Contrastive Learning (NCL)…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Contrastive learning"
  - "non-negative representation"
  - "Bayesian gating"
  - "semantic disentanglement"
  - "interpretability"
date: 2026-05-08
content_hash: 12d7e3faff26609a
---

# Bayesian Gated Non-Negative Contrastive Learning

**Conference**: ICML 2026  
**arXiv**: [2605.28441](https://arxiv.org/abs/2605.28441)  
**Code**: https://github.com/Cui-Peng-624/BayesNCL  
**Area**: Self-Supervised/Representation Learning  
**Keywords**: Contrastive learning, non-negative representation, Bayesian gating, semantic disentanglement, interpretability  

## TL;DR
Addressing the optimization conflicts (gradient oscillation) caused by shared background features in Non-Negative Contrastive Learning (NCL), BayesNCL is proposed. It utilizes a Bayesian gating head to learn a Bernoulli distribution for each feature dimension to dynamically filter high-frequency common features, achieving a 142.1% improvement in semantic consistency on ImageNet-100 without sacrificing downstream accuracy.

## Background & Motivation

**Background**: Self-supervised contrastive learning (CL) has become the dominant paradigm for unlabeled visual representation learning, encoding high-level semantics by pulling positive pairs closer and pushing negative pairs apart. However, standard CL representations are highly entangled—semantic concepts are scattered across all feature dimensions, and individual dimensions lack clear semantic meaning, which poses serious risks in safety-critical scenarios.

**Limitations of Prior Work**: Non-Negative Contrastive Learning (NCL) establishes equivalence with Non-negative Matrix Factorization (NMF) by enforcing non-negativity constraints, aligning feature dimensions with semantic clusters. However, NCL employs a deterministic similarity measure that treats all dimensions equally. This deterministic approach fails when real-world images exhibit compositionality—where different objects share high-frequency background features (e.g., "blue sky").

**Key Challenge**: The authors identify a fundamental **optimization conflict**. Taking the "blue sky" background shared by "bird" and "airplane" as an example: positive pairs require aligning the "blue sky" dimension, while negative pairs require repelling that same dimension. This results in the expected gradient of conflicting feature dimensions approaching zero while maintaining extremely high variance ($\mathbb{E}[\nabla_{z_k} L] \approx 0, \text{Var}(\nabla_{z_k} L) \gg 0$), which hinders semantic disentanglement.

**Key Insight**: Re-evaluating the similarity definition from a probabilistic perspective. Joint likelihood analysis reveals that optimal similarity should be weighted by inverse frequency: $s_{IPW}(z,z') = \sum_k \frac{1}{\pi_k} z_k z_k'$. This means rare discriminative features should dominate the alignment score, while high-frequency background features should be down-weighted. However, explicitly calculating the semantic frequency $\pi_k$ is intractable.

**Core Idea**: Introduce a learnable Bayesian gating mechanism to automatically "turn off" conflicting high-frequency common feature dimensions via variational inference, serving as a feasible approximation for inverse frequency weighted similarity through probabilistic filtering.

## Method

### Overall Architecture
BayesNCL adds a **Bayesian gating head** to the standard NCL encoder $f: \mathcal{X} \to \mathbb{R}_{\geq 0}^K$. Given an input image, the encoder produces non-negative features $z = f(x)$. The gating head predicts Bernoulli parameters $\alpha = \sigma(g_\theta(x))$ for each dimension. Discrete masks $m_{\text{hard}} = \mathbb{I}(\alpha > 0.5)$ are generated via thresholding, and the final gated representation $\tilde{z} = z \odot m_{\text{hard}}$ is used to calculate the contrastive loss. The training objective combines the gated InfoNCE loss with KL sparsity regularization.

### Key Designs

1. **Bayesian Gating Head**:

    - **Function**: Dynamically determines "on/off" status for each feature dimension to filter conflicting common features.
    - **Mechanism**: The gating head is a 2-layer MLP that takes encoder features as input and outputs the activation probability for each dimension $\alpha_k = \sigma(g_\theta(x)_k)$. The forward pass uses hard thresholding $m_{\text{hard}} = \mathbb{I}(\alpha > 0.5)$ to ensure strict feature selection; the backward pass utilizes the Straight-Through Estimator (STE) to propagate gradients through the continuous probabilities $\alpha$: $m_{\text{train}} = \text{sg}[m_{\text{hard}} - \alpha] + \alpha$. Experiments demonstrate that hard gating significantly outperforms soft gating (ImageNet-100 consistency 36.14 vs 33.15), as strict zeroing is necessary to completely terminate gradient oscillation.
    - **Design Motivation**: Deterministic methods (e.g., TopK) can only apply global fixed thresholds and cannot dynamically judge which dimensions are conflicting based on specific sample content. Probabilistic gating allows "bird" and "airplane" images to independently close different background dimensions.

2. **Variational Sparsity Regularization**:

    - **Function**: Constrains the gating distribution to align with a sparse prior, automatically suppressing high-frequency features.
    - **Mechanism**: Feature selection is formulated as a variational inference problem. The training objective is the ELBO: $\mathcal{L}_{\text{total}} = \mathbb{E}_{m \sim q_\theta}[\mathcal{L}_{\text{InfoNCE}}(z \odot m)] + \lambda \cdot D_{\text{KL}}(q_\theta(m|x) \| p(m))$, where the prior $p(m) = \text{Bern}(\rho)$ controls the expected sparsity. Theoretical proof (Theorem 4.5) shows that as feature frequency $\pi_k \to 1$, its alignment gain $\gamma \cdot \pi_k(1-\pi_k) \to 0$, falling below the sparsity cost $\lambda \cdot \log(1/\rho)$, which causes the dimension to be automatically disabled.
    - **Design Motivation**: The KL term acts as an "inverse frequency filter," allocating limited activation dimension resources to discriminative features rather than consuming them on high-frequency backgrounds that do not aid in instance discrimination.

3. **Gradient Detachment**:

    - **Function**: Prevents the encoder backbone from being misled by the sparsity loss, preserving representation quality.
    - **Mechanism**: Gradients from the sparsity regularization term $\mathcal{L}_{\text{sparsity}}$ are back-propagated only to the gating head parameters $\theta$, and not to the encoder backbone. The encoder learns solely through the gated InfoNCE loss, ensuring the backbone focuses on learning discriminative representations rather than satisfying sparsity constraints.
    - **Design Motivation**: Ablation experiments show that removing gradient detachment causes ImageNet-100 consistency to drop from 36.14 to 22.29, indicating that forcing the backbone to optimize both objectives simultaneously leads to objective conflict.

## Key Experimental Results

### Main Results (Interpretability Metrics)

| Method | CIFAR-10 Cons.↑ | CIFAR-100 Cons.↑ | IN-100 Cons.↑ | IN-100 $H_s$↓ | IN-100 Act. |
|------|-----------------|-------------------|---------------|----------------|-------------|
| CL | 10.00 | 1.00 | 1.00 | 4.59 | 1.00 |
| NCL | 53.82 | 9.91 | 14.93 | 3.28 | 0.48 |
| NCL+TopK | 51.81 | 12.32 | 14.27 | 3.35 | 0.47 |
| BayesNCL (GS) | 53.68 | 20.75 | 11.66 | 3.36 | 0.13 |
| **BayesNCL (STE)** | **56.50** | **22.02** | **36.14** | **2.10** | 0.50 |

BayesNCL-STE achieves a semantic consistency of 36.14 on ImageNet-100, which is a **142.1%** improvement compared to NCL's 14.93. Meanwhile, its activation rate (0.50) is higher than NCL's (0.48), indicating that it does not simply "turn off channels" but achieves **effective sparsity**.

### Downstream Task Performance

| Method | CIFAR-10 LP@1 | CIFAR-100 LP@1 | IN-100 LP@1 | IN-100 LP@5 |
|------|---------------|----------------|-------------|-------------|
| CL | 87.88 | 59.72 | 68.31 | 90.24 |
| NCL | 87.80 | 60.67 | 69.63 | 91.23 |
| **BayesNCL** | **88.02** | 60.69 | **70.44** | **91.71** |

While significantly enhancing interpretability, BayesNCL also increases linear probing accuracy (IN-100: 70.44% vs NCL 69.63%), breaking the "interpretability-performance" trade-off.

### Ablation Study

| Configuration | IN-100 Cons.↑ | IN-100 LP@1 | Description |
|------|---------------|-------------|------|
| BayesNCL (Full) | 36.14 | 70.44 | Baseline |
| Soft Gating | 33.15 | 69.87 | Soft gating fails to terminate gradient oscillation |
| Remove Detachment | 22.29 | 67.47 | Backbone misled by sparsity loss |
| 1-Layer MLP | 40.85 | 69.55 | High consistency but slightly lower accuracy |
| 3-Layer MLP | 26.37 | 68.09 | Overly deep gating head leads to unstable training |

### Computational Overhead

| Dataset | Method | Training Time (min) | FLOPs |
|--------|------|---------------|-------|
| CIFAR-100 | NCL | 70.95 | 1.416G |
| CIFAR-100 | BayesNCL | 75.12 | 1.419G |
| IN-100 | NCL | 193.78 | 3.731G |
| IN-100 | BayesNCL | 218.53 | 3.815G |

The increase in FLOPs is only ~2%, and training time increases by ~13%, representing minimal overhead.

## Highlights & Insights
- **Precise Problem Localization**: Attributes the interpretability bottleneck of NCL to "optimization conflicts" and provides a formalized proof via gradient analysis, offering a new perspective on understanding feature entanglement in contrastive learning.
- **Theoretical Completeness**: Provides theoretical guarantees from four complementary perspectives: local semantic filtering (Theorem 4.5), global error reduction (Theorem 4.7), information-theoretic constraints (Theorem 4.9), and generalization bounds.
- **Effective Sparsity vs. Simple Sparsity**: BayesNCL does not simply shut down channels; instead, it dynamically recruits more dimensions to encode different semantic concepts while filtering entangled noise, resulting in an activation rate that is actually higher than NCL.

## Limitations & Future Work
- Validated only on CIFAR-10/100 and ImageNet-100; lacks large-scale experiments on ImageNet-1K.
- The backbone networks are limited to ResNet-18/50, and effectiveness on modern architectures like ViT has not been verified.
- Hyperparameters $\rho$ and $\lambda$ require tuning and exhibit an "inverted U-shape" sensitivity interval.
- The gating head makes decisions based solely on single-sample features without considering global statistical information across samples.

## Related Work & Insights
- NCL (Wang et al., 2024) establishes the equivalence between CL and NMF through non-negativity constraints and is the direct predecessor of this work.
- Variational Information Bottleneck (VIB) compresses information by injecting noise, whereas BayesNCL achieves a "hard information bottleneck" through structured sparsity.
- The "feature superposition" in Sparse Autoencoders (SAE) bears similarities to the "optimization conflict" described in this paper.
- Scalable to multimodal contrastive learning (e.g., CLIP) to resolve conflict issues regarding shared features between modalities.

## Rating
- Novelty: 8/10 — The formal definition of optimization conflict and the Bayesian gating solution are novel theoretical contributions.
- Experimental Thoroughness: 7/10 — Ablations are detailed, but validation on large-scale datasets is lacking.
- Writing Quality: 9/10 — Problem definition is clear, theoretical derivations are complete, and experimental analysis is thorough.
- Value: 7/10 — Provides an effective solution for interpretable self-supervised learning, though the scope of applicability requires further verification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning](../../CVPR2026/optimization/bd-merging_bias-aware_dynamic_model_merging_with_evidence-guided_contrastive_lea.md)
- [\[ICML 2026\] Cost-Aware Stopping for Bayesian Optimization](cost-aware_stopping_for_bayesian_optimization.md)
- [\[ICML 2026\] Multi-Objective Bayesian Optimization via Adaptive $\varepsilon$-Constraints Decomposition](multi-objective_bayesian_optimization_via_adaptive_varepsilon-constraints_decomp.md)
- [\[ICML 2026\] SyMerge: From Non-Interference to Synergistic Merging via Single-Layer Adaptation](symerge_from_non-interference_to_synergistic_merging_via_single-layer_adaptation.md)
- [\[ICML 2026\] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization](safeguarded_stochastic_polyak_step_sizes_for_non-smooth_optimization_robust_perf.md)

</div>

<!-- RELATED:END -->
