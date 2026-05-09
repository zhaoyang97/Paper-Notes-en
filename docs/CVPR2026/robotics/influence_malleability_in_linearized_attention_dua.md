---
title: >-
  [Paper Note] Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics
description: >-
  [CVPR 2026][Robotics][Linearized attention] This paper establishes via the NTK framework that linearized attention fails to converge to the infinite-width kernel limit (requiring width $m = \Omega(\kappa^6)$), and proposes the "influence malleability" metric to quantify its dual implications: attention exhibits 6–9× higher data-dependent flexibility than ReLU networks, which simultaneously reduces approximation error and increases adversarial vulnerability.
tags:
  - CVPR 2026
  - Robotics
  - Linearized attention
  - neural tangent kernel
  - influence malleability
  - feature learning
  - adversarial robustness
date: 2026-05-08
content_hash: 8d8c2d3f109e5c51
---

# Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics

**Conference**: CVPR 2026
**arXiv**: [2603.13085](https://arxiv.org/abs/2603.13085)
**Code**: None
**Area**: Deep Learning Theory / Attention Mechanisms
**Keywords**: Linearized attention, neural tangent kernel, influence malleability, feature learning, adversarial robustness

## TL;DR

This paper establishes via the NTK framework that linearized attention fails to converge to the infinite-width kernel limit (requiring width $m = \Omega(\kappa^6)$), and proposes the "influence malleability" metric to quantify its dual implications: attention exhibits 6–9× higher data-dependent flexibility than ReLU networks, which simultaneously reduces approximation error and increases adversarial vulnerability.

## Background & Motivation

**Background**: NTK (Neural Tangent Kernel) theory establishes an equivalence between infinitely wide networks and kernel methods, predicting that sufficiently wide networks maintain an approximately constant kernel throughout training (the "lazy training" regime). This framework has been extended to deep networks and arbitrary architectures, yet attention mechanisms have not been rigorously incorporated into NTK analysis.

**Limitations of Prior Work**: (1) Prior studies focus either on the architectural properties of attention or on final performance, neglecting the dynamical characteristics of the attention learning process. (2) Whether NTK theory applies to attention is entirely unknown—Wenger et al. note that NTK theory is valid only for networks "orders of magnitude wider than their depth." (3) There is a lack of theoretical tools for quantifying attention's sensitivity to training data.

**Key Challenge**: The expressiveness of attention mechanisms (flexible adaptation to data structure) and their fragile sensitivity to training data may share a common origin—deviation from the kernel regime.

**Goal**: Does linearized attention converge to the infinite-width NTK limit? If not, what does this non-convergent behavior imply for the model's dependence on training data?

**Key Insight**: The paper designs a parameter-free linearized attention $f^{att}(X) = XX^TX$, establishes its exact correspondence with a data-dependent Gram-induced kernel, leverages spectral analysis to explain non-convergence, and quantifies the dual effects via "influence malleability."

**Core Idea**: The power and vulnerability of attention share the same origin—their transcendence of the kernel regime. The data-dependent kernel yields flexibility while simultaneously introducing fragility.

## Method

### Overall Architecture

Raw input $X \in \mathbb{R}^{n \times d}$ → linearized attention transform $f^{att}(X) = XX^TX$ → $\ell_2$ normalization → two-layer ReLU MLP → output predictions. The baseline is a 2L-ReLU network operating directly on raw inputs. The distance $\|f_m - f_{NTK}\|$ between the finite-width NTK and the infinite-width NTK is computed across varying widths $m$, and influence functions are used to quantify data dependence.

### Key Designs

1. **Linearized Attention and the Gram-Induced Kernel (Theorem 4.1)**
   - The parameter-free linearized attention $f^{att}(X) = XX^TX$ corresponds to scaled dot-product attention with $W_Q = W_K = W_V = I$ and a linearized softmax.
   - It corresponds exactly to a data-dependent Gram-induced kernel $K_{LinAttn} = G^3$ (where $G = XX^T$).
   - Each kernel element takes the form of a fourth-order interaction term $\sum_{k,\ell}(x_i^Tx_k)(x_k^Tx_\ell)(x_\ell^Tx_j)$, realizing transitive similarity propagation $i \to k \to \ell \to j$.
   - A key distinction from standard polynomial kernels $(x^Ty)^p$: the sensitivity of $K_{LinAttn}$ depends on the correlation structure of the entire dataset (through $G$), not merely the relationship between individual input pairs.

2. **Spectral Amplification and NTK Non-Convergence (Theorem 4.7)**
   - The attention transform cubicizes the condition number of the Gram matrix: $\kappa(\tilde{G}) = \kappa(G)^3$.
   - NTK convergence requires width $m = \Omega(\kappa(G)^6 / \epsilon^2)$: for MNIST ($\kappa \approx 1.2 \times 10^3$) this demands $m \gg 10^{18}$, and for CIFAR-10 ($\kappa \approx 8.7 \times 10^3$) it demands $m \gg 10^{24}$—far beyond the experimental range of $m \leq 4096$.
   - By contrast, a 2L-ReLU network requires only $m = \Omega(1/\epsilon^2)$, with no spectral amplification factor.
   - Physical interpretation: stacking $k$ layers of linearized self-attention produces $G^{2k+1}$, with the condition number growing as $\kappa^{2k+1}$—the more layers, the further from the kernel regime.

3. **Influence Malleability Metric (Definition 3.4)**
   - *Influence Flip Rate*: the proportion of top-10% high-influence training samples whose influence function sign flips after PGD perturbation ($\epsilon = 0.3$).
   - Complementary metric: Spearman correlation $\rho$ of influence rankings (lower values indicate higher malleability).
   - Three data intervention strategies: Curated (removing top-$\tau$ influential samples), Transformed (replacing with adversarial versions), and Adversarial (applying PGD perturbation to all training data).
   - Influence functions are computed efficiently via the empirical finite-width NTK matrix $(K_m + \lambda I)^{-1}$, without retraining.

### Loss & Training

Cross-entropy (multi-class) or MSE (binary classification) + L2 regularization ($\lambda = 10^{-3}$). Adam optimizer, $lr = 10^{-3}$, batch size 128, 500 training epochs. FGSM, PGD, and MIM adversarial perturbation methods are used for influence malleability measurement.

## Key Experimental Results

### Main Results

| Dataset | Metric | MLP-Attn | 2L-ReLU | Ratio |
|--------|------|----------|---------|------|
| MNIST (10-class) | Flip Rate (PGD) | 28.9% | 3.3% | 8.8× |
| MNIST (10-class) | Flip Rate (FGSM) | 34.6% | 4.1% | 8.4× |
| CIFAR-10 (10-class) | Flip Rate (PGD) | 19.1% | 3.1% | 6.2× |
| CIFAR-10 (10-class) | Flip Rate (FGSM) | 26.4% | 3.3% | 8.0× |
| MNIST (binary) | Flip Rate (PGD) | 41.0% | 8.4% | 4.9× |

| Dataset | NTK Distance ($m=16$) | NTK Distance ($m=4096$) | Trend |
|--------|---------------|-----------------|------|
| MNIST 2L-ReLU | 45.1 | 39.2 | Monotone ↓ (converges) |
| MNIST MLP-Attn | 10.3 | 43.4 | Non-monotone ↑ (does not converge) |
| CIFAR-10 2L-ReLU | 246.2 | 56.9 | Monotone ↓ (converges) |
| CIFAR-10 MLP-Attn | 3.7 | 12.6 | Monotone ↑ (does not converge) |

### Ablation Study

| Experiment | Result |
|------|------|
| Adversarial training on 2L-ReLU | Flip Rate: 3.3% → 43.4% (MNIST), showing AT can induce malleability |
| Adversarial training on MLP-Attn | Flip Rate: 28.9% → 42.2% (MNIST), smaller gain—architecture is intrinsically highly malleable |
| Binary CIFAR-10 | MLP-Attn advantage disappears (≈1×), as binary Gram condition numbers are lower, weakening the cubic amplification effect |
| Perturbation strength $\epsilon$: 0.1→0.5 | MLP-Attn consistently exceeds 2L-ReLU; relative ordering unchanged |

### Key Findings

- Attention consistently exhibits far higher influence malleability than ReLU (6–9×) across all tested conditions, without requiring adversarial training.
- The empirical Gram condition numbers closely match theoretical width requirements: MNIST $\kappa \approx 10^3$, CIFAR-10 $\kappa \approx 10^{3.9}$.
- Adversarial training and the attention architecture induce malleability through distinct mechanisms: the former is training-induced, the latter is architecturally intrinsic.

## Highlights & Insights

- This work provides the first rigorous NTK-based proof that attention does not enter the kernel regime: spectral amplification by $\kappa^3$ causes width requirements to scale as a sixth-power, a clean and elegant theoretical result.
- The concept of "influence malleability" precisely characterizes the shared origin of attention's power and vulnerability, offering a new perspective on why Transformers are simultaneously powerful and fragile.
- The adversarial training experiments reveal two distinct mechanisms underlying malleability (architecturally intrinsic vs. training-induced), disentangling confounding factors.
- Theory and experiment are highly consistent: empirical Gram condition numbers accurately predict non-convergent behavior.

## Limitations & Future Work

- Only linearized attention (identity QKV) is analyzed; the framework is not extended to full softmax attention—the row normalization of softmax may further amplify non-convergence.
- Experimental scale is constrained by exact NTK computation (MNIST/CIFAR-10, two-layer networks, $m \leq 4096$), leaving a gap with practical Transformers.
- Theorem 4.7 bounds only the NTK deviation at initialization and does not directly predict post-training trajectories.
- It remains unexplored whether low-rank regularization (truncated attention retaining top-$r$ singular values) can restore convergence and reduce adversarial vulnerability.

## Related Work & Insights

- **vs. Jacot et al. (NTK, 2018)**: Classical NTK theory predicts convergence for wide networks; this paper demonstrates that attention architectures violate this prediction.
- **vs. Chizat et al. (Lazy vs. Feature Learning, 2019)**: This paper provides a concrete architectural instantiation and quantitative evidence of attention as a "feature learning regime" model.
- **vs. Zhang et al. (NTK-based Influence, 2022)**: Extends their NTK influence function methodology, but applies it for the first time to compare influence malleability across architectures.
- **Practical implications**: The spectral amplification effect in linearized attention suggests that low-rank approximation of attention modules may preserve expressiveness while restoring NTK convergence, providing new theoretical guidance for attention head pruning and low-rank decomposition.
- Influence malleability can serve as a new metric for measuring model robustness at a finer granularity than conventional adversarial accuracy.

## Rating

- Novelty: ⭐⭐⭐⭐ Establishing attention theory from the NTK perspective is a novel viewpoint; the influence malleability concept is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple datasets, perturbation types, and classification settings with strong theory–experiment alignment.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous and complete; the paper is clearly structured, with proofs in the appendix that do not impede readability.
- Value: ⭐⭐⭐ A theoretical contribution; direct applicability to practical Transformers awaits further validation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] CycleManip: Enabling Cyclic Task Manipulation via Effective Historical Perception and Understanding](cyclemanip_enabling_cyclic_task_manipulation_via_effective_historical_percepti.md)
- [\[CVPR 2026\] DeepSketcher: Internalizing Visual Manipulation for Multimodal Reasoning](deepsketcher_internalizing_visual_manipulation_for_multimodal_reasoning.md)

<!-- RELATED:END -->
