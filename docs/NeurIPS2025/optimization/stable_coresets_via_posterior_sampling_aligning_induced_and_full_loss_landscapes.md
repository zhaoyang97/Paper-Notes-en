---
title: >-
  [Paper Note] Stable Coresets via Posterior Sampling: Aligning Induced and Full Loss Landscapes
description: >-
  [NeurIPS 2025][Optimization][Coreset Selection] This paper proposes a coreset selection framework based on posterior sampling. By sampling weight perturbations on BatchNorm layers to smooth the loss landscape, it guarantees alignment between the coreset and full-dataset loss landscapes (including approximations of the Hessian and Newton step), significantly outperforming existing methods under high label noise.
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Coreset Selection"
  - "Posterior Sampling"
  - "Loss Landscape Alignment"
  - "Label Noise Robustness"
  - "SGD Convergence"
date: 2026-05-08
content_hash: a2b46c90c2929c29
---

# Stable Coresets via Posterior Sampling: Aligning Induced and Full Loss Landscapes

**Conference**: NeurIPS 2025
**arXiv**: [2511.17399](https://arxiv.org/abs/2511.17399)  
**Code**: [https://github.com/changwk1001/stable-coreset.git](https://github.com/changwk1001/stable-coreset.git)  
**Area**: Efficient Training / Data Selection
**Keywords**: Coreset Selection, Posterior Sampling, Loss Landscape Alignment, Label Noise Robustness, SGD Convergence

## TL;DR

This paper proposes a coreset selection framework based on posterior sampling. By sampling weight perturbations on BatchNorm layers to smooth the loss landscape, it guarantees alignment between the coreset and full-dataset loss landscapes (including approximations of the Hessian and Newton step), significantly outperforming existing methods under high label noise.

## Background & Motivation

Coreset selection aims to identify a small, representative subset of training data to accelerate training. Gradient-matching methods (e.g., Craig, GradMatch) achieve this by selecting subsets whose gradient directions align with those of the full dataset.

**Core failure mode**: Under practical conditions (label noise, adversarial perturbations, extremely small subset budgets), gradient-matching methods tend to select samples with large gradient magnitudes, which are precisely the mislabeled or outlier samples. As a result, the **loss landscape induced by the coreset becomes severely misaligned with that of the full dataset**—the optimization trajectory on the subset deviates from the optimal direction of the full dataset. Experiments show that under 20%–50% label noise, many gradient-matching methods suffer catastrophic performance degradation.

**The dilemma of second-order methods**: Methods that attempt to match the Hessian (e.g., Crest) can improve alignment to some extent, but Hessian computation/approximation is extremely costly (sometimes negating the speedup from coresets), incurs enormous memory overhead (up to 3× that of full-data training), and remains unstable under noisy labels.

**A new perspective**: Rather than explicitly computing the Hessian, loss landscape alignment can be achieved implicitly via posterior sampling—by sampling weight perturbations in a Gaussian neighborhood of the current parameters and using Monte Carlo–averaged gradient information for coreset selection, yielding a smooth and geometrically aligned selection criterion automatically.

## Method

### Overall Architecture

The standard coreset selection objective of maximizing over domain $W$ is replaced with a $(\sigma, \epsilon, \bar{w})$-stability constraint. At each epoch, $P$ sub-samplings are performed, each selecting $m$ data points to form a shadow dataset $S_t$, on which batched SGD is then applied.

### Key Designs

1. **$(\sigma, \epsilon, \bar{w})$-stability definition**:
   A subset $S'$ is stable if and only if:
   $E_{w \sim \mathcal{N}(\bar{w}, \sigma I)} \|\nabla l_{S'}(w) - \nabla l(w)\|_2^2 \leq \epsilon$

   That is, within the Gaussian neighborhood of parameter $\bar{w}$, the expected squared distance between the subset gradient and the full-data gradient is sufficiently small. This is more robust than matching gradients only at the current point $\bar{w}$—it requires the subset to be a good representative over the entire neighborhood.

2. **Posterior sampling = implicit Hessian alignment (Theorem 3.2)**:
   If subset $S'$ is $(\sigma, \epsilon, w)$-stable, letting the Hessian discrepancy be $\mathcal{E} = H_{S',w} - H_{S,w}$, then:
    - $\|\mathcal{E}\| \leq O(\epsilon^{1/2})$: spectral norm bounded
    - $\text{tr}(\mathcal{E}^2) \leq O(\epsilon/\sigma)$: Frobenius norm bounded
    - $\|H_{S'}^{-1}\nabla l_{S'} - H_S^{-1}\nabla l_S\| \leq O(\epsilon^{1/2})$: Newton step approximation

   This means that through simple Gaussian sampling, the coreset's loss landscape (including curvature information) is guaranteed to align with the full dataset's, without ever explicitly computing the Hessian.

3. **BatchNorm layer sampling**:
   Sampling over all model parameters is computationally expensive. Motivated by recent findings that BN layers are critical for controlling sharpness and optimization performance, the authors apply Gaussian perturbations exclusively to BatchNorm layers. Ablation experiments (Table 2 Left) confirm that BN-layer sampling outperforms all-layer and FC-layer sampling across all noise levels. Only 4 sampled models are needed, with $\sigma$ selected from $\{0.1, 0.01, 0.001\}$ via cross-validation.

### Loss & Training

**Convergence analysis (Theorem 3.3)**:
- **Additive noise**: convergence rate $O(1/\sqrt{T})$, with the key term improved by a factor of $O(1/M)$ ($M$ = number of samples)
- **Multiplicative noise**: convergence rate improved from $O(1/\sqrt{RT})$ to $O(1/\sqrt{MRT})$ ($R$ = mini-batch size)

Settings: $\sigma_2^2 d = 1/(M\sqrt{T})$, learning rate $\eta = \min\{1/\sqrt{T}, 1/\beta\}$.

## Key Experimental Results

### Main Results (Table 1, Large-Scale Experiments)

| Dataset | Noise Rate | Ours | Random | Crest (SOTA) |
|---------|-----------|------|--------|--------------|
| SNLI | 0.0 | **0.9132** | 0.9046 | 0.9098 |
| SNLI | 0.5 | **0.6062** | 0.5316 | 0.5104 |
| TinyImageNet | 0.0 | 0.5732 | 0.5520 | 0.5609 |
| TinyImageNet | 0.5 | **0.3644** | 0.2857 | 0.3567 |
| ImageNet-1k | 0.0 | 0.7091 | 0.7074 | **0.7136** |
| ImageNet-1k | 0.5 | **0.6388** | 0.5939 | 0.6051 |

### Ablation Study (Table 2, Sampling Layer Selection + Extreme Noise)

| Configuration | CIFAR-10 (0.5 noise) | CIFAR-100 (0.5 noise) | Notes |
|--------------|---------------------|----------------------|-------|
| BN-layer sampling | **0.7318** | **0.5014** | Best choice |
| All-layer sampling | 0.7288 | — | Higher cost, slightly worse |
| FC-layer sampling | 0.7232 | — | Worst performance |
| Ours (0.8 noise) | **0.3701** | **0.1680** | Effective under extreme noise |
| Crest (0.8 noise) | 0.1520 | 0.1613 | Severe degradation |

### Key Findings

- Under 50% label noise on SNLI, the proposed method surpasses the next-best baseline (Random) by **7% absolute accuracy**
- On ImageNet-1k with zero noise, Crest is marginally better, but requires **2× the training time** (42h vs. 20h) and **3× the memory**
- Consistent improvements across 6 datasets and multiple architectures (LeNet/ResNet/ViT/RoBERTa)
- Time-to-accuracy is **20%–200% faster** than Crest
- Gradient matching error visualizations show that the proposed method consistently yields better gradient estimates than Craig

## Highlights & Insights

- **Posterior sampling = free curvature information**: This is the core insight—averaging gradients under Gaussian perturbations implicitly captures Hessian information without explicit computation
- **BN layers are the key to coreset stability**: Perturbing only BN layers suffices, consistent with research showing BN layers govern model sharpness
- **The surprising strength of the Random baseline**: Under high noise, random sampling outperforms most carefully designed methods, indicating that complex selection strategies are easily misled by noise. The proposed method is the first coreset approach to reliably surpass Random in this regime
- **Loss landscape smoothing effect**: Visualizations clearly demonstrate that standard Craig coresets produce sharp, irregular loss surfaces, while the proposed method yields smooth surfaces aligned with the full-data landscape

## Limitations & Future Work

- Generalization to non-standard data modalities (audio, video) has not been validated
- The theoretical optimality of the posterior distribution choice (spherical vs. inverse-Hessian vs. ensemble) has not been fully established
- The advantage is smaller in zero-noise, large-dataset settings (e.g., slightly below Crest on ImageNet-1k at 0% noise)
- The theoretical analysis lacks a complete formal explanation for the special effectiveness of BN-layer sampling

## Related Work & Insights

- The paper forms a natural progression with Craig (2020), GradMatch (2021), and Crest (2023) in the coreset methods literature
- The posterior sampling smoothing idea draws from SAM (Sharpness-Aware Minimization) and Bayesian learning
- Shin et al. (2023)'s loss curvature matching is a direct motivation; replacing explicit curvature computation with sampling is more efficient
- The approach may inspire applications of posterior sampling to other data selection problems (e.g., active learning, curriculum learning)

## Rating

- Novelty: ⭐⭐⭐⭐ The perspective of replacing Hessian computation with posterior sampling is concise and compelling, with a complete theoretical connection established
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6+ datasets, 4+ architectures, multiple noise levels, time/memory efficiency analysis, and ablations over posterior choices
- Writing Quality: ⭐⭐⭐⭐ Motivation is vividly presented (loss landscape visualizations), with tight integration of theory and experiments
- Value: ⭐⭐⭐⭐⭐ Highly practical—simple, efficient, and robust, improving coreset quality with virtually no additional overhead

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Exploring Landscapes for Better Minima along Valleys](exploring_landscapes_for_better_minima_along_valleys.md)
- [\[NeurIPS 2025\] FedRTS: Federated Robust Pruning via Combinatorial Thompson Sampling](fedrts_federated_robust_pruning_via_combinatorial_thompson_sampling.md)
- [\[ICML 2026\] SVRG and Beyond via Posterior Correction](../../ICML2026/optimization/svrg_and_beyond_via_posterior_correction.md)
- [\[ICML 2025\] Can Transformers Learn Full Bayesian Inference In Context?](../../ICML2025/optimization/can_transformers_learn_full_bayesian_inference_in_context.md)
- [\[ICML 2026\] Convex Basins in Single-Index Model Loss Landscapes: Applications to Robust Recovery under Strong Adversarial Corruption](../../ICML2026/optimization/convex_basins_in_single-index_model_loss_landscapes_applications_to_robust_recov.md)

</div>

<!-- RELATED:END -->
