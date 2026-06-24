---
title: >-
  [Paper Note] Beyond Entropy: Region Confidence Proxy for Wild Test-Time Adaptation
description: >-
  [ICML 2025][Test-Time Adaptation] Reveals the fundamental limitation of entropy minimization in wild test-time adaptation (WTTA)—conflicting optimization dynamics caused by inconsistent predictions of semantically similar samples in local regions. Proposes the ReCAP framework, which models regions probabilistically and utilizes a finite-to-infinite asymptotic approximation to convert the intractable region confidence into an efficiently optimizable proxy objective…
tags:
  - "ICML 2025"
  - "Test-Time Adaptation"
  - "Entropy Minimization"
  - "Region Confidence"
  - "Distribution Shift"
  - "Wild Scenarios"
date: 2026-05-08
content_hash: 3e9b22c60e0ac7fd
---

# Beyond Entropy: Region Confidence Proxy for Wild Test-Time Adaptation

**Conference**: ICML 2025  
**arXiv**: [2505.20704](https://arxiv.org/abs/2505.20704)  
**Code**: [https://github.com/hzcar/ReCAP](https://github.com/hzcar/ReCAP)  
**Area**: Others/Test-Time Adaptation  
**Keywords**: Test-Time Adaptation, Entropy Minimization, Region Confidence, Distribution Shift, Wild Scenarios

## TL;DR
Reveals the fundamental limitation of entropy minimization in wild test-time adaptation (WTTA)—conflicting optimization dynamics caused by inconsistent predictions of semantically similar samples in local regions. Proposes the ReCAP framework, which models regions probabilistically and utilizes a finite-to-infinite asymptotic approximation to convert the intractable region confidence into an efficiently optimizable proxy objective, consistently outperforming the state-of-the-art on ImageNet-C.

## Background & Motivation

**Background**: Test-time adaptation (TTA) adapts a source model to a target distribution on the fly during inference. Mainstream methods center around entropy minimization—reducing prediction uncertainty to enhance adaptation to the target domain.

**Limitations of Prior Work**:
   - In "wild" scenarios (Wild TTA)—where extreme data scarcity and multiple distribution shifts coexist—entropy minimization faces severe local inconsistency issues.
   - Phenomenon: Neighboring samples with semantic similarities can yield drastically different predictions, causing conflicting optimization gradient directions for these samples' entropy, leading to high optimization noise and low efficiency.
   - Existing solutions (such as SAR, DeYO) mitigate this by filtering "bad samples" via sample selection, but they do not address the root cause—entropy itself is flawed as an optimization objective.

**Key Challenge**: Entropy is defined sample-by-sample, completely ignoring the relationships between samples in a local region. When neighboring samples have inconsistent predictions, minimizing their individual entropy amplifies conflicts.

**Goal**: Replace sample-wise entropy with "region confidence" (which considers both global bias and local variance) as the optimization objective.

**Key Insight**: Region confidence = bias term (average region entropy, global certainty) + variance term (prediction divergence within the region, local consistency). However, direct optimization is computationally intractable (requiring traversal of all samples in the region).

**Core Idea**: Probabilistic region modeling (representing the region as a multivariate Gaussian distribution in the feature space) + finite-to-infinite asymptotic approximation (transforming region confidence into a tractable upper-bound proxy).

## Method

### Overall Architecture
ReCAP replaces the entropy minimization step in standard TTA:
1. **Probabilistic Region Modeling**: For each test sample, its local region in the feature space is defined as a multivariate Gaussian distribution.
2. **Region Confidence Calculation**: Composed of bias (average region entropy) + variance (region prediction divergence).
3. **Asymptotic Proxy Optimization**: Through a finite-to-infinite approximation, the two intractable terms above are converted into analytically optimizable upper bounds.

### Key Designs

1. **Probabilistic Region Modeling**:

    - Function: Defines the local region of each sample as a probability distribution in the feature space (rather than a fixed window or KNN).
    - Mechanism: For the feature $f(x)$ of a test sample $x$, estimate the multivariate Gaussian distribution $\mathcal{N}(\mu_r, \Sigma_r)$ of the local region using the running mean and variance.
    - Design Motivation:
        - Fixed window: Assumes spatial proximity equals semantic similarity, which fails for unstructured data.
        - KNN: Requires storage and retrieval, with computational overhead scaling linearly with buffer size.
        - Probability distribution: Adaptively captures semantic variations in the feature space, and updating parameters has constant complexity.

2. **Mathematical Definition of Region Confidence**:

    - Function: Replaces sample-wise entropy with two statistics.
    - Bias term $B$: Average prediction entropy of samples in the region = $\mathbb{E}_{x' \in R}[H(p(y|x'))]$—measures the overall certainty of the region.
    - Variance term $V$: Prediction divergence among samples in the region = $\mathbb{E}_{x' \in R}[D_{KL}(p(y|x') || \bar{p})]$—measures local consistency in the region.
    - Region confidence = $B + \lambda V$—minimizes both uncertainty and inconsistency simultaneously.
    - Design Motivation: Minimizing only the bias is equivalent to entropy minimization (which does not solve the consistency issue). Adding the variance term forces the optimization to promote consistent predictions among neighboring samples.

3. **Finite-to-Infinite Asymptotic Proxy**:

    - Function: Transforms the optimization objective, which requires traversing all samples in the region (intractable), into an analytically optimizable upper bound.
    - Mechanism:
        - Finite sampling analysis: Estimate $B$ and $V$ by drawing $K$ samples from the region distribution.
        - Asymptotic approximation ($K \to \infty$): Leverage Gaussian properties to transform expectations into functions of distribution parameters $(\mu_r, \Sigma_r)$.
        - Upper-bound derivation: $B + \lambda V \leq \tilde{B}(\mu_r) + \lambda \tilde{V}(\Sigma_r)$, where the upper bound can be efficiently computed in a single forward pass.
    - Design Motivation: Reduces the computation complexity of "traversing neighbors" from $O(K \cdot C)$ to $O(C)$ (where $C$ is the number of classes), enabling real-time execution of the method.

### Loss & Training
- Proxy Loss: $\mathcal{L}_{\text{ReCAP}} = \tilde{B} + \lambda \tilde{V}$
- Only BatchNorm affine parameters are updated (consistent with Tent).
- Orthogonal and stackable with any sample selection method (such as SAR, DeYO).
- Online update of the mean and variance of the region distribution (exponential moving average).

## Key Experimental Results

### Main Results
15 corruption shifts on ImageNet-C (ResNet50, Wild Scenarios = unbalanced label shift + mixed shifts + online shifts):

| Method | Scenario 1: Unbalanced | Scenario 2: Mixed | Scenario 3: Online | Avg Gain |
|------|-----------|----------|----------|---------|
| Tent (Entropy Min.) | 46.3 | 44.8 | 43.2 | Baseline |
| SAR (Selection+Entropy) | 48.1 | 46.5 | 45.7 | +2.0 |
| DeYO (Transform+Selection+Entropy) | 49.2 | 47.3 | 46.8 | +3.0 |
| **Ours (ReCAP)** | **50.1** | **48.6** | **48.0** | **+4.1** |
| **SAR+ReCAP** | **51.3** | **49.2** | **49.1** | **+5.3** |

### ViT Experiments

| Method | ImageNet-C (ViT-B/16) | Gain |
|------|---------------------|------|
| Tent | 62.5 | Baseline |
| DeYO | 64.8 | +2.3 |
| **Ours (ReCAP)** | **65.9** | **+3.4** |

### Ablation Study

| Configuration | Acc | Description |
|------|-----|------|
| Standard Entropy Minimization | 46.3 | Ignores regions |
| Bias Term Only (Avg Region Entropy) | 47.8 | No consistency constraint |
| Variance Term Only (Prediction Consistency) | 47.2 | No certainty optimization |
| **Bias + Variance** | **50.1** | Complete region confidence |
| Ours (ReCAP) + SAR Selection | **51.3** | Orthogonal stacking with selection methods |
| Fixed K-Neighbors (Non-probabilistic Modeling) | 48.9 | Less flexible than probabilistic modeling |
| **Probabilistic Region Modeling** | **50.1** | Adaptive region range |

### Key Findings
- The contribution of the region variance term (+1.5%) is almost equal to that of the bias term (+1.5%)—both are equally important.
- ReCAP is orthogonal to sample selection methods (SAR/DeYO)—stacking them yields an additional 1.2% improvement.
- Probabilistic region modeling outperforms KNN-based region definition (+1.2%)—implying that the Gaussian assumption in the feature space is reasonable.
- Computational overhead of the asymptotic proxy is negligible—increasing computing time by <5% compared to the baseline Tent.
- Effective on both ResNet and ViT—rendering the method robust/insensitive to architectures.

## Highlights & Insights
- **A paradigm shift of "Beyond Entropy"**—instead of refining details of entropy minimization, it fundamentally replaces the optimization objective.
- Clear optimization intuition from bias+variance—certainty (knowing the answer) + consistency (neighbors agreeing) = trustworthy adaptation.
- Elegant theoretical derivation of the finite-to-infinite asymptotic approximation—turning expensive neighbor traversal into constant-time statistical distribution operations.
- Being orthogonal to existing selection methods makes ReCAP a general "underlying optimization upgrade".
- Addressed an overlooked but fundamental issue—questioning a long-standing default assumption in the TTA field (entropy minimization = optimal).

## Limitations & Future Work
- The Gaussian assumption of probabilistic regions might not hold in certain feature spaces.
- $\lambda$ (bias-variance trade-off) is a hyperparameter.
- Under extreme sample scarcity (e.g., only 1-2 samples per batch), the region statistics estimation becomes unreliable.
- The drift problem of region distributions in continual TTA remains undiscussed.
- Only validated on classification tasks; other tasks like detection/segmentation remain to be explored.

## Related Work & Insights
- **vs Tent**: Standard sample-wise entropy minimization without considering local consistency; ReCAP replaces it with region confidence.
- **vs SAR/DeYO**: Filter out noisy samples through sample selection (orthogonal to optimizing the objective itself); ReCAP improves the optimization objective itself.
- **vs EATA**: Uses Fisher information weight updates, still based on entropy; ReCAP goes beyond entropy.
- **Insights**: Other entropy-based methods (like active learning, pseudo-labeling in semi-supervised learning) may suffer from similar local inconsistency issues—the idea of region confidence could be generalized.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Questions and replaces the long-standing default optimization objective in the field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ ResNet+ViT, three Wild scenarios, stacked with multiple methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Thorough analysis of issues, clear asymptotic derivation.
- Value: ⭐⭐⭐⭐⭐ Directional contribution to WTTA and distribution shift adaptation research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Ranked Entropy Minimization for Continual Test-Time Adaptation](ranked_entropy_minimization_for_continual_test-time_adaptation.md)
- [\[NeurIPS 2025\] Test-Time Adaptation by Causal Trimming](../../NeurIPS2025/others/test-time_adaptation_by_causal_trimming.md)
- [\[NeurIPS 2025\] SPACE: SPike-Aware Consistency Enhancement for Test-Time Adaptation in Spiking Neural Networks](../../NeurIPS2025/others/space_spike-aware_consistency_enhancement_for_test-time_adaptation_in_spiking_ne.md)
- [\[CVPR 2025\] Effortless Active Labeling for Long-Term Test-Time Adaptation](../../CVPR2025/others/effortless_active_labeling_for_long-term_test-time_adaptation.md)
- [\[ECCV 2024\] MemBN: Robust Test-Time Adaptation via Batch Norm with Statistics Memory](../../ECCV2024/others/membn_robust_test-time_adaptation_via_batch_norm_with_statistics_memory.md)

</div>

<!-- RELATED:END -->
