---
title: >-
  [Paper Note] Geometrically Constrained Outlier Synthesis
description: >-
  [ICML 2026][AI Safety][Virtual Outlier Synthesis] GCOS synthesizes virtual outliers along geometric off-manifold directions in the "small-variance subspace" of ID feature PCA. It controls synthesis intensity via a "confo…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Virtual Outlier Synthesis"
  - "Conformal Prediction"
  - "Feature Manifold"
  - "Contrastive Regularization"
  - "Near-OOD"
date: 2026-05-08
content_hash: 227895ca263e8811
---

# Geometrically Constrained Outlier Synthesis

**Conference**: ICML 2026  
**arXiv**: [2603.08413](https://arxiv.org/abs/2603.08413)  
**Code**: None  
**Area**: AI Safety / OOD Detection  
**Keywords**: Virtual Outlier Synthesis, Conformal Prediction, Feature Manifold, Contrastive Regularization, Near-OOD

## TL;DR
GCOS synthesizes virtual outliers along geometric off-manifold directions in the "small-variance subspace" of ID feature PCA. It controls synthesis intensity via a "conformal shell" $[\alpha_\text{inner},\alpha_\text{outer}]$ derived from Mahalanobis quantiles of a calibration set. Combined with a contrastive regularization loss using an adaptive margin, it improves the average AUROC from 86.21 (VOS) to 93.47 across four near-OOD datasets.

## Background & Motivation

**Background**: Image classifiers are generally overconfident on OOD inputs. A mainstream mitigation strategy is Outlier Exposure—constructing "virtual outliers" during training and separating them from ID features via energy regularization. A representative work, VOS, fits Gaussians in the feature space of each class and samples from the tails as virtual outliers.

**Limitations of Prior Work**: Methods like VOS model outliers as samples from simple parametric distributions (e.g., class-conditional Gaussians), which faces two issues: (1) real anomalies are often structured and non-Gaussian, which Gaussian tail sampling fails to cover; (2) if the learned feature space geometry is poor, synthesized points may fall into ID regions or meaningless distant areas, wasting training signals.

**Key Challenge**: Tuning the "difficulty" of synthesized outliers is difficult—points too close to ID are inseparable, while those too far are trivial. VOS uses a fixed probability density threshold, but such thresholds are sensitive to the manifold shape. Furthermore, the field primarily evaluates on far-OOD (semantically unrelated to training), avoiding the more hazardous **near-OOD** (unseen fine-grained classes within the same domain).

**Goal**: (1) Eliminate dependence on preset parametric distributions to allow synthesized points to follow the learned manifold geometry; (2) use a calibratable mechanism that does not require per-dataset tuning to control outlier "strangeness"; (3) shift the evaluation focus to near-OOD.

**Key Insight**: After PCA decomposition, the "large-variance principal components" characterize the main manifold structure, while moving along "small-variance principal components" corresponds to off-manifold directions that are rare yet close to the data center—a natural geometric prior determined by the data itself. Additionally, Conformal Prediction (CP) provides a natural language for "strangeness" via quantiles: using $q_{95}$ and $q_{99}$ of nonconformity scores as thresholds can define a "hard negative band" without manual tuning.

**Core Idea**: Use small-variance PCA directions to determine "where to move" and CP-inspired Mahalanobis quantile shells to determine "how far to move," then apply a contrastive loss to push these geometrically-aware virtual outliers away from ID features.

## Method

### Overall Architecture
The GCOS pipeline operates on the penultimate features $\mathbf{z}\in\mathbb{R}^D$ and is decoupled from the backbone (WRN-40-2 used in experiments). Training occurs across two time scales:

- **Epoch-level Calibration**: Fits a "calibration subspace model" $\mathcal{M}_\text{calib}=(\mu_k,\mathbf{V}_k,\boldsymbol{\Lambda}_k)$ for each class on a calibration set $\mathcal{D}_\text{calib}$, calculating Mahalanobis nonconformity quantiles $q_{95}$ and $q_{99}$ as shell targets for the current epoch.
- **Batch-level Training**: Maintains a rolling queue of ID features to estimate covariance, performs PCA to obtain a "training-side subspace model" $\mathcal{M}_\text{train}$, and extracts small-variance directions $v$. Synthesis $\mathbf{z}_\text{ood}$ is sampled uniformly within the shell $[\alpha_\text{inner},\alpha_\text{outer}]$ along $v$. Finally, the CE classification loss and contrastive regularization loss $\mathcal{L}_{reg}$ are combined for gradient descent.

The final training loss is $\mathcal{L}=\mathcal{L}_{CE}+\lambda\mathcal{L}_{reg}$. Inference follows standard energy score paths (Appendix D provides an extension for conformal hypothesis testing).

### Key Designs

1. **Geometrically-aware Synthesis Direction (off-manifold direction from small PCs)**:
    - **Function**: Provides "rare directions" following the learned manifold without additional generative models.
    - **Mechanism**: Performs eigen-decomposition $(\mathbf{V}_\text{train},\boldsymbol{\Lambda}_\text{train})$ on per-class features. Components explaining the top $\eta$ (default 90%) variance are "large," others are "small." The synthesis direction $v$ is either the average of small PCs or processed per direction. $\mathbf{z}_\text{ood}(\alpha)=\mu+\alpha v$ is synthesized with random sign flipping.
    - **Design Motivation**: Samples in small-variance directions are "near the data center but rare in training," representing the weakest blind spots of OOD detectors. Being data-driven, this avoids external distributional assumptions, bypassing the Gaussian bottleneck of VOS.

2. **Conformal-inspired Shell $[\alpha_\text{inner},\alpha_\text{outer}]$ (conformal shell on synthesis magnitude)**:
    - **Function**: Quantifies the "difficulty of virtual outliers" into a statistically meaningful scalar interval.
    - **Mechanism**: Computes thresholds $q_{95}$ and $q_{99}$ of nonconformity scores (GCOS uses Mahalanobis distance $\mathcal{S}_{Mahal}(z,\mu,\{\lambda_i\},\{v_i\})=\sum_i\frac{((z-\mu)^Tv_i)^2}{\lambda_i+\epsilon}$). $\alpha_\text{inner}$ is the minimum $\alpha$ such that $\mathcal{S}(\mathbf{z}_\text{ood}(\alpha))=q_{95}$, solved via binary search. $\alpha$ is sampled from $\mathcal{U}[\alpha_\text{inner},\alpha_\text{outer}]$.
    - **Design Motivation**: $q_{95}$ and $q_{99}$ correspond to standard 0.05 and 0.01 significance levels, acting as "principled defaults." This shell excludes points that are "too ID-like" ($< q_{95}$) or "too trivial" ($> q_{99}$), making hard negative mining geometrically controllable. Note: CP is used as a geometric heuristic here; training does not enjoy CP coverage guarantees as the calibration set is part of the feedback loop.

3. **Adaptive Margin Contrastive Regularization Loss $\mathcal{L}_{reg}$**:
    - **Function**: Directly separates distributions of ID and synthesized OOD features under the chosen score to improve inference.
    - **Mechanism**: $\mathcal{L}_{reg}=\mathbb{E}[\max(0,\mathcal{S}_\mathcal{L}(\mathbf{z}_{id}|\mathcal{M}_{y_{id}})-\min_k\mathcal{S}_\mathcal{L}(\mathbf{z}_{ood}|\mathcal{M}_k)+m)]$. GCOS uses Mahalanobis for synthesis and Energy Strangeness Score $\mathcal{S}_\mathcal{L}(\mathbf{z})=\log\sum_i w_i\exp(h_\phi(\mathbf{z})_i)$ for $\mathcal{L}_{reg}$. The margin $m$ is adaptive, set to the difference between the 95th and 50th percentiles of positive scores within each batch.
    - **Design Motivation**: Fixed margins are often too loose or tight as score scales shift. Using batch-wise inter-quantile range as a margin allows it to contract with the score distribution without tuning. Decoupling Mahalanobis for "geometric guidance" and Energy for "inference consistency" yields the best performance.

### Loss & Training
Total loss $\mathcal{L}=\mathcal{L}_{CE}+\mathcal{L}_{reg}$. To mitigate the breakdown of exchangeability, two independent calibration sets are maintained: one for online synthesis/regularization and one for final conformal hypothesis testing at inference. $\eta$ (PCA threshold) is 90% by default; $q_{95}/q_{99}$ require no tuning.

## Key Experimental Results

### Main Results
Four near-OOD datasets: Colored MNIST, Stanford Dogs, MVTec, and Retinopathy. Backbone: WRN-40-2.

| Dataset | Metric | GCOS | VOS | NCIS (Prev. SOTA) | Gain |
|--------|------|------|-----|---------------|------|
| C-MNIST | AUROC / FPR95 | **99.50 / 1.00** | 94.71 / 18.50 | 96.72 / 24.50 | +2.78 AUROC, −23.5 FPR95 |
| Dogs | AUROC / FPR95 | **99.55 / 0.00** | 99.25 / 5.00 | 99.35 / 10.00 | +0.20 AUROC, −10 FPR95 |
| MVTec | AUROC / FPR95 | 95.61 / 23.08 | 80.37 / 70.77 | **96.50 / 3.08** | Slightly below NCIS |
| Retinopathy | AUROC / FPR95 | **79.23 / 73.00** | 70.52 / 80.00 | 75.29 / 85.50 | +3.94 AUROC, −12.5 FPR95 |
| **Average AUROC** | — | **93.47** | 86.21 | 91.97 | **+1.50 vs SOTA** |

### Ablation Study
| Configuration | Avg. AUROC | Note |
|------|-----------|------|
| GCOS Full (Mahalanobis Synthesis + Energy Reg) | 93.47 | Default |
| Mahalanobis Regularization | App. H | Decoupling is superior |
| VOS-style Uncertainty Loss | App. H | Validates synthesis strategy |
| Direction: Average vs Per direction | App. J | Per direction is finer but costlier |
| Variance Threshold $\eta$ | App. J | Robust to 90% default |
| No Regularization Baseline | 84.64 | Drops ~9 AUROC |

### Key Findings
- On near-OOD, geometric synthesis plus energy inference is significantly more lightweight and effective than heavy synthesis schemes like NCIS (diffusion/flow-based).
- In C-MNIST, FPR95 dropped from 18.5% (VOS) to 1.0%, indicating that adaptive per-class calibration is critical for complex geometries.
- UMAP visualizations (Fig. 2) show VOS outliers scattered near class boundaries (risking decision boundary collapse), while GCOS outliers fall in off-manifold regions "outside" adjacent clusters, forcing the decision boundary to shrink tightly around data clusters.

## Highlights & Insights
- Formalizing the "strength of hard negative sampling" using CP quantiles ($q_{95}/q_{99}$) provides a principled default for outlier synthesis without per-dataset threshold tuning.
- The observation "Small-variance PC = off-manifold rare direction" is a simple yet effective tool: PCA alone provides better manifold-aligned synthesis than diffusion/flow models for training-time outlier exposure.
- The adaptive margin using batch-wise quantile differences is a reusable trick for any max-margin or triplet-style loss where scores drift during training.

## Limitations & Future Work
- The authors acknowledge that CP coverage guarantees only strictly hold during post-hoc inference (Appendix D); during training, it serves only as a geometric heuristic.
- GCOS relies on a separable feature space. For long-tailed or overlapping distributions, per-class covariance estimation becomes unstable, degrading the meaning of small-variance directions.
- The evaluation is limited to 4 relatively small datasets; performance on large-scale benchmarks like OpenOOD is not yet demonstrated.
- Future work could integrate small-variance PCs with diffusion-based synthesis to combine manifold alignment with image-space diversity.

## Related Work & Insights
- **vs VOS (Du et al., 2022)**: VOS uses Gaussian tail sampling which lacks manifold alignment. GCOS replaces this with PCA directions and conformal shells, improving average AUROC by +7.26.
- **vs NCIS (Doorenbos 2024)**: NCIS synthesizes in image space using complex flows. GCOS operates in feature space, reducing cost while outperforming NCIS by 1.50 AUROC points on average.
- **vs ViM (Wang et al., 2022)**: ViM uses PCA residuals only at inference. GCOS demonstrates that bringing this geometric insight into the training phase via synthesis and regularization is far more effective for shaping the feature space.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of small-variance PCA and CP shells is elegant and novel.
- Experimental Thoroughness: ⭐⭐⭐ Limited to 4 small-scale datasets; lacks certain large-scale benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, honest discussion of CP boundaries, well-structured.
- Value: ⭐⭐⭐⭐ Lightweight and modular; demonstrates strong utility for safety-sensitive near-OOD applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient Resource-Constrained Training of Transformers via Subspace Optimization](../../ICLR2026/ai_safety/efficient_resource-constrained_training_of_transformers_via_subspace_optimizatio.md)
- [\[ICML 2026\] Limits of Convergence-Rate Control for Open-Weight Safety](limits_of_convergence-rate_control_for_open-weight_safety.md)
- [\[ICML 2026\] VPD-100K: Towards Generalizable and Fine-grained Visual Privacy Protection](vpd-100k_towards_generalizable_and_fine-grained_visual_privacy_protection.md)
- [\[ICML 2026\] Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods](extending_fair_null-space_projections_for_continuous_attributes_to_kernel_method.md)
- [\[ICML 2026\] OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL](omnivl-guard_towards_unified_vision-language_forgery_detection_and_grounding_via.md)

</div>

<!-- RELATED:END -->
