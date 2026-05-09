---
title: >-
  [Paper Note] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics
description: >-
  [Medical Imaging] COMPASS constructs conformal prediction intervals by applying linear perturbations along the **low-dimensional subspace most sensitive to the target metric** within the intermediate feature space of a segmentation network. It achieves significantly narrower prediction intervals than conventional CP methods across four medical segmentation tasks while maintaining valid coverage.
tags:
  - Medical Imaging
date: 2026-05-08
content_hash: d4050febbfcba951
---

# COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics

- **Conference**: ICLR2026
- **arXiv**: [2509.22240](https://arxiv.org/abs/2509.22240)
- **Code**: [GitHub](https://github.com/matthewyccheung/compass)
- **Area**: Medical Image Segmentation / Uncertainty Quantification
- **Keywords**: conformal prediction, medical segmentation, uncertainty quantification, feature perturbation, covariate shift

## TL;DR

COMPASS constructs conformal prediction intervals by applying linear perturbations along the **low-dimensional subspace most sensitive to the target metric** within the intermediate feature space of a segmentation network. It achieves significantly narrower prediction intervals than conventional CP methods across four medical segmentation tasks while maintaining valid coverage.

## Background & Motivation

**Background**: In medical image segmentation, clinical value typically lies not in pixel-level accuracy but in **downstream metrics** derived from segmentations (e.g., organ area/volume and other radiomics indices). Conformal Prediction (CP) is a distribution-free uncertainty quantification framework that provides statistical guarantees for predictions.

**Limitations of Prior Work**: (1) Pixel-level CP methods (e.g., those generating pixel-wise prediction sets) provide guarantees misaligned with the scalar metrics that clinicians actually care about; (2) treating the segmentation–metric pipeline as a black box and applying CP directly to the scalar output (Split CP) yields aligned but **inefficient** intervals (too wide), as it ignores the inductive biases of the neural network.

**Key Challenge**: Feature CP (FCP) has been shown to produce tighter intervals by operating in the semantic feature space, but FCP requires solving a complex adversarial optimization problem in high-dimensional feature spaces, which is **computationally infeasible** for the feature dimensions typical of CNNs and Transformers.

**Goal**: Design a computationally feasible Feature CP method that leverages intermediate representations of segmentation networks to generate **efficient** (narrow) and **valid** (coverage-guaranteed) prediction intervals for downstream clinical metrics.

**Key Insight**: Rather than searching over the full-dimensional feature space, the method exploits the **Jacobian of the target metric with respect to features** to identify a low-dimensional sensitive subspace, restricting perturbations to that direction.

**Core Idea**: The Jacobian of the target metric is computed with respect to intermediate-layer features of the segmentation network. PCA extracts the principal directions as perturbation axes; linear perturbation along these directions monotonically changes the metric. Consequently, the prediction interval requires only two forward passes (positive/negative endpoints), enabling efficient construction of nested prediction intervals.

## Method

### Overall Architecture

The segmentation network is decomposed into three components: encoder $f: \mathcal{X} \to \mathcal{Z}$, decoder $g: \mathcal{Z} \to \mathcal{S}$, and metric function $h: \mathcal{S} \to \mathbb{R}$. COMPASS perturbs features $\hat{z}_i$ in $\mathcal{Z}$ along a sample-specific direction $\Delta_i$, propagates the perturbation through $g$ and $h$ to obtain metric variation, and constructs the prediction interval $S_\beta(x) = [\min_{b \in [-\beta, \beta]} m_x(b),\ \max_{b \in [-\beta, \beta]} m_x(b)]$.

### Key Designs

#### 1. Jacobian PCA-Based Sensitive Direction Estimation (COMPASS-J)

**Function**: Identifies the perturbation direction in feature space to which the target metric is **most sensitive**, on a per-sample basis.

**Mechanism**: For each training sample $i$, the Jacobian of the target metric $\hat{y}$ with respect to features $\hat{z}_i$ is computed as $J_i = \frac{d\, h(g(\hat{z}_i))}{d\hat{z}_i}$. Summing over spatial dimensions yields a channel-wise vector $\mathcal{J}_i$. PCA is applied to all $\mathcal{J}_i$ in the training set, retaining the top $L$ principal components $V_L$. For any new sample, the perturbation direction is:

$$\mathbf{d}_i = V_L V_L^T \mathcal{J}_i, \quad \Delta_i = \mathbf{d}_i / \|\mathbf{d}_i\|_2$$

**Design Motivation**: Full-dimensional search is infeasible, but the first principal component of PCA typically explains >90% of metric variance (empirically verified). Perturbation along the principal direction consistently produces **monotonic metric changes** across experiments — this enables efficient evaluation of only two endpoints (positive/negative $\beta$) rather than a full interval scan.

#### 2. Nestedness Guarantee via Linear Perturbation

**Function**: Proves that prediction sets constructed through linear perturbation satisfy nestedness, thereby guaranteeing marginal coverage.

**Mechanism**: The prediction set is defined as the range of the metric over the perturbation interval: $S_\beta(x) = [\min m_x(b),\ \max m_x(b)]_{b \in [-\beta, \beta]}$. Since $\beta_1 \leq \beta_2 \Rightarrow [-\beta_1, \beta_1] \subseteq [-\beta_2, \beta_2]$, the extrema can only grow, ensuring $S_{\beta_1} \subseteq S_{\beta_2}$ (nestedness). The standard CP exchangeability condition combined with nestedness directly guarantees:

$$\mathbb{P}(Y_{n+1} \in S_{\hat{\beta}}(X_{n+1}) | D_{\text{tr}}) \geq 1 - \alpha$$

**Design Motivation**: Nestedness is a necessary condition for CP validity. This is non-trivial in the nonlinear feature space of deep networks; the paper achieves it by construction through a conservative envelope defined as the range of the metric.

#### 3. Weighted COMPASS for Covariate Shift

**Function**: Re-weights calibration samples via density ratios to restore target coverage under covariate shift.

**Mechanism**: An auxiliary classifier is trained to distinguish calibration and test samples, estimating the density ratio $w(X_i) = p_{\text{test}}(X_i) / p_{\text{cal}}(X_i)$. Deep-layer features or Jacobians are used as classifier inputs (richer signals than class labels or logits). A weighted conformal quantile replaces the uniform quantile.

### Loss & Training

COMPASS does not modify segmentation model training. The area metric is computed by applying a soft sigmoid to logits followed by summation (differentiable), enabling Jacobian computation.

## Key Experimental Results

### Main Results: Interval Width Across CP Methods (pixels², Mean±Std, α=0.10)

| Dataset | COMPASS-J | COMPASS-L | E2E-CQR | Local CP | Output-CQR | SCP |
|---------|----------|----------|---------|----------|-----------|-----|
| H&E | **3160±336** | 3139±375 | 3433±293 | 4223±558 | 3879±369 | 3509±333 |
| Skin Lesion | **1179±53** | 1208±58 | 1351±75 | 2433±101 | 4581±36 | 1813±127 |
| Nodule | **2444±174** | 2510±180 | 2788±154 | 3311±133 | 5603±57 | 3076±200 |
| PolyP | **4056±293** | 4397±469 | 6184±616 | 5965±1011 | 4981±675 | 6237±564 |

> COMPASS-J produces the narrowest intervals across all datasets and all $\alpha$ levels. Compared to SCP, intervals are narrowed by **35%** on Skin Lesion and **35%** on PolyP.

### Ablation Study: Weighted CP Under Covariate Shift (α=0.10)

| Method | H&E (hard shift) Coverage | Skin Lesion (easy shift) Coverage |
|--------|--------------------------|----------------------------------|
| Unweighted SCP | ❌ Under-covered | ✅ Over-covered |
| Label-weighted SCP | ✅ | ❌ Under-covered |
| COMPASS-L + Feature weighting | ❌ Under-covered | ✅ |
| COMPASS-J + Feature weighting | ✅ **Narrowest** | ✅ **Narrowest** |
| COMPASS-J + Jacobian weighting | ✅ **Narrowest** | ✅ **Narrowest** |

> Only COMPASS-J (with deep feature or Jacobian weighting) maintains target coverage under **both shift directions** simultaneously, with the narrowest intervals.

### Key Findings

1. **Monotonicity holds universally**: Perturbation along the COMPASS-J direction induces monotonic metric changes across all four datasets, validating the efficient endpoint algorithm.
2. **Power-law compression**: A sublinear scaling relationship (log-log slope <1) exists between the feature-space residual $R_{\text{COMPASS}}$ and the output-space residual $R_{\text{SCP}}$, systematically compressing the tail distribution — the fundamental mechanism underlying tighter intervals.
3. **Deep representations > shallow**: COMPASS-J (deep features) consistently outperforms COMPASS-L (logits), as deep features provide richer metric-sensitive signals.

## Highlights & Insights

- The core idea of "perturbation along sensitive subspaces" is elegantly concise: Jacobian → PCA → a single direction → two endpoints, reducing the intractable optimization of FCP to two forward passes.
- Empirical validation of monotonicity is critical — it is the prerequisite for the efficient algorithm and can be anticipated from the explained variance of the first Jacobian principal component.
- The power-law compression finding provides a principled explanation for COMPASS's efficiency advantage, beyond mere empirical observation.
- The robustness of weighted COMPASS under covariate shift carries direct clinical significance.

## Limitations & Future Work

- COMPASS performance depends on the quality of pretrained model representations — if the feature–metric relationship is non-monotonic, a full scan algorithm is required as a fallback.
- Under large distribution shifts (insufficient overlap between calibration and test feature spaces), density ratio estimation becomes unreliable.
- Only area metrics are validated; applicability to more complex metrics such as texture or shape descriptors remains unexplored.
- Validation is based on U-Net architectures; optimal layer selection for Transformer-based segmentation models may differ.

## Related Work & Insights

- **Feature CP** (Teng et al., 2022): First demonstrated that CP in feature space yields tighter intervals, but adversarial search is computationally infeasible.
- **Lambert et al. (2024)**: End-to-end CQR trains pixel-level upper/lower bounds using Tversky loss, optimizing a surrogate rather than the target metric directly.
- **Split CP / CQR**: Standard output-space methods; simple but yield wide intervals.
- **Insights**: The COMPASS paradigm of "Jacobian → PCA → principal direction perturbation" is generalizable to uncertainty quantification for any differentiable metric (3D volume, shape indices, etc.).

## Rating

⭐⭐⭐⭐⭐ (5/5)

- **Novelty**: ⭐⭐⭐⭐⭐ — Elegantly resolves the computational bottleneck of FCP via Jacobian PCA dimensionality reduction, with solid theoretical proofs and empirical validation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 4 datasets × 3 $\alpha$ levels × 6 baselines × 100 random splits, covering standard, covariate shift, and ablation settings — exceptionally comprehensive.
- **Value**: ⭐⭐⭐⭐⭐ — Open-source, plug-and-play, with direct clinical value for uncertainty quantification of segmentation-derived metrics.
- **Writing Quality**: ⭐⭐⭐⭐ — Theory and experiments are clearly structured with intuitive figures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](../../AAAI2026/medical_imaging/hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](../../AAAI2026/medical_imaging/small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[ICLR 2026\] Protein as a Second Language for LLMs](protein_as_a_second_language_for_llms.md)
- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](thompson_sampling_via_fine-tuning_of_llms.md)
- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](tracing_pharmacological_knowledge_in_large_language_models.md)

</div>

<!-- RELATED:END -->
