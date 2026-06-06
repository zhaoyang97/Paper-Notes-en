---
title: >-
  [Paper Note] Noise-Aware Generalization: Robustness to In-Domain Noise and Out-of-Domain Generalization
description: >-
  [ICLR 2026][LLM Evaluation][Noise-Aware Generalization] This paper is the first to formally define the Noise-Aware Generalization (NAG) problem — simultaneously pursuing in-domain robustness and out-of-domain generalizat…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "Noise-Aware Generalization"
  - "Domain Generalization"
  - "Learning with Noisy Labels"
  - "Cross-Domain Noise Detection"
  - "DL4ND"
date: 2026-05-08
content_hash: 078c657547568c1b
---

# Noise-Aware Generalization: Robustness to In-Domain Noise and Out-of-Domain Generalization

**Conference**: ICLR 2026
**arXiv**: [2504.02996](https://arxiv.org/abs/2504.02996)  
**Code**: [GitHub](https://github.com/SunnySiqi/Noise-Aware-Generalization)  
**Area**: Robust Learning / Domain Generalization
**Keywords**: Noise-Aware Generalization, Domain Generalization, Learning with Noisy Labels, Cross-Domain Noise Detection, DL4ND

## TL;DR

This paper is the first to formally define the Noise-Aware Generalization (NAG) problem — simultaneously pursuing in-domain robustness and out-of-domain generalization under label noise — and proposes DL4ND, a method that detects noisy labels via cross-domain comparison, achieving up to 12.5% improvement across 7 datasets.

## Background & Motivation

**Background**: Domain Generalization (DG) methods train models to generalize from multiple source domains to unseen target domains by learning domain-invariant features; Learning with Noisy Labels (LNL) methods improve model performance by detecting and handling noisy labels. Both fields have achieved significant progress, but have typically been studied in isolation.

**Limitations of Prior Work**:
1. **DG methods ignore label noise**: Label noise is prevalent in real-world datasets (including DG benchmarks themselves), yet DG methods suffer severe performance degradation in its presence.
2. **LNL methods disregard domain shift**: LNL methods detect noise within a single domain, but when applied to multi-domain data, they misidentify domain shift as label noise, leading to overfitting on "easy-to-learn" domains.
3. **Domain shift and noise shift are hard to disentangle**: When analyzing via feature distances or loss values, distributional shifts caused by domain shift and label noise are highly overlapping in feature space (as shown in Figure 1).

**Key Challenge**: The core assumption of LNL noise detection — "noisy samples are far from class centers" — breaks down in multi-domain settings, because domain shift makes the source of distributional deviation (noise vs. domain) indistinguishable via simple feature distances. Naively combining DG and LNL methods also fails to resolve this, as more than 20% of support vectors fall in the overlapping region of the two types of shift, and these samples are critical to the decision boundary.

**Goal**: This paper defines the NAG (Noise-Aware Generalization) problem and designs DL4ND (Domain Labels for Noise Detection). The core insight is that **noisy samples that appear similar within a single domain reveal discrepancies under cross-domain comparison** — because spurious correlations within a domain (e.g., color) do not persist across domains, forcing the model to rely on intrinsic features. DL4ND constructs $(class, domain)$ proxy representations from high-confidence, low-loss samples, then re-labels high-loss samples via cross-domain comparison.

## Method

### Overall Architecture

DL4ND proceeds in three stages:

1. **Warmup Stage**: Standard training using ERM or any DG method, during which the influence of noisy labels is minimal.
2. **Noise Detection Stage**:
    - A GMM is applied to the loss distribution to separate low-loss (clean) and high-loss (noisy) clusters.
    - Low-loss samples are grouped by $(class, domain)$ to construct proxy representations $\bar{g}_{c,i}$.
    - High-loss samples are re-labeled via cross-domain comparison.
3. **Continued Training**: Training resumes with updated labels, optionally combined with any DG method.

Formally, the multi-domain dataset is $\mathcal{D} = \{\mathcal{D}_1, \mathcal{D}_2, \ldots, \mathcal{D}_m\}$, where each domain $\mathcal{D}_i = \{(x_{i,j}, \tilde{y}_{i,j})\}_{j=1}^{n_i}$ and $\tilde{y}$ may be noisy. The objective is to learn a feature extractor $f_\theta(\cdot)$ that performs well on all source domains and unseen target domains.

### Key Design 1: Separability Condition and Low-Loss Proxies

A sufficient condition for separating domain shift from class shift is established theoretically:

$$d(f_\theta(G_{c,\hat{i}}), \bar{g}_{c,i}) < d(f_\theta(G_{\hat{c},i}), \bar{g}_{c,i}), \quad i \neq \hat{i}, \; c \neq \hat{c}$$

That is, the distance between the same class across domains should be smaller than that between different classes within the same domain. Experiments on RotatedMNIST reveal:
- Using **all samples** to construct proxies results in severe overlap between the two types of distributional shift, making them indistinguishable.
- Using **low-loss samples** to construct proxies enables clear separation between the two shift types.

This is because low-loss samples in early training are typically clean (as established by prior work), yielding purer class proxies. Further analysis shows that over 20% of samples in the overlapping region are SVM support vectors, confirming their critical role in determining decision boundaries and that they cannot be simply discarded.

### Key Design 2: Cross-Domain Noise Detection

The core assumption is that intra-domain spurious features (e.g., the golden hue associated with lions in the photo domain) do not persist across domains; thus, cross-domain comparison more accurately reflects the true class of a sample. For a high-loss sample $x_i$ identified as potentially noisy, re-labeling is performed via cross-domain comparison:

$$\hat{y_i} = \arg\min_{\forall g_{c,\hat{i}}} d(f_\theta(x_i), \bar{g}_{c,\hat{i}}), \quad i \neq \hat{i}$$

That is, the nearest-neighbor class is found among proxy representations of **other domains** and assigned as the new label. Compared to within-domain comparison, cross-domain comparison:
- Eliminates domain-specific spurious correlations.
- Forces the model to rely on intrinsic features that are consistent across domains.
- Experimentally yields significantly improved noise detection accuracy (label accuracy on RotatedMNIST improves from 75.7% to 98.1%).

### Key Design 3: Plug-and-Play Integration with DG Methods

DL4ND is a noise detection module that can be seamlessly combined with any DG method (ERM, ERM++, SAGM, SWAD, etc.). The integration is straightforward: the DL4ND label detection and correction step is inserted into the training pipeline of the chosen DG method, without requiring additional data or training overhead. Experiments demonstrate that such combinations outperform using DG or LNL methods in isolation in most settings.

## Key Experimental Results

### Main Results

Evaluation is conducted on 7 datasets, covering both real-world noise and controlled noise experiments.

**RotatedMNIST (30% asymmetric noise)**:

| Method | Label Accuracy | ID Acc | OOD Acc |
|--------|:---:|:---:|:---:|
| Baseline (within-domain comparison) | 75.7 | 87.7 | 87.9 |
| **DL4ND (Ours)** | **98.1** | **98.1** | **97.8** |

**OfficeHome (60% symmetric noise)**:

| Method | ID Acc | OOD Acc | AVG |
|--------|:---:|:---:|:---:|
| ERM | 45.8 | 40.5 | 43.2 |
| ERM + DL4ND | 47.9 | 49.9 | 48.9 |
| SAGM | 48.6 | 40.3 | 44.4 |
| SAGM + DL4ND | 52.0 | 52.6 | 52.2 |
| ERM++ | 56.7 | 48.7 | 52.7 |
| **ERM++ + DL4ND** | **60.3** | **59.4** | **59.8** |

The maximum gain reaches **12.5%** (ERM++ OOD from 48.7% to 59.4% under symmetric noise).

**PACS (real-world noise)**:

| Method | ID Acc | OOD Acc | AVG |
|--------|:---:|:---:|:---:|
| SAGM | 96.3 | 85.3 | 90.8 |
| SAGM + DL4ND | 97.3 | 88.8 | 93.1 |
| ERM++ | 96.7 | 89.2 | 92.9 |
| **ERM++ + DL4ND** | **96.5** | **90.1** | **93.3** |

### Ablation Study

Contributions of individual DL4ND components are ablated across multiple datasets:

| Ablation Configuration | VLCS ID/OOD | CHAMMI-CP ID/OOD | OfficeHome (40% asym) AVG |
|------------------------|:---:|:---:|:---:|
| w/o relabel (removing surrogate re-labeling) | −2–3% | −1–2% | Below full model |
| w/o cross-domain (within-domain comparison) | −2–4% | −2–4% | Significantly lower accuracy |
| w/o small-loss proxy (full-sample proxy) | −2–4% | −2–3% | Lower proxy quality |
| **DL4ND (Full)** | **Best** | **Best** | **Best** |

Each component contributes 2–4% performance improvement. Ablation of cross-domain comparison shows that the accuracy gains it provides (Table 6) directly explain the final performance improvements.

## Highlights & Insights

### Strengths

1. **Valuable problem formulation**: NAG naturally unifies DG and LNL — two previously separate fields — in a setting more aligned with real-world applications.
2. **Solid theoretical analysis**: The separability condition is mathematically formalized and the unique challenges of NAG are clearly articulated through SVM support vector analysis.
3. **Novel cross-domain comparison**: The observation that leveraging inter-domain differences eliminates spurious correlations is simple yet effective.
4. **Extensive experimental validation**: 12 SOTA baselines, 20 combination methods, and 7 datasets provide comprehensive coverage.

### Limitations & Future Work

1. DL4ND relies on GMM to separate low- and high-loss samples; the two-component GMM assumption may not hold when the noise ratio is extremely high.
2. Cross-domain comparison assumes that spurious features differ across domains; if all domains share the same bias (e.g., all have color bias), the method may fail.
3. A single round of re-labeling proves effective, but a thorough analysis of the effects of iterative re-labeling is lacking.

## Rating

⭐⭐⭐⭐ — The problem formulation is practical and important, the method is concise and effective, and the experiments comprehensively cover diverse noise types and datasets. The work makes a significant contribution to the intersection of domain generalization and learning with noisy labels.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Aggregation Hides OOD Generalization Failures from Spurious Correlations](../../NeurIPS2025/llm_evaluation/aggregation_hides_out-of-distribution_generalization_failures_from_spurious_corr.md)
- [\[ICCV 2025\] PHATNet: A Physics-guided Haze Transfer Network for Domain-adaptive Real-world Image Dehazing](../../ICCV2025/llm_evaluation/phatnet_a_physics-guided_haze_transfer_network_for_domain-adaptive_real-world_im.md)
- [\[NeurIPS 2025\] Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](../../NeurIPS2025/llm_evaluation/generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)
- [\[NeurIPS 2025\] Not All Splits Are Equal: Rethinking Attribute Generalization Across Unrelated Categories](../../NeurIPS2025/llm_evaluation/not_all_splits_are_equal_rethinking_attribute_generalization_across_unrelated_ca.md)
- [\[ICLR 2026\] Optimal Transport-Induced Samples against Out-of-Distribution Overconfidence](optimal_transport-induced_samples_against_out-of-distribution_overconfidence.md)

</div>

<!-- RELATED:END -->
