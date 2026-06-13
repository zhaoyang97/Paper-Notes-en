---
title: >-
  [Paper Note] Quantifying Task-Relevant Representational Similarity Using Decision Variable Correlation
description: >-
  [NeurIPS 2025][LLM Pretraining][Decision Variable Correlation] This paper proposes Decision Variable Correlation (DVC), a novel metric for quantifying trial-by-trial consistency between two neural representations on clas…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "Decision Variable Correlation"
  - "Brain-Model Alignment"
  - "Representational Similarity"
  - "Signal Detection Theory"
  - "Visual Cortex"
date: 2026-05-08
content_hash: 90dfa37c9cd60619
---

# Quantifying Task-Relevant Representational Similarity Using Decision Variable Correlation

**Conference**: NeurIPS 2025
**arXiv**: [2506.02164](https://arxiv.org/abs/2506.02164)  
**Code**: [github.com/wei-bbc-lab/DVC](https://github.com/wei-bbc-lab/DVC)  
**Area**: Computational Neuroscience / Representational Similarity Analysis
**Keywords**: Decision Variable Correlation, Brain-Model Alignment, Representational Similarity, Signal Detection Theory, Visual Cortex

## TL;DR

This paper proposes Decision Variable Correlation (DVC), a novel metric for quantifying trial-by-trial consistency between two neural representations on classification tasks. The authors find that higher ImageNet accuracy in deep networks is associated with *lower* DVC relative to monkey V4/IT, and that neither adversarial training nor large-scale dataset pretraining closes this gap.

## Background & Motivation

Comparing the representations of deep networks and the brain is a central problem in computational neuroscience. Existing approaches fall into two categories:

**Representational similarity methods** (RSA, CKA, CCA): measure the geometric similarity of representational structure, but are sensitive to task-irrelevant dimensions.

**Behavioral consistency methods** (Cohen's Kappa, error consistency): focus on image-level decision consistency, but are susceptible to confounds from accuracy differences and decoder bias.

A prominent tension in the literature is that some studies report high similarity between deep networks and the brain, while others find limited and saturating alignment. This discrepancy is partly attributable to limitations of the evaluation methods themselves.

The motivation of this paper is to propose a **principled, task-relevant, accuracy- and decision-bias-insensitive** similarity metric that unifies the advantages of both representational and behavioral analyses.

## Method

### Overall Architecture

The DVC framework is grounded in a generalization of Signal Detection Theory (SDT):

1. For a binary classification task, each observer (brain region or network layer) uses a continuous **decision variable (DV)** to make a choice.
2. For two observers, the correlation between their decision variables computed over the same set of images is defined as the DVC.
3. DVC captures the **similarity in the encoding and decoding strategies employed by the two observers on the given classification task**.

The key innovation is that DVC is estimated directly from high-dimensional neural representations, rather than inferred from behavioral choice data.

### Key Designs

**Decoding Decision Variables from Neural Representations**

For each pair of categories (e.g., cat vs. dog):
1. Linear Discriminant Analysis (LDA) is applied to identify the axis that maximally separates the two classes.
2. High-dimensional representations are projected onto this axis to obtain per-image decision variable values.
3. To address instability in high-dimensional settings, PCA is first applied to reduce representations to a common dimensionality (25 dimensions) before LDA.

**Split-Half Noise Correction Procedure**

Measurement noise attenuates DVC estimates. A split-half correction procedure is designed to address this:
- Each observer's representations are split into two halves: $\text{DV}_{A1}, \text{DV}_{A2}$ and $\text{DV}_{B1}, \text{DV}_{B2}$
- The noise-corrected Pearson correlation is:

$$\rho_{\text{corrected}} = \frac{r_{\text{cross}}}{r_{\text{self}}}$$

where $r_{\text{cross}}$ is the geometric mean of cross-observer correlations, and $r_{\text{self}}$ is the geometric mean of within-observer split-half reliabilities. This removes the attenuation bias introduced by independent additive symmetric noise.

**Multi-Class Extension**

For $N > 2$ categories, DVC is computed separately for each pair of classes, and the final reported DVC is the average across all pairwise values.

### Loss & Training

This paper does not involve model training; DVC is an analysis method. The core evaluation uses the Pearson correlation coefficient as the measure of DVC.

## Key Experimental Results

### Main Results

A publicly available monkey V4/IT dataset (2 monkeys, ~100 neurons each, 400 images per class across 8 object categories, 3,200 images total) was used to evaluate 43 ImageNet-1k pretrained models.

| Comparison | Mean DVC | Notes |
|---|---|---|
| Monkey–Monkey (V4+IT) | 0.57 | High inter-monkey consistency |
| Monkey–Monkey (V4 only) | 0.63 | Highest consistency in V4 |
| Monkey–Monkey (IT only) | 0.41 | Slightly lower in IT |
| Model–Monkey (43 models avg.) | 0.29 ± 0.05 | Significantly below inter-monkey |
| Adversarially trained–Monkey (9 models) | 0.27 ± 0.02 | Slight decrease |
| Large-dataset models–Monkey (13 models) | 0.24 ± 0.05 | Further decrease |

**Model accuracy vs. DVC with monkey brain**:

| Pearson $r$ | $p$-value | Conclusion |
|---|---|---|
| -0.70 | 2.28e-07 | Higher ImageNet accuracy correlates with lower DVC relative to V4/IT |

### Ablation Study

**Model–Model DVC Analysis**

| Comparison Type | DVC | $p$-value |
|---|---|---|
| Same-family model pairs | Significantly higher | 1.33e-56 |
| Different-family model pairs | Lower | — |
| Adversarially trained models (inter-model) | 0.69 ± 0.09 | Highly consistent |
| Adversarial vs. standard models | Significantly lower | 5.20e-37 |

**Comparison with Cohen's Kappa**

| Metric | Model–Brain | Model–Model | Monkey–Monkey |
|---|---|---|---|
| DVC (proposed) | 0.29 ± 0.05 | Moderate, not extreme | 0.57 |
| Cohen's Kappa (cross-validated LR) | 0.13 ± 0.04 | 0.23 ± 0.07 | 0.22 |
| Cohen's Kappa (original decoder) | Extremely low | Extremely high | — |

The inflated inter-model Cohen's Kappa values arise from decision bias introduced by the decoder. When replaced with cross-validated logistic regression, Cohen's Kappa results align closely with DVC.

### Key Findings

1. **Negative correlation**: Networks with higher ImageNet accuracy exhibit lower trial-by-trial agreement with monkey V4/IT on task-relevant dimensions ($r = -0.70$), contradicting the positive correlation trend reported by BrainScore and related work.
2. **Adversarial training paradox**: Adversarial training makes models highly consistent with each other (DVC = 0.69), but does not improve—and slightly decreases—their alignment with the brain.
3. **Large datasets do not help**: Models pretrained on large-scale datasets (ImageNet-21k, JFT-300M) show lower DVC (0.24) with the monkey brain, not higher.
4. **The pitfall of Cohen's Kappa**: This metric is highly sensitive to accuracy differences and decision bias, potentially misleading researchers into concluding high inter-model consistency and near-zero model-brain consistency.

## Highlights & Insights

- DVC is a principled metric that naturally decouples accuracy from trial-by-trial consistency within the signal detection theory framework—something that existing methods (RSA, Cohen's Kappa) struggle to achieve.
- The split-half noise correction procedure is elegant and effective, requiring no strong assumptions about the noise structure.
- The empirical findings challenge the popular hypothesis that "better networks = more brain-like," revealing potentially **fundamental strategic differences** between deep network and primate visual representations.
- The systematic reanalysis of the Cohen's Kappa literature constitutes an important methodological contribution in its own right.

## Limitations & Future Work

1. Only two monkeys' data are used; the limited sample size warrants validation with larger-scale neural recording datasets.
2. Dimensionality reduction of high-dimensional feature spaces may discard information, though the authors provide robustness checks across multiple dimensionality settings.
3. Visual behavior may differ between monkeys and humans; generalizing conclusions to humans requires caution.
4. The noise correction assumes additive symmetric noise; recovering DVC under more general noise conditions remains an open problem.
5. Experiments are restricted to object classification tasks and do not cover more complex visual tasks.

## Related Work & Insights

- **BrainScore** (Schrimpf et al.): Reports a positive (but saturating) correlation between accuracy and brain alignment; DVC reveals an opposing trend.
- **Error Consistency** (Geirhos et al.): Behavioral consistency analysis based on Cohen's Kappa; the present work identifies its susceptibility to decoder bias.
- **Platonic Representation Hypothesis** (Huh et al.): Posits that deep learning systems converge toward a common representation; DVC results present a more nuanced picture.
- Future directions include: combining behavioral and representational DVC analyses; training on ecologically relevant datasets; and incorporating internal noise models during training.

## Rating

- **Novelty**: ★★★★☆ (Principled extension of SDT to high-dimensional representational comparison with clear theoretical motivation)
- **Experimental Design**: ★★★★★ (Multi-faceted analysis, systematic comparison with Cohen's Kappa, thorough robustness checks)
- **Practicality**: ★★★★☆ (Open-source code, directly applicable to brain-model alignment research)
- **Impact of Findings**: ★★★★★ (Negative correlation and adversarial training paradox challenge prevailing assumptions)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Enhancing Training Data Attribution with Representational Optimization](enhancing_training_data_attribution_with_representational_optimization.md)
- [\[AAAI 2026\] Beyond Cosine Similarity: Magnitude-Aware CLIP for No-Reference Image Quality Assessment](../../AAAI2026/llm_pretraining/beyond_cosine_similarity_magnitude-aware_clip_for_no-reference_image_quality_ass.md)
- [\[NeurIPS 2025\] Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining](breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)
- [\[NeurIPS 2025\] Disaggregation Reveals Hidden Training Dynamics: The Case of Agreement Attraction](disaggregation_reveals_hidden_training_dynamics_the_case_of_agreement_attraction.md)
- [\[NeurIPS 2025\] ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data](zeus_zero-shot_embeddings_for_unsupervised_separation_of_tabular_data.md)

</div>

<!-- RELATED:END -->
