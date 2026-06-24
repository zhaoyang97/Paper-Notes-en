---
title: >-
  [Paper Note] Aggregation Hides OOD Generalization Failures from Spurious Correlations
description: >-
  [NeurIPS 2025 Spotlight][OOD generalization] This paper reveals the "aggregation masking" phenomenon in OOD generalization benchmarks: while aggregate evaluation exhibits accuracy-on-the-line (AoTL)—a positive correlation between ID and OOD accuracy—the proposed OODSelect method can identify large, semantically coherent subsets (up to 75%) from the same OOD data on which higher ID accuracy corresponds to lower OOD accuracy (Pearson R as low as −0.92)…
tags:
  - "NeurIPS 2025 Spotlight"
  - "OOD generalization"
  - "spurious correlations"
  - "aggregation bias"
  - "accuracy-on-the-line"
  - "subset analysis"
date: 2026-05-08
content_hash: d195377f0cd1b784
---

# Aggregation Hides OOD Generalization Failures from Spurious Correlations

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.24884](https://arxiv.org/abs/2510.24884)  
**Code**: [https://github.com/olawalesalaudeen/OODSELECT](https://github.com/olawalesalaudeen/OODSELECT)  
**Area**: OOD Generalization / Robustness
**Keywords**: OOD generalization, spurious correlations, aggregation bias, accuracy-on-the-line, subset analysis

## TL;DR
This paper reveals the "aggregation masking" phenomenon in OOD generalization benchmarks: while aggregate evaluation exhibits accuracy-on-the-line (AoTL)—a positive correlation between ID and OOD accuracy—the proposed OODSelect method can identify large, semantically coherent subsets (up to 75%) from the same OOD data on which higher ID accuracy corresponds to lower OOD accuracy (Pearson R as low as −0.92), demonstrating that the harm of spurious correlations is systematically concealed by aggregate evaluation.

## Background & Motivation

**Background**: The OOD generalization literature has observed accuracy-on-the-line (AoTL)—on DomainBed/WILDS benchmarks, models with higher ID accuracy also tend to achieve higher OOD accuracy. This is commonly interpreted as evidence that spurious correlations are not a serious concern.

**Limitations of Prior Work**: (a) AoTL holds only when aggregating over all OOD samples; (b) when large subgroups are sensitive to spurious features, their effects are diluted by "clean" samples; (c) existing subset discovery methods require explicit group metadata.

**Key Challenge**: AoTL may be an aggregation artifact rather than a genuine positive signal—on certain subgroups, models that achieve higher ID accuracy may rely more heavily on spurious features, leading to worse OOD performance.

**Goal**: To demonstrate that aggregate evaluation masks the harm of spurious correlations and to provide a method for discovering the hidden failure subsets.

**Key Insight**: The problem of OOD subset selection is formulated as an optimization problem—minimizing the Pearson correlation between ID and OOD accuracy over a selected subset.

**Core Idea**: Gradient-based optimization selects accuracy-on-the-inverse-line subsets from OOD data, revealing generalization failures concealed by aggregation.

## Method

### Overall Architecture
Given $N$ models and $d$ OOD samples, a correct-classification matrix $\mathbf{Z} \in \{0,1\}^{N \times d}$ is constructed. A selection vector $\mathbf{s} \in [0,1]^d$ (sigmoid relaxation) is learned to minimize the ID–OOD correlation over the selected subset, optimized with the Adam optimizer.

### Key Designs

1. **OODSelect Optimization**:

    - **Function**: Identify a subset of OOD data on which the ID–OOD correlation is maximally negative.
    - **Mechanism**: The discrete selection problem is relaxed to a continuous variable, optimizing $\min_{\mathbf{s}} \text{corr}(\text{acc}_{ID}, \text{acc}^s_{OOD}) + \lambda(S - \|\mathbf{s}\|_1)^2$, with cosine annealing scheduling. Models are split into training/validation/test sets to prevent overfitting.
    - **Design Motivation**: The objective is neither convex nor submodular (proven by theorem), making greedy approaches infeasible; however, Lipschitz continuity guarantees stable convergence of gradient descent.

2. **Multi-Level Validation**:

    - **Function**: Ensure discovered subsets reflect genuine spurious correlations rather than sampling noise.
    - **Mechanism**: (a) Comparison against random selection (consistently positive correlation); (b) comparison against "most misclassified" samples (near-zero but not negative correlation); (c) Spearman rank correlation to rule out outlier effects; (d) cross-architecture validation (ResNet vs. ViT separation).
    - **Design Motivation**: Negative correlation admits multiple interpretations, necessitating systematic elimination of alternative explanations.

3. **Selection Consistency Validation**:

    - **Function**: Confirm that subsets of different sizes are semantically coherent.
    - **Mechanism**: Independent selections are performed for different subset sizes $S$, and a normalized Jaccard Index is computed—smaller subsets are found to be nearly strict subsets of larger ones.
    - **Design Motivation**: Inconsistent selections would indicate that the optimization is merely fitting noise.

### Loss & Training
- Optimization: Adam + cosine annealing
- Three-way model split: training models (for learning the selection), validation models, and test models (fully non-overlapping)
- Hundreds to thousands of models trained per dataset

## Key Experimental Results

### Main Results

| Dataset | Full OOD R | Max OODSelect R | Subset Size | # Models |
|--------|---------|----------------|---------|--------|
| CXR No Finding | +0.86 | **−0.60** | **75%** | 1800 |
| TerraIncognita | +0.89 | **−0.77** | 25% | 2980 |
| VLCS | +0.62 | **−0.92** | 30% | 4200 |
| WILDSCamelyon-H5 | +0.74 | <−0.3 | 40% | 944 |
| WILDSCivilComments | +0.81 | <−0.3 | 50% | 710 |
| PACS | +0.81 | −0.33 | 6% | 2804 |

### Ablation Study

| Selection Method | CXR R | VLCS R | Note |
|---------|-------|--------|------|
| Random selection | +0.85 | +0.61 | Consistently positive |
| Most misclassified | ~0 | ~0 | Weak correlation, not negative |
| CLIP distance selection | +0.52 | −0.10 | Fails to uncover deep spurious correlations |
| **OODSelect** | **−0.60** | **−0.92** | Discovers spurious-correlation-driven subsets |

### Key Findings
- **75% of CXR OOD data is affected by spurious correlations**—the aggregate R shifts from −0.60 to +0.86 entirely due to dilution by the 25% of clean samples.
- **Spurious correlations ≠ sample difficulty**: the hardest samples yield R~0, while OODSelect samples yield R<0—the two are fundamentally distinct.
- **CXR subsets are semantically coherent**: OODSelect subsets are enriched with Pleural Other and Support Devices labels—known spurious predictors.
- **Cross-architecture consistency**: subsets selected using ResNet models still exhibit negative correlation when evaluated on ViTs.
- **VLMs are not necessarily robust**: zero-shot VLM ID–OODSelect correlation remains positive (both splits are OOD for VLMs).

## Highlights & Insights
- **"Aggregation masking" poses a fundamental challenge to OOD evaluation methodology**—it not only casts doubt on the "good news" interpretation of AoTL, but also suggests that all aggregate-metric OOD benchmarks may systematically underestimate the harm of spurious correlations.
- **The philosophy of OODSelect**: rather than "finding the hardest samples," the goal is to "find samples where spurious correlations cause failures"—an entirely new evaluation perspective.
- **75% of CXR OOD samples are affected**—in medical imaging, relying solely on aggregate accuracy is fundamentally unreliable.

## Limitations & Future Work
- Training thousands of models is computationally expensive (though a one-time cost)—all selection results have been open-sourced.
- Semantic interpretation is difficult: images such as histopathology slides are hard to explain in natural language.
- Non-convex optimization cannot guarantee globally optimal solutions.
- This work serves as a diagnostic tool only; it does not propose methods for correcting training.

## Related Work & Insights
- **vs. Miller et al. (2021)**: They proposed AoTL and argued that spurious correlations are not a serious concern. This paper directly challenges that conclusion—AoTL is an aggregation artifact.
- **vs. Teney et al. (2023)**: They used more diverse models to show that some datasets do not satisfy AoTL. This paper goes further—identifying inverse subsets within the same dataset.
- **vs. SliceFinder/SSD++**: These methods require explicit group metadata; OODSelect requires no metadata whatsoever.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A disruptive finding—AoTL is an aggregation artifact.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 datasets, thousands of models, multi-level validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Arguments build progressively; figures are highly persuasive.
- Value: ⭐⭐⭐⭐⭐ Fundamental implications for OOD evaluation methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] OOD-Chameleon: Is Algorithm Selection for OOD Generalization Learnable?](../../ICML2025/others/ood-chameleon_is_algorithm_selection_for_ood_generalization_learnable.md)
- [\[ICLR 2026\] Regulating Internal Alignment Flows for Robust Learning Under Spurious Correlations](../../ICLR2026/others/regulating_internal_alignment_flows_for_robust_learning_under_spurious_correlati.md)
- [\[NeurIPS 2025\] Impact of Layer Norm on Memorization and Generalization in Transformers](impact_of_layer_norm_on_memorization_and_generalization_in_transformers.md)
- [\[ICML 2025\] NeuronTune: Towards Self-Guided Spurious Bias Mitigation](../../ICML2025/others/neurontune_towards_self-guided_spurious_bias_mitigation.md)
- [\[ACL 2025\] Subword Models Struggle with Word Learning, but Surprisal Hides It](../../ACL2025/others/subword_models_struggle_with_word_learning_but_surprisal_hides_it.md)

</div>

<!-- RELATED:END -->
