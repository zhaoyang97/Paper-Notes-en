---
title: >-
  [Paper Note] Double Descent Meets Out-of-Distribution Detection: Theoretical Insights and Empirical Analysis
description: >-
  [NeurIPS 2025][double descent] This paper is the first to reveal a double descent phenomenon in post-hoc OOD detection—OOD detection performance exhibits a valley near the interpolation threshold as model width increases, then recovers—provides a theoretical explanation via random matrix theory, and proposes an NC1 criterion based on Neural Collapse to identify the optimal model complexity regime.
tags:
  - NeurIPS 2025
  - double descent
  - OOD detection
  - model complexity
  - random matrix theory
  - Neural Collapse
date: 2026-05-08
content_hash: ea9c06bbaa6efe51
---

# Double Descent Meets Out-of-Distribution Detection: Theoretical Insights and Empirical Analysis

**Conference**: NeurIPS 2025
**arXiv**: [2411.02184](https://arxiv.org/abs/2411.02184)
**Code**: Available (link provided in paper)
**Area**: Other
**Keywords**: double descent, OOD detection, model complexity, random matrix theory, Neural Collapse

## TL;DR
This paper is the first to reveal a double descent phenomenon in post-hoc OOD detection—OOD detection performance exhibits a valley near the interpolation threshold as model width increases, then recovers—provides a theoretical explanation via random matrix theory, and proposes an NC1 criterion based on Neural Collapse to identify the optimal model complexity regime.

## Background & Motivation

**Background**: OOD detection is critical for ensuring the reliability of ML systems. Post-hoc methods (e.g., MSP, Energy, Mahalanobis) are widely adopted as they require no modification to the training procedure. Separately, double descent has been extensively studied in the context of in-distribution (ID) generalization, where test error peaks near the interpolation threshold before declining again with increasing model complexity.

**Limitations of Prior Work**: Despite widespread attention to double descent in ID generalization, its behavior in OOD detection remains entirely unexplored. In practice, larger models are commonly assumed to yield better OOD detection, yet this assumption has never been systematically validated.

**Key Challenge**: Does the benefit of overparameterization for ID generalization transfer to OOD detection? If double descent also occurs in OOD detection, model selection strategies may require fundamental reconsideration.

**Goal**: (a) Verify whether OOD detection exhibits double descent; (b) provide a theoretical explanation; (c) propose a principled approach for selecting model complexity when overparameterization is no longer optimal.

**Key Insight**: The paper defines an expected OOD risk as the evaluation metric and derives its relationship with model complexity $p/n$ under a Gaussian covariate model using random matrix theory.

**Core Idea**: OOD detection also exhibits double descent; overparameterization is not always optimal; and the NC1 metric from Neural Collapse can predict which complexity regime is more suitable for OOD detection.

## Method

### Overall Architecture
The paper is structured into theoretical and empirical components. Theoretically, upper and lower bounds on OOD risk are derived under a binary Gaussian model, proving divergence at $p/n=1$. Empirically, model width is varied across CNN, ResNet, ViT, and Swin Transformer architectures, and the AUC of 11 post-hoc OOD detection methods is evaluated as a function of model complexity.

### Key Designs

1. **Expected OOD Risk Definition**:

   - Function: Provides a unified measure of classifier confidence on both ID and OOD data.
   - Core formula: $R_{\text{OOD}}(\hat{f}) = \mathbb{E}_{P}[(\hat{f}(x) - f^{\text{OOD}}(x))^2] + \mathbb{E}_{P^{\text{OOD}}}[(\hat{f}(x) - f^{\text{OOD}}(x))^2]$
   - Here $f^{\text{OOD}}(x)$ approaches 0.5 (uncertainty) on OOD samples and approaches $f^*(x)$ (high confidence) on ID samples.
   - Design Motivation: Low OOD risk implies high confidence on ID data and low confidence on OOD data, which is precisely the objective of OOD detection.

2. **Double Descent Theory for OOD Risk (Theorem 1)**:

   - Function: Proves that the expected OOD risk of a least-squares binary classifier diverges near $p \approx n$.
   - Core result: There exist constants $c, C > 0$ such that $c \cdot c(n,p) \leq \mathbb{E}[R_{\text{OOD}}(\hat{f})] \leq C \cdot c(n,p)$, where:
     - Underparameterized ($p \leq n-2$): $c(n,p) = \frac{p}{n-p-1}(\|w^{\text{OOD}}_{\mathcal{T}^c}\|^2 + \sigma^2) + \|w^{\text{OOD}}_{\mathcal{T}^c}\|^2$
     - Interpolation threshold ($n-1 \leq p \leq n+1$): $c(n,p) = +\infty$, risk diverges.
     - Overparameterized ($p \geq n+2$): Contains the term $(1-n/p)\|w^{\text{OOD}}_\mathcal{T}\|^2 + \frac{n}{p-n-1}(\cdot)$, which gradually decreases.
   - Design Motivation: Extends the regression theory of Belkin et al. (2020) to the classification + OOD setting, requiring handling of nonlinear activation functions.

3. **Neural Collapse Criterion (NC1 Metric)**:

   - Function: Determines whether overparameterization is preferable to underparameterization for OOD detection.
   - Mechanism: Computes $NC1_{u/o} = NC1_u / NC1_o$, where $NC1 = \text{Tr}[\Sigma_W \Sigma_B^+ / C]$ measures the ratio of within-class to between-class covariance.
   - Decision rule: $NC1_{u/o} > 1$ indicates that overparameterization yields better class separation and thus superior OOD detection performance.
   - Design Motivation: The accuracy ratio $Acc_{o/u}$ fails to reliably predict OOD detection trends, whereas $NC1_{u/o}$ succeeds—OOD detection depends more on the geometric quality of learned representations.

### Experimental Setup
- Architectures: 4-block CNN, ResNet-18, ResNet-34, ViT, Swin Transformer
- Width range: $k=1$ to $k=128$
- Training: Cross-entropy loss + Adam + 4000 epochs + 20% label noise
- ID datasets: CIFAR-10, CIFAR-100; OOD datasets: Textures, Places365, iNaturalist, ImageNet-O, SUN

## Key Experimental Results

### Main Results: NC1 Metric vs. OOD Detection Performance

| Architecture | NC1_u/o | Acc_o/u | Softmax AUC_u | Softmax AUC_o | Better Regime |
|---|---|---|---|---|---|
| CNN | 0.88 | 0.99 | 76.09 | 75.08 | Underparameterized |
| ResNet-18 | 1.96 | 1.08 | 71.18 | **75.82** | Overparameterized |
| ViT | >1 | >1 | — | Improved | Overparameterized |
| Swin | >1 | >1 | — | Improved | Overparameterized |

### OOD Detection Methods Across Architectures (CIFAR-10 vs. CIFAR-100)

| Method Type | Representative Methods | Double Descent Evident | Notes |
|---|---|---|---|
| Logit-based | MSP, Energy, MaxLogit | Yes | Directly depends on output logits; sensitive to model complexity |
| Feature-based | Mahalanobis, Residual | Partially | Depends on representation space structure; sensitivity varies by architecture |
| Hybrid | ViM, ASH, NECO | Yes | Combines features and logits |

### Key Findings
- All architectures and all logit-based OOD methods exhibit double descent, with AUC valleys near the interpolation threshold.
- Feature-based methods do not consistently exhibit double descent across architectures, indicating that the phenomenon depends on the scoring function type and architecture.
- CNN achieves better OOD detection in the underparameterized regime ($NC1_{u/o}=0.88<1$), while ResNet, ViT, and Swin perform better in the overparameterized regime.
- $NC1_{u/o}$ is a more reliable predictor of the optimal OOD detection regime than $Acc_{o/u}$.

## Highlights & Insights
- **First discovery of double descent in OOD detection**: Challenges the default assumption that larger models are always better, showing that smaller models can sometimes be superior OOD detectors.
- **Theoretical derivation of OOD risk**: Extends classical double descent theory from regression to the classification + OOD setting; although derived under a Gaussian model, findings are consistent with DNN experiments.
- **NC1 as a model selection criterion**: Eliminates the need to train models across all widths; comparing NC1 values in the under- and overparameterized regimes suffices to identify the optimal regime.

## Limitations & Future Work
- Theoretical analysis is restricted to Gaussian covariate models with linear classifiers; rigorous theory for DNNs remains unavailable.
- Only width-wise double descent is studied; depth-wise and epoch-wise variants are not addressed.
- The NC1 criterion is validated on a limited set of architectures; broader generalizability requires further investigation.
- Label noise is a prerequisite for triggering double descent; results are inconsistent in noise-free settings.

## Related Work & Insights
- **vs. Belkin et al. (2019)**: The seminal double descent work focuses solely on ID generalization; this paper extends the framework to OOD detection and derives classification-specific upper and lower bounds.
- **vs. Nakkiran et al. (2021)**: Validates double descent in DNNs; this paper adds an OOD detection dimension to that framework.
- **vs. NECO (Ammar et al. 2024)**: Uses Neural Collapse for OOD detection scoring; this paper inversely uses NC1 to select the optimal model complexity regime.

## Rating
- Novelty: ⭐⭐⭐⭐ First connection between double descent and OOD detection; both the finding and the NC1 criterion are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four architectures × 11 OOD methods × 6 OOD datasets; exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Theory and experiments are clearly organized, though the theoretical assumptions are relatively strong.
- Value: ⭐⭐⭐⭐ Provides practical guidance for OOD detection and surfaces the counterintuitive finding that smaller models can be effective OOD detectors.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Harnessing Feature Resonance under Arbitrary Target Alignment for Out-of-Distribution Node Detection](harnessing_feature_resonance_under_arbitrary_target_alignment_for_out-of-distrib.md)
- [\[NeurIPS 2025\] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](redundancy-aware_test-time_graph_out-of-distribution_detection.md)
- [\[NeurIPS 2025\] Out-of-distribution Generalisation is Hard: Evidence from ARC-like Tasks](out-of-distribution_generalisation_is_hard_evidence_from_arc-like_tasks.md)
- [\[AAAI 2026\] Theoretical and Empirical Analysis of Lehmer Codes to Search Permutation Spaces with Evolutionary Algorithms](../../AAAI2026/others/theoretical_and_empirical_analysis_of_lehmer_codes_to_search_permutation_spaces_.md)
- [\[NeurIPS 2025\] Product Distribution Learning with Imperfect Advice](product_distribution_learning_with_imperfect_advice.md)

</div>

<!-- RELATED:END -->
