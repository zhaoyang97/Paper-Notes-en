---
title: >-
  [Paper Note] Concepts' Information Bottleneck Models
description: >-
  [ICLR 2026][Interpretability][Concept Bottleneck Models] Information Bottleneck (IB) regularization is introduced at the concept layer of Concept Bottleneck Models (CBM) to learn minimal sufficient concept representations by penalizing $I(X;C)$ while preserving $I(C;Y)$. This consistently improves predictive performance and concept intervention reliability across six CBM variants and three benchmarks.
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Concept Bottleneck Models"
  - "Information Bottleneck"
  - "Regularization"
  - "Concept Leakage"
date: 2026-05-08
content_hash: 48f08a44e2bde478
---

# Concepts' Information Bottleneck Models

**Conference**: ICLR 2026  
**arXiv**: [2602.14626](https://arxiv.org/abs/2602.14626)  
**Code**: Yes (mentioned in paper)  
**Area**: Interpretability  
**Keywords**: Concept Bottleneck Models, Information Bottleneck, Interpretability, Regularization, Concept Leakage

## TL;DR
Information Bottleneck (IB) regularization is introduced at the concept layer of Concept Bottleneck Models (CBM) to learn minimal sufficient concept representations by penalizing $I(X;C)$ while preserving $I(C;Y)$. This consistently improves predictive performance and concept intervention reliability across six CBM variants and three benchmarks.

## Background & Motivation

Concept Bottleneck Models (CBMs) represent a class of interpretable AI methods. The core idea is to insert a human-understandable concept layer $C$ between input $X$ and prediction $Y$ to make the decision process transparent. This design allows human experts to perform concept intervention during inference to correct a model's erroneous reasoning.

However, existing CBMs face two fundamental issues:

**Accuracy Drop**: Forcing information through a concept bottleneck leads to information loss, often resulting in lower accuracy than end-to-end black-box models. This occurs because the concept layer may encode redundant information irrelevant to the task while losing critical task-related information.

**Concept Leakage**: Extra information unrelated to concept definitions is mixed into concept representations. While this "leaked" information might temporarily boost accuracy, it undermines the faithfulness of the concept layer, making concept intervention unreliable—modifying one concept value may cause unpredictable chain reactions.

The **Key Challenge** lies in the fact that concept layer encoding is neither "pure" enough (due to leakage) nor "sufficient" enough (due to task information loss).

The **Key Insight** of this work is that this contradiction can be resolved using Information Bottleneck (IB) theory. The goal of the IB principle is to learn a minimal sufficient statistic of the input $X$—in the context of CBMs, this means ensuring the concept representation $C$ retains only the minimum information necessary to predict $Y$ while compressing task-irrelevant redundancy.

## Method

### Overall Architecture

Ours does not modify the CBM network structure or require new annotations; it solely incorporates an Information Bottleneck regularization term into the training objective. The data flow of a CBM is $X \to Z \to C \to Y$: the input $X$ is mapped to a latent representation $Z$ via an encoder, then the human-readable concept layer $C$ is predicted from $Z$, and finally the label $Y$ is derived from $C$. Standard CBMs only force $C$ to predict labels accurately but allow concept-irrelevant details from the input to leak into $C$ via $Z \to C$ (concept leakage), damaging both interpretability and intervention reliability. The **Mechanism** involves applying the IB principle directly to the concept layer: retaining $I(Z;C)$ and $I(C;Y)$ while penalizing $I(X;C)$, forcing the concept layer to carry only "sufficient and clean" information. The objective is formulated as $\mathcal{L}_{CIBM}=I(Z;C)+I(C;Y)-\beta\,I(X;C)$, where $\beta$ is a Lagrange multiplier controlling compression strength. Since $I(X;C)$ cannot be directly computed in high-dimensional spaces, the authors provide two trainable implementations: the variational upper bound version (IBB) and the estimator-based proxy version (IBE).

### Key Designs

**1. Applying IB to the concept layer rather than the latent layer: Enforcing "purity" constraints on the actual interpretable layer**

Classic IB (Tishby 2000; Alemi 2017) compresses the latent representation $Z$ via $I(X;Z)$. According to the data processing inequality $I(X;C) \le I(X;Z)$, compressing $Z$ indirectly restricts information in $C$. However, the authors argue this is merely an "upper bound side effect": if $X \to Z$ is compressed first and then $C$ is derived from $Z$, leakage can still survive the $Z \to C$ step. Thus, this work places the constraint **directly** on the concept layer, minimizing $I(X;C)$ instead of $I(X;Z)$, resulting in the objective:

$$\mathcal{L}_{CIBM}=I(Z;C)+I(C;Y)-\beta\,I(X;C).$$

This is an intentional design choice rather than a fallback approximation: regardless of the capacity of the latent layer $Z$, it strictly controls how much source information enters $C$, prioritizing the "purity of the interpretable layer." This distinguishes the work from prior IB applications on general latent features and is the reason it yields more faithful and interventible concepts.

**2. IBB: Converting the objective into an optimizable variational lower bound of cross-entropy**

Mutual information terms like $I(X;C)$ contain marginal terms that cannot be estimated directly. The authors apply a variational approximation to the data distribution, bounding $\mathcal{L}_{CIBM}$ with a series of entropy/cross-entropy terms:

$$\mathcal{L}_{CIBM}\ge(1-\beta)\,\mathbb{E}_{p(z)}\!\big[H(p(c\mid z))-H(p(c\mid z),q(c\mid z))\big]-\mathbb{E}_{p(c)}H(p(y\mid c),q(y\mid c)).$$

Maximizing this bound is equivalent to minimizing the cross-entropy of concepts $c$ and labels $y$ relative to ground truth, while adjusting the entropy of the concept distribution. This maps abstract mutual information optimization to a standard, back-propagatable training loss; the cost is the additional estimation of the entropy of the concept distribution $p(c)$. Models trained with this objective are denoted as **IBB** (Bounded CIB).

**3. IBE: Treating entropy as a constant for a more efficient mutual information estimator**

IBB still requires estimating concept entropy. The authors offer a lightweight alternative: expanding only the conditional entropies that are not marginalized out and treating concept entropy $H(C)$ and label entropy $H(Y)$ as constants, yielding:

$$\mathcal{L}_{E\text{-}CIB}=\mathbb{E}_{p(c)}H(p(y\mid c),q(y\mid c))+\mathbb{E}_{p(z)}H(p(c\mid z),q(c\mid z))-\beta\big(\rho-I(X;C)\big),$$

where $\rho$ is a constant and $I(X;C)$ is provided directly by a mutual information estimator. This version avoids the overhead of concept entropy estimation and is more efficient. Formally, it is isomorphic to the latent IB loss of Kawaguchi et al. (2023), but shifts the condition from the latent layer to the concept layer. Models trained with this objective are denoted as **IBE** (Estimator-based CIB), performing comparably to IBB in experiments.

Both implementations are simply losses appended to the original training objective without touching the forward architecture. Thus, they can be layered onto various training paradigms (joint/sequential/independent) and concept embedding families like CEM or ProbCBM.

### Loss & Training

The final training loss is either $\mathcal{L}_{S\text{-}CIBM}$ (IBB) or $\mathcal{L}_{E\text{-}CIB}$ (IBE), both of which integrate the cross-entropy for predicting concepts $c$ and labels $y$ with the IB compression term. The compression strength $\beta$ is the critical hyperparameter: if too small, compression is insufficient and leakage persists; if too large, task-relevant information is discarded, causing accuracy to drop. The paper searches for the optimal $\beta$ balance between "compression and retention" on a validation set. Through PAC-Bayes analysis (Theorem 2), it is proven that as long as $\beta$ is small enough to keep the generalization gap $\Delta > 0$, the true risk upper bound of CIBM is strictly tighter than that of a standard CBM—the reduction in complexity outweighs the slight increase in training error from the $\beta$ penalty.

## Key Experimental Results

### Main Results

The paper evaluates six CBM families across three benchmark datasets:

| CBM Variant | Dataset | Without IB | +IB | Change |
|-------------|---------|------------|-----|--------|
| Joint CBM | CUB-200 | Baseline | Gain | ✓ Consistent Improvement |
| Sequential CBM | CUB-200 | Baseline | Gain | ✓ Consistent Improvement |
| Independent CBM | CUB-200 | Baseline | Gain | ✓ Consistent Improvement |
| CEM | CUB-200 | Baseline | Gain | ✓ Consistent Improvement |
| CBM-AUC | CUB-200 | Baseline | Gain | ✓ Consistent Improvement |
| ProbCBM | CUB-200 | Baseline | Gain | ✓ Consistent Improvement |

On all six CBM families and across three benchmarks, the IB-regularized versions consistently outperform the corresponding original versions.

### Ablation Study

| Configuration | Key Metrics | Description |
|---------------|-------------|-------------|
| No IB Regularization (Vanilla) | Baseline | Standard CBM training |
| Variational IB ($\beta$ = small) | Slight Gain | Gentle compression |
| Variational IB ($\beta$ = medium) | Best | Optimal compression-retention balance |
| Variational IB ($\beta$ = large) | Drop | Over-compression |
| Entropy-based Proxy | Comparable to IBB | Simpler, no extra parameters |

### Key Findings
- IB regularization brought consistent gains to **all** tested CBM variants, demonstrating strong generalization.
- Information Plane analysis confirmed that IB regularization indeed compresses $I(X;C)$ while maintaining $I(C;Y)$.
- Concept intervention (TTI) experiments show the IB-regularized versions are more predictable and reliable in their response to interventions.
- This method addresses previous inconsistencies in CBM evaluations by demonstrating robust gains across a unified training protocol.

## Highlights & Insights
- **Theoretical Elegance**: Unifies empirical CBM issues (concept leakage, accuracy drop) within an information-theoretic framework, naturally providing a solution via the IB principle.
- **Architecture-Agnostic**: As a pure regularization method, it can be applied in a plug-and-play manner to any existing CBM variant.
- **Dual Benefit**: Simultaneously improves predictive accuracy and concept layer faithfulness, breaking the common trade-off between "accuracy vs. interpretability."
- **Information Plane Validation**: Visually demonstrates the effects of regularization through information plane analysis, increasing the credibility of the method.

## Limitations & Future Work
- The hyperparameter $\beta$ for IB regularization requires careful tuning; different datasets and CBM variants may require different optimal values.
- Variational methods require Gaussian assumptions for marginal distributions, which may be inflexible in some scenarios.
- The paper primarily validates on small-to-medium scale vision classification tasks; performance on large-scale and non-visual tasks remains to be explored.
- The cost of obtaining concept annotations remains a universal bottleneck for CBM methods.

## Related Work & Insights
- **vs. Standard CBM (Koh et al., 2020)**: Standard CBMs do not constrain concept layer information, leading to concept leakage. IB regularization provides a principled solution.
- **vs. CEM (Zarlenga et al., 2022)**: CEM increases concept layer capacity via concept embeddings but lacks information compression constraints. IB regularization can be layered on top for further improvement.
- **vs. Deep VIB (Alemi et al., 2017)**: While Deep VIB applies IB to general classification, this work specializes it for the CBM concept layer, utilizing structural properties to design more effective regularization.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Applying Information Bottleneck to CBMs is natural and elegant, though the core technique (VIB) has precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across six CBM variants and three benchmarks; information plane analysis adds credibility.
- **Writing Quality**: ⭐⭐⭐⭐ Clear theoretical derivations and standardized experimental settings.
- **Value**: ⭐⭐⭐⭐ Provides a simple and effective universal tool for the CBM community; plug-and-play characteristics offer high utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] There Was Never a Bottleneck in Concept Bottleneck Models](there_was_never_a_bottleneck_in_concept_bottleneck_models.md)
- [\[AAAI 2026\] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](../../AAAI2026/interpretability/concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)
- [\[ICLR 2026\] Information Shapes Koopman Representation](information_shapes_koopman_representation.md)
- [\[ICLR 2026\] Debugging Concept Bottleneck Models through Removal and Retraining](debugging_concept_bottleneck_models_through_removal_and_retraining.md)
- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](../../AAAI2026/interpretability/partially_shared_concept_bottleneck_models.md)

</div>

<!-- RELATED:END -->
