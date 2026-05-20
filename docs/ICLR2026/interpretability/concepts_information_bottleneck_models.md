---
title: >-
  [Paper Note] Concepts' Information Bottleneck Models
description: >-
  [ICLR 2026][Interpretability][Concept Bottleneck Models] This paper introduces Information Bottleneck (IB) regularization into the concept layer of Concept Bottleneck Models (CBMs)…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Concept Bottleneck Models"
  - "Information Bottleneck"
  - "Regularization"
  - "Concept Leakage"
date: 2026-05-08
content_hash: 28cea4c942ce1701
---

# Concepts' Information Bottleneck Models

**Conference**: ICLR 2026
**arXiv**: [2602.14626](https://arxiv.org/abs/2602.14626)  
**Code**: Available (mentioned in the paper)  
**Area**: Interpretability
**Keywords**: Concept Bottleneck Models, Information Bottleneck, Interpretability, Regularization, Concept Leakage

## TL;DR
This paper introduces Information Bottleneck (IB) regularization into the concept layer of Concept Bottleneck Models (CBMs), learning minimal sufficient concept representations by penalizing $I(X;C)$ while preserving $I(C;Y)$. The approach consistently improves both predictive performance and concept intervention reliability across six CBM variants and three benchmarks.

## Background & Motivation

Concept Bottleneck Models (CBMs) are a class of interpretable AI methods that insert a human-understandable concept layer $C$ between input $X$ and prediction $Y$, making the decision process transparent and interpretable. This design enables domain experts to intervene on concept values at inference time (concept intervention), thereby correcting erroneous model reasoning.

However, existing CBMs suffer from two fundamental problems:

**Accuracy degradation**: Forcing information through the concept bottleneck causes information loss, and model accuracy is often lower than that of end-to-end black-box models. This occurs because the concept layer may encode redundant information irrelevant to the task while discarding some task-relevant information.

**Concept Leakage**: Concept representations encode extraneous information unrelated to the concept definitions. Although such "leaked" information may improve accuracy in the short term, it undermines the faithfulness of the concept layer and renders concept interventions unreliable—modifying the value of one concept may trigger unpredictable cascading effects.

The root cause of these two problems is that the information encoded in the concept layer is neither sufficiently "pure" (due to leakage) nor sufficiently "complete" (due to information loss).

The core insight of this paper is that this tension can be naturally resolved through Information Bottleneck (IB) theory. The IB principle aims to learn a minimal sufficient statistic of the input $X$—in the context of the concept layer, this means that the concept representation $C$ should retain only the minimum information necessary to predict $Y$, while compressing away task-irrelevant redundancy.

## Method

### Overall Architecture

The paper introduces an additional IB regularization term into the standard CBM training pipeline without modifying the model architecture or requiring extra supervision signals. The standard CBM training objective minimizes a weighted sum of concept prediction loss and task prediction loss; this work augments that objective with a penalty term constraining $I(X;C)$:

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_{c} \mathcal{L}_{concept} + \beta \cdot R_{IB}$$

where $R_{IB}$ is the IB regularization term and $\beta$ controls the compression strength. The key principle is to preserve (or even enhance) the mutual information from the concept layer to the label $I(C;Y)$, while compressing the mutual information from input to concept layer $I(X;C)$.

### Key Designs

1. **Variational IB**: Direct computation of $I(X;C)$ is intractable; hence a variational upper bound is employed. A learnable marginal distribution $q(C)$ is introduced to approximate the true marginal $p(C)$, and the KL divergence $\mathrm{KL}[p(C|X) \| q(C)]$ serves as an upper bound on $I(X;C)$. In practice, $q(C)$ is parameterized as a multivariate Gaussian with learnable mean and variance. This approach is theoretically principled and directly optimizes a surrogate for the mutual information.

2. **Entropy-based Surrogate**: While the variational approach is theoretically elegant, it introduces additional parameters for the marginal distribution. As an alternative, the paper proposes a simpler entropy-based surrogate: directly minimizing an estimate of the conditional entropy of the concept layer output. This requires no additional learnable parameters, incurs lower computational overhead, and is more suitable for large-scale applications. The core idea is to encourage the concept layer output distribution to be more concentrated, reducing unnecessary information encoding.

3. **Architecture-agnostic Integration**: Both regularization methods are incorporated as additional loss terms in standard CBM training, requiring no modification to the network architecture. This enables direct application to existing CBM variants, including jointly trained, sequentially trained, and independently trained models.

### Loss & Training

The total training objective comprises three components:
- **Task loss**: Cross-entropy loss for predicting the target label $Y$
- **Concept prediction loss**: Binary cross-entropy for predicting concepts $C$ from input $X$
- **IB regularization term**: Either the variational KL divergence term or the entropy-based surrogate, controlling the degree of information compression in the concept layer

The choice of hyperparameter $\beta$ is critical: too small a value renders the regularization ineffective, while too large a value leads to excessive compression and information loss. The optimal $\beta$ is selected via a search over the validation set.

## Key Experimental Results

### Main Results

The paper evaluates six CBM families on three benchmark datasets:

| CBM Variant | Dataset | w/o IB | +IB | Change |
|-------------|---------|--------|-----|--------|
| Joint CBM | CUB-200 | Baseline | Improved | ✓ Consistent gain |
| Sequential CBM | CUB-200 | Baseline | Improved | ✓ Consistent gain |
| Independent CBM | CUB-200 | Baseline | Improved | ✓ Consistent gain |
| CEM | CUB-200 | Baseline | Improved | ✓ Consistent gain |
| CBM-AUC | CUB-200 | Baseline | Improved | ✓ Consistent gain |
| ProbCBM | CUB-200 | Baseline | Improved | ✓ Consistent gain |

IB-regularized variants consistently outperform their respective baselines across all six CBM families and all three benchmarks.

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| No IB regularization (Vanilla) | Baseline | Standard CBM training |
| Variational IB ($\beta$ = small) | Slight improvement | Mild compression |
| Variational IB ($\beta$ = medium) | Best | Optimal compression–retention balance |
| Variational IB ($\beta$ = large) | Degraded | Excessive compression |
| Entropy-based surrogate | Comparable to variational IB | Simpler, no extra parameters |

### Key Findings
- IB regularization yields consistent improvements across **all** tested CBM variants, demonstrating strong generalizability.
- Information plane analysis confirms that IB regularization effectively compresses $I(X;C)$ while preserving $I(C;Y)$.
- Test-time intervention (TTI) experiments show that IB-regularized models respond to concept interventions in a more predictable and reliable manner.
- The method resolves inconsistencies observed in prior CBM evaluations and demonstrates robust gains under a unified training protocol.

## Highlights & Insights
- **Theoretical Elegance**: The paper unifies the empirical problems of CBMs (concept leakage, accuracy degradation) within an information-theoretic framework and derives a natural solution via the IB principle.
- **Architecture-agnostic**: As a purely regularization-based method, it can be applied in a plug-and-play fashion to any existing CBM variant.
- **Dual Benefits**: The approach simultaneously improves predictive accuracy and enhances the faithfulness of the concept layer, breaking the commonly observed accuracy–interpretability trade-off.
- **Information Plane Validation**: Information plane analysis provides intuitive evidence of the regularization's effect, strengthening the credibility of the method.

## Limitations & Future Work
- The hyperparameter $\beta$ requires careful tuning; different datasets and CBM variants may demand different optimal values.
- The variational approach relies on a Gaussian assumption for the marginal distribution, which may lack flexibility in certain settings.
- Validation is primarily conducted on small-to-medium-scale visual classification tasks; effectiveness on large-scale and non-visual tasks remains to be explored.
- The cost of obtaining concept annotations remains a general bottleneck for CBM-based methods.

## Related Work & Insights
- **vs. Standard CBM (Koh et al., 2020)**: Standard CBMs impose no constraint on the information content of the concept layer, making them susceptible to concept leakage; IB regularization provides a principled solution.
- **vs. CEM (Zarlenga et al., 2022)**: CEM enhances the expressiveness of the concept layer via concept embeddings but lacks an information compression constraint; IB regularization can be layered on top to yield further improvements.
- **vs. Deep VIB (Alemi et al., 2017)**: Deep VIB applies IB to general classification; this paper specializes the framework to the CBM concept layer, exploiting its structural properties to design more effective regularization.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing information bottleneck into CBMs is natural and elegant, though the core technique (VIB) has prior precedent.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across six CBM variants and three benchmarks; information plane analysis adds credibility.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear and experimental setup is rigorous.
- Value: ⭐⭐⭐⭐ Provides the CBM community with a simple, effective, and universally applicable improvement tool with strong practical utility.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] There Was Never a Bottleneck in Concept Bottleneck Models](there_was_never_a_bottleneck_in_concept_bottleneck_models.md)
- [\[AAAI 2026\] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](../../AAAI2026/interpretability/concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)
- [\[ICLR 2026\] Information Shapes Koopman Representation](information_shapes_koopman_representation.md)
- [\[CVPR 2026\] Towards Faithful Multimodal Concept Bottleneck Models](../../CVPR2026/interpretability/towards_faithful_multimodal_concept_bottleneck_models.md)
- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](../../AAAI2026/interpretability/partially_shared_concept_bottleneck_models.md)

</div>

<!-- RELATED:END -->
